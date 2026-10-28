"""Creative Writing ICRL loop with the original coherence rubric."""

import re
from pathlib import Path

from .client import judge_options
from .feedback import format_feedback
from .reward_polarity import shown_reward

BASE_PASSAGE = (
    "At dawn, golden light slips past pale curtains, rousing the world in quiet celebration. "
    "A lone robin greets the morning with a clear, cheerful trill, its song drifting across dew-laden grass. "
    "A gentle breeze stirs the leaves, carrying the fresh, earthy scent of new growth. "
    "Nearby, rooftops and empty streets lie poised between night’s calm and the city’s stirring pulse, "
    "promising simple comforts like a warm cup of coffee. In this tranquil pause, one senses life’s "
    "renewal and the gentle invitation to greet the day with hope and gratitude."
)
_COHERENCE = re.compile(r"Coherency score:\s*(10|[1-9])\b", re.I)
COT_PROMPT = """Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {ending}

Make a plan then write. Your output should be of the following format:

Plan:
Your plan here.

Passage:
Your passage here."""


def load_prompts(path: Path, count: int) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Creative Writing prompt file not found: {path}")
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) < count:
        raise ValueError(f"Requested {count} prompts; file contains {len(prompts)}")
    return prompts[:count]


def build_prompt(ending: str, history: list[dict], round_idx: int, history_limit: int | None) -> str:
    selected = history if history_limit is None else history[-history_limit:] if history_limit else []
    question = COT_PROMPT.format(ending=ending)
    parts = []
    for attempt in selected:
        reward = format_feedback(attempt.get("shown_reward", attempt["reward"]))
        parts.append(
            f"<attempt>\n**Prompt**: {question}\n**Reward**: {reward}\n"
            f"{attempt['answer']}\n**Reward**: {reward}\n</attempt>"
        )
    instruction = (
        "Instruction: Examine all the <attempt> examples, each showing a candidate Response and its Reward. "
        "Provide a response that is different from previous attempts and wrap it in "
        "<answer>Plan:...Passage:...</answer>."
        if round_idx % 2 == 0
        else "Instruction: Based on the previous <attempt> entries and their rewards, try to produce "
        "a response with a higher reward, following the task and using "
        "<answer>Plan:...Passage:...</answer>."
    )
    parts.append(f"{instruction}\n**Prompt**: {question}")
    return "\n".join(parts)


def evaluate(
    output: str, judge, judge_model: str, log, *,
    sample_id: int, round_idx: int, judge_max_tokens: int = 1024,
    reward_polarity: str = "normal", seed_policy: str = "fixed",
) -> dict:
    match = re.search(r"<answer>(.*?)</answer>", output, re.I | re.S)
    answer = match.group(1).strip() if match else ""
    passage = answer.split("Passage:", 1)[-1].strip() if "Passage:" in answer else answer
    if round_idx == 0 and seed_policy == "fixed":
        reward, raw, prompt, status = 1, "", "", "seed_round"
    else:
        prompt = (
            "Instruction: You are a seasoned text coherence evaluator. Read the TEXT below and "
            "rate the TEXT's overall coherency on a 1-to-10 scale, where 1 means less coherent "
            "than the Base Answer, 5 means as coherent as the Base Answer, and 10 means way more "
            "coherent than the Base Answer. Be a strict and conservative evaluator and only give a "
            "high score when the TEXT is truly better than the Base Answer. "
            f"Base Answer: <<<{BASE_PASSAGE}>>> TEXT:<<<{passage}>>> "
            "Return your answer in exactly this format: Coherency score: <integer 1-10>.\nResponse:"
        )
        raw = judge.generate(
            judge_model, prompt, temperature=0, max_tokens=judge_max_tokens,
            retry_empty_budgets=(judge_max_tokens * 2, judge_max_tokens * 4),
            on_response=lambda report: log.emit(
                "judge_call", sample_id=sample_id, round=round_idx, step=1, **report,
            ),
            **judge_options(judge_model),
        )
        parsed = _COHERENCE.search(raw)
        reward = int(parsed.group(1)) if parsed else 0
        status = "ok" if parsed else "invalid_judge_score"
    shown = shown_reward("creative_writing", reward, reward_polarity, status=status)
    log.emit(
        "eval_step", sample_id=sample_id, round=round_idx, step=1,
        judge_prompt=prompt, judge_response=raw, reward=reward,
        shown_reward=shown, display_reward=format_feedback(shown), status=status,
    )
    return {"answer": answer, "passage": passage, "reward": reward,
            "shown_reward": shown, "eval_status": status}


def run(settings: dict, client, log):
    prompts = load_prompts(Path(settings["writing_data"]), settings["samples"])
    histories = [[] for _ in prompts]
    for round_idx in range(settings["rounds"]):
        rewards = []
        for sample_id, ending in enumerate(prompts):
            prompt = build_prompt(ending, histories[sample_id], round_idx, settings["history"])
            output = client.generate(
                settings["model"], prompt, temperature=0.6, max_tokens=settings["max_tokens"],
                on_response=lambda report: log.emit(
                    "generation_call", sample_id=sample_id, round=round_idx, **report,
                ),
            )
            log.emit("generation", sample_id=sample_id, round=round_idx, prompt=prompt, model_response=output)
            result = evaluate(
                output, client, settings["judge_model"], log,
                sample_id=sample_id, round_idx=round_idx,
                judge_max_tokens=settings["judge_max_tokens"],
                reward_polarity=settings.get("reward_polarity", "normal"),
                seed_policy=settings.get("writing_seed_policy", "fixed"),
            )
            histories[sample_id].append(result)
            rewards.append(result["reward"])
            log.emit(
                "attempt_end", sample_id=sample_id, round=round_idx,
                reward=result["reward"], shown_reward=result["shown_reward"],
                display_reward=format_feedback(result["shown_reward"]),
                answer=result["answer"], eval_status=result["eval_status"],
            )
        log.emit("round_end", round=round_idx, mean_reward=sum(rewards) / len(rewards), count=len(rewards))
    return {"rounds_completed": settings["rounds"], "samples": len(prompts)}
