"""Game24 ICRL loop with the original four-part reward rubric."""

import ast
import csv
import json
import re
from pathlib import Path

from .client import judge_options
from .feedback import format_feedback
from .reward_polarity import shown_reward

ROOT = Path(__file__).resolve().parents[1]
ONE_SHOT = ROOT / "experiments" / "game24" / "game24_3steps.txt"
_STEP = re.compile(r"Step\s*([123])\s*:\s*(.*?)(?=Step\s*[123]\s*:|\*\*Answer\*\*\s*:|$)", re.I | re.S)
_ANSWER = re.compile(r"\*\*Answer\*\*\s*:\s*([^\n]+)", re.I)
_JUDGE_ANSWER = re.compile(r"(?:\*\*)?Answer(?:\*\*)?\s*:\s*([+-]?\d+)\b", re.I)


def load_puzzles(path: Path, start: int, count: int) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Game24 puzzle file not found: {path}")
    puzzles = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            for field in row:
                numbers = field.strip().split()
                if len(numbers) == 4 and all(re.fullmatch(r"\d+", n) for n in numbers):
                    puzzles.append(" ".join(numbers))
                    break
    chosen = puzzles[start : start + count]
    if len(chosen) != count:
        raise ValueError(f"Requested {count} puzzles from offset {start}; file contains {len(puzzles)}")
    return chosen


def _safe_arithmetic(expression: str):
    """Evaluate only arithmetic, avoiding sympify on model-generated code."""
    operands = []

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            operands.append(node.value)
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("Expression contains unsupported syntax")

    result = visit(ast.parse(expression, mode="eval"))
    return result, operands


def diagnostics(puzzle: str, answer: str) -> dict:
    try:
        expression = answer.split("=", 1)[0].strip()
        value, operands = _safe_arithmetic(expression)
        expected = sorted(int(x) for x in puzzle.split())
        used = sorted(int(x) for x in operands)
        return {
            "accuracy_reward": -abs(float(value) - 24),
            "format_reward": 0 if used == expected and len(operands) == 4 else -30,
        }
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError):
        return {"accuracy_reward": -24.0, "format_reward": -30}


def _one_shot_with_feedback(policy: str = "legacy") -> str:
    source = ONE_SHOT.read_text(encoding="utf-8")
    if policy == "unscored":
        return re.sub(r"(?m)^\*\*Reward\*\*:[ \t]*-?\d+[ \t]*\n", "", source)
    if policy != "legacy":
        raise ValueError(f"Unknown Game24 demo policy: {policy}")
    return re.sub(
        r"(?m)(\*\*Reward\*\*:\s*)(-?\d+)(?=\s*$)",
        lambda match: match.group(1) + format_feedback(int(match.group(2))),
        source,
    )


def build_prompt(puzzle: str, history: list[dict], round_idx: int,
                 history_limit: int | None, demo_policy: str = "legacy") -> str:
    selected = history if history_limit is None else history[-history_limit:] if history_limit else []
    parts = [_one_shot_with_feedback(demo_policy)]
    for attempt in selected:
        scores = attempt.get("shown_step_rewards", attempt["step_rewards"])
        annotated = attempt["response"]
        # The original prompt gives feedback after each intermediate step.
        for index in (1, 2, 3):
            marker = re.compile(rf"(?=Step\s*{index + 1}\s*:|\*\*Answer\*\*\s*:)", re.I)
            match = marker.search(annotated)
            if match:
                annotated = annotated[: match.start()] + f"<Reward: {format_feedback(scores[index - 1])}> " + annotated[match.start() :]
        parts.append(
            f"<attempt>\n**Prompt**: Input: {puzzle}\n{annotated}\n"
            f"**Reward**: {format_feedback(attempt.get('shown_reward', attempt['reward']))}\n</attempt>"
        )
    instruction = (
        "Examine previous attempts and try steps different from them."
        if round_idx % 2 == 0
        else "Use previous attempts and their rewards to improve the solution."
    )
    parts.append(
        f"{instruction} Make exactly one attempt. Use all four input numbers once with + - * / to obtain 24. "
        "Write <answer>**Response** Step 1: ... (left: ...) Step 2: ... (left: ...) "
        "Step 3: ... (left: ...) **Answer**: <expression> = 24</answer>.\n"
        f"**Prompt**: Input: {puzzle}"
    )
    return "\n".join(parts)


def _judge_prompt(puzzle: str, part: str, index: int) -> str:
    rule = "Use all four input numbers once and + - * / to obtain 24 in three steps."
    if index < 3:
        return (
            f"Rule of Game of 24: {rule}\nInput: {puzzle}\nStep {index + 1}: {part}\n"
            "Look at the numbers in the left state and decide whether 24 is reachable: "
            "sure = 3, likely = 1, impossible or invalid = 0. "
            "Return exactly **Answer**: <integer score>."
        )
    return (
        f"Rule of Game of 24: {rule}\nInput: {puzzle}\nSolution: {part}\n"
        "Return 10 if it validly uses exactly the four input numbers and equals 24; otherwise return 0. "
        "Return exactly **Answer**: <integer score>."
    )


def evaluate(
    puzzle: str, response: str, judge, judge_model: str, log, *,
    sample_id: int, round_idx: int, judge_max_tokens: int = 1024,
    saved_steps: list[dict] | None = None,
    reward_polarity: str = "normal",
) -> dict:
    saved_steps = saved_steps or []
    block_match = re.search(r"<answer>(.*?)</answer>", response, re.I | re.S)
    block = block_match.group(1) if block_match else ""
    step_matches = {int(m.group(1)): m.group(2).strip() for m in _STEP.finditer(block)}
    answer_match = _ANSWER.search(block)
    answer = answer_match.group(1).strip() if answer_match else ""
    if not block_match or any(not step_matches.get(i) for i in (1, 2, 3)) or not answer:
        # Invalid model formatting is a task failure, not a judge/API failure.
        if any(event.get("status") != "invalid_generation" for event in saved_steps):
            raise ValueError("Saved judge scores do not match the invalid generation")
        for index in range(len(saved_steps), 4):
            log.emit(
                "eval_step", sample_id=sample_id, round=round_idx, step=index + 1,
                judge_prompt="", judge_response="", parsed_score=None, reward=0,
                shown_reward=0, display_reward=format_feedback(0), status="invalid_generation",
            )
        return {"response": block if block_match else response, "answer": "", "step_rewards": [0, 0, 0, 0],
            "shown_step_rewards": [0, 0, 0, 0], "step_statuses": ["invalid_generation"] * 4,
            "reward": 0, "shown_reward": 0, "eval_status": "invalid_generation", **diagnostics(puzzle, "")}
    if any(event.get("status") == "invalid_generation" for event in saved_steps):
        raise ValueError("Saved invalid-generation scores do not match the generated answer")
    parts = [step_matches.get(i, "") for i in (1, 2, 3)] + [answer]
    scores = [event["reward"] for event in saved_steps]
    statuses = [event["status"] for event in saved_steps]
    shown_scores = [shown_reward("game24", event["reward"], reward_polarity,
                                 step=event["step"], status=event["status"]) for event in saved_steps]
    for index in range(len(saved_steps), 4):
        part = parts[index]
        prompt = _judge_prompt(puzzle, part, index)
        raw = judge.generate(
            judge_model, prompt, temperature=0, max_tokens=judge_max_tokens,
            retry_empty_budgets=(judge_max_tokens * 2, judge_max_tokens * 4),
            on_response=lambda report, step=index + 1: log.emit(
                "judge_call", sample_id=sample_id, round=round_idx, step=step, **report,
            ),
            **judge_options(judge_model),
        )
        match = _JUDGE_ANSWER.search(raw)
        parsed = int(match.group(1)) if match else None
        valid = parsed in ((0, 1, 3) if index < 3 else (0, 10))
        score = parsed if valid else 0
        status = "ok" if valid else "invalid_judge_score"
        shown = shown_reward("game24", score, reward_polarity, step=index + 1, status=status)
        scores.append(score)
        statuses.append(status)
        shown_scores.append(shown)
        log.emit(
            "eval_step", sample_id=sample_id, round=round_idx, step=index + 1,
            judge_prompt=prompt, judge_response=raw, parsed_score=parsed,
            reward=score, shown_reward=shown, display_reward=format_feedback(shown), status=status,
        )
    return {"response": block, "answer": answer, "step_rewards": scores,
            "shown_step_rewards": shown_scores, "step_statuses": statuses,
            "reward": sum(scores), "shown_reward": sum(shown_scores),
            "eval_status": "ok", **diagnostics(puzzle, answer)}


def _resume_state(path: Path, sample_count: int, round_count: int):
    """Rebuild completed per-puzzle histories from the append-only event log."""
    generations, completed, round_ends, evaluated = {}, {}, set(), {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event.get("event")
            if kind not in ("generation", "eval_step", "attempt_end", "round_end"):
                continue
            if kind == "round_end":
                round_ends.add(event["round"])
                continue
            key = (event["round"], event["sample_id"])
            if not (0 <= key[0] < round_count and 0 <= key[1] < sample_count):
                raise ValueError(f"Out-of-range event in resume log: {key}")
            if kind == "eval_step":
                evaluated.setdefault(key, []).append(event)
            elif kind == "generation":
                if key in generations:
                    raise ValueError(f"Duplicate generation in resume log: {key}")
                generations[key] = event["model_response"]
            else:
                if key in completed or key not in generations:
                    raise ValueError(f"Duplicate or unpaired completed attempt: {key}")
                block_match = re.search(r"<answer>(.*?)</answer>", generations[key], re.I | re.S)
                completed[key] = {
                    "response": block_match.group(1) if block_match else generations[key],
                    "answer": event["answer"], "step_rewards": event["step_rewards"],
                    "reward": event["reward"],
                    "shown_step_rewards": event.get("shown_step_rewards", event["step_rewards"]),
                    "shown_reward": event.get("shown_reward", event["reward"]),
                }
    if completed:
        last_index = max(round_idx * sample_count + sample_id for round_idx, sample_id in completed)
        for index in range(last_index + 1):
            if divmod(index, sample_count) not in completed:
                raise ValueError("Resume log has a gap before its last completed attempt")
    for round_idx in round_ends:
        if any((round_idx, sample_id) not in completed for sample_id in range(sample_count)):
            raise ValueError(f"Round {round_idx} ended without all samples")
    histories = [[] for _ in range(sample_count)]
    for (_, sample_id), attempt in sorted(completed.items()):
        histories[sample_id].append(attempt)
    if any(key not in generations for key in evaluated):
        raise ValueError("Resume log has step evaluations without a generation")
    pending = {}
    for key, response in generations.items():
        if key in completed:
            continue
        steps = evaluated.get(key, [])
        if [event.get("step") for event in steps] != list(range(1, len(steps) + 1)) or len(steps) > 4:
            raise ValueError(f"Incomplete attempt has duplicate or non-contiguous step evaluations: {key}")
        for event in steps:
            allowed = (0, 1, 3) if event["step"] < 4 else (0, 10)
            if type(event.get("reward")) is not int or event["reward"] not in allowed:
                raise ValueError(f"Incomplete attempt has an invalid saved step reward: {key}")
            if event.get("status") not in ("ok", "invalid_judge_score", "invalid_generation"):
                raise ValueError(f"Incomplete attempt has an invalid saved step status: {key}")
            if event["status"] != "ok" and event["reward"] != 0:
                raise ValueError(f"Incomplete attempt has a nonzero failed-step reward: {key}")
        pending[key] = {"response": response, "saved_steps": steps}
    if len(pending) > 1:
        raise ValueError("Resume log has multiple uncompleted generations")
    return histories, completed, round_ends, pending


def run(settings: dict, client, log):
    reward_polarity = settings.get("reward_polarity", "normal")
    puzzles = load_puzzles(Path(settings["game24_data"]), settings["start_index"], settings["samples"])
    if settings.get("_resume_path"):
        histories, completed, round_ends, pending = _resume_state(
            Path(settings["_resume_path"]) / "events.jsonl", len(puzzles), settings["rounds"],
        )
        log.emit(
            "resume_state", completed_attempts=len(completed), pending_generations=len(pending),
            saved_step_evaluations=sum(len(item["saved_steps"]) for item in pending.values()),
        )
    else:
        histories, completed, round_ends, pending = [[] for _ in puzzles], {}, set(), {}
    for round_idx in range(settings["rounds"]):
        rewards = []
        for sample_id, puzzle in enumerate(puzzles):
            key = (round_idx, sample_id)
            if key in completed:
                rewards.append(completed[key]["reward"])
                continue
            prompt = build_prompt(puzzle, histories[sample_id], round_idx, settings["history"],
                                  settings.get("game24_demo_policy", "legacy"))
            saved = pending.pop(key, None)
            if saved is None:
                output = client.generate(
                    settings["model"], prompt, temperature=0.6, max_tokens=settings["max_tokens"],
                    on_response=lambda report: log.emit(
                        "generation_call", sample_id=sample_id, round=round_idx, **report,
                    ),
                )
                log.emit("generation", sample_id=sample_id, round=round_idx, prompt=prompt, model_response=output)
                saved_steps = []
            else:
                output, saved_steps = saved["response"], saved["saved_steps"]
                log.emit(
                    "reuse_generation", sample_id=sample_id, round=round_idx,
                    saved_step_evaluations=len(saved_steps),
                )
            result = evaluate(
                puzzle, output, client, settings["judge_model"], log,
                sample_id=sample_id, round_idx=round_idx,
                judge_max_tokens=settings["judge_max_tokens"], saved_steps=saved_steps,
                reward_polarity=reward_polarity,
            )
            histories[sample_id].append(result)
            rewards.append(result["reward"])
            log.emit(
                "attempt_end", sample_id=sample_id, round=round_idx, puzzle=puzzle,
                reward=result["reward"], shown_reward=result["shown_reward"],
                display_reward=format_feedback(result["shown_reward"]),
                step_rewards=result["step_rewards"], accuracy_reward=result["accuracy_reward"],
                shown_step_rewards=result["shown_step_rewards"], step_statuses=result["step_statuses"],
                format_reward=result["format_reward"], answer=result["answer"],
                eval_status=result["eval_status"],
            )
        if round_idx not in round_ends:
            log.emit("round_end", round=round_idx, mean_reward=sum(rewards) / len(rewards), count=len(rewards))
    return {"rounds_completed": settings["rounds"], "samples": len(puzzles)}
