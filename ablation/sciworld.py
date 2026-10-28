"""ScienceWorld loop reusing the repository's action wrapper and score patch."""

import copy
import sys
from pathlib import Path

from .feedback import format_feedback
from .reward_polarity import shown_reward

ROOT = Path(__file__).resolve().parents[1]
SCI_DIR = ROOT / "experiments" / "sciworld"

_LEGACY_METHODS = {
    "get_task_names": "getTaskNames",
    "get_num_moves": "getNumMoves",
    "get_valid_action_object_combinations": "getValidActionObjectCombinations",
    "get_possible_objects": "getPossibleObjects",
}


def _install_scienceworld_compat(env_class):
    """Adapt the installed ScienceWorld camelCase API for the legacy wrapper."""
    for legacy_name, current_name in _LEGACY_METHODS.items():
        if not hasattr(env_class, legacy_name):
            if not hasattr(env_class, current_name):
                raise RuntimeError(
                    f"ScienceWorld has neither {legacy_name} nor {current_name}; "
                    "check the installed ScienceWorld version"
                )
            setattr(env_class, legacy_name, getattr(env_class, current_name))


def _load_envs_compat(count, config, base_class, wrapper_class, task_class):
    """Select the same sorted tasks as the original loader without api_key=None."""
    selector = base_class()
    task_names = sorted(selector.get_task_names())[1:]
    if count > len(task_names):
        raise ValueError(f"Requested {count} ScienceWorld tasks; only {len(task_names)} are available")
    envs = []
    try:
        for index, task_name in enumerate(task_names[:count]):
            base_env = selector if index == 0 else base_class()
            task = task_class(
                task_id=f"{task_name}-0", sub_task_name=task_name, variation_idx=0,
            )
            env = wrapper_class(task=task, env=base_env, max_env_steps=config.max_env_steps)
            env.reset()
            envs.append(env)
    except Exception:
        for env in envs:
            close = getattr(env.env, "close", None)
            if callable(close):
                close()
        raise
    return envs


def _original_components():
    # Import only when running this task, so the other tasks need no JVM.
    if str(SCI_DIR) not in sys.path:
        sys.path.insert(0, str(SCI_DIR))
    from sciworld import SciWorldConfig
    from sciworld_armap.envs.base import raw_icl
    from sciworld_armap.envs.sciworld_env import SciWorldEnv
    from sciworld_armap.tasks.sciworld import SciWorldTask
    from scienceworld import ScienceWorldEnv as ScienceWorldEnvBase

    _install_scienceworld_compat(ScienceWorldEnvBase)

    def load_envs(count, config):
        return _load_envs_compat(count, config, ScienceWorldEnvBase, SciWorldEnv, SciWorldTask)

    return SciWorldConfig, load_envs, raw_icl


def _history_block(history: list[dict], limit: int | None) -> str:
    selected = history if limit is None else history[-limit:] if limit else []
    blocks = []
    for number, attempt in enumerate(selected, 1):
        steps = []
        for step in attempt["steps"]:
            steps.append(
                f"{step['action']} -> {step['observation']} "
                f"(reward={format_feedback(step.get('shown_reward', step['reward']))})"
            )
        blocks.append(
            f"<Attempt {number}>\n" + "\n".join(steps)
            + f"\nTotal reward: {format_feedback(attempt.get('shown_reward', attempt['reward']))}\n</Attempt>"
        )
    return "\n".join(blocks)


def _attempt(env, history, config, settings, client, log, *, sample_id: int, round_idx: int, bootstrap: bool, raw_icl):
    env.reset()
    task_description = env.env.taskdescription()
    if bootstrap:
        messages = copy.deepcopy(raw_icl[0])
        messages[0]["content"] = (
            config.task_prompt_cot.format(available_actions=config.available_actions)
            + "\n<Attempts>\nHere is an example:\n" + messages[0]["content"]
        )
        messages[-1]["content"] += (
            "\n</Attempts>\n<Instructions>\n" + config.neutral_round_instruction
            + "\n</Instructions>\n" + task_description
        )
    else:
        if round_idx % 2 == 0:
            instruction = config.exploration_instruction
        else:
            instruction = config.exploitation_instruction
        prompt = config.task_prompt_cot.format(available_actions=config.available_actions)
        previous = _history_block(history, settings["history"])
        if previous:
            prompt += "\n<Attempts>\n" + previous + "\n</Attempts>"
        prompt += f"\n<Instructions>{instruction}</Instructions>\n{task_description}"
        messages = [{"role": "user", "content": prompt}]

    steps = []
    for step_idx in range(settings["max_env_steps"]):
        raw_messages = copy.deepcopy(messages)
        output = client.generate(settings["model"], messages, temperature=1.0, max_tokens=settings["max_tokens"])
        observation, state = env.step(output)
        reward = int(state.reward)
        shown = shown_reward("sciworld", reward, settings.get("reward_polarity", "normal"))
        action = env.parse_action(output)
        step = {"step": step_idx + 1, "action": action, "observation": observation,
                "reward": reward, "shown_reward": shown}
        steps.append(step)
        log.emit(
            "eval_step", sample_id=sample_id, round=round_idx, step=step_idx + 1,
            raw_prompt=raw_messages, model_response=output, action=action,
            observation=observation, reward=reward, shown_reward=shown,
            display_reward=format_feedback(shown),
            cumulative_reward=sum(item["reward"] for item in steps), finished=bool(state.finished),
        )
        messages.append({"role": "assistant", "content": output})
        if state.finished:
            break
        messages.append({
            "role": "user", "content": observation
            + f" (Reward: {format_feedback(shown)})\nAvailable objects: "
            + ", ".join(env.env.get_possible_objects()) + "\n" + config.available_actions,
        })
    total = sum(item["reward"] for item in steps)
    shown_total = sum(item["shown_reward"] for item in steps)
    attempt = {"steps": steps, "reward": total, "shown_reward": shown_total,
               "finished": bool(state.finished)}
    log.emit(
        "attempt_end", sample_id=sample_id, round=round_idx, task=task_description,
        reward=total, shown_reward=shown_total,
        display_reward=format_feedback(shown_total), steps=len(steps),
        finished=bool(state.finished), bootstrap=bootstrap,
    )
    return attempt


def run(settings: dict, client, log):
    SciWorldConfig, load_envs, raw_icl = _original_components()
    config = SciWorldConfig()
    config.max_env_steps = settings["max_env_steps"]
    config.use_openai_embedding = False
    envs = load_envs(settings["samples"], config)
    histories = [[] for _ in envs]
    try:
        for bootstrap_idx in range(settings["bootstrap_attempts"]):
            for sample_id, env in enumerate(envs):
                attempt = _attempt(
                    env, histories[sample_id], config, settings, client, log,
                    sample_id=sample_id, round_idx=-settings["bootstrap_attempts"] + bootstrap_idx,
                    bootstrap=True, raw_icl=raw_icl,
                )
                histories[sample_id].append(attempt)
        for round_idx in range(settings["rounds"]):
            rewards = []
            for sample_id, env in enumerate(envs):
                attempt = _attempt(
                    env, histories[sample_id], config, settings, client, log,
                    sample_id=sample_id, round_idx=round_idx, bootstrap=False, raw_icl=raw_icl,
                )
                histories[sample_id].append(attempt)
                rewards.append(attempt["reward"])
            log.emit("round_end", round=round_idx, mean_reward=sum(rewards) / len(rewards), count=len(rewards))
    finally:
        for env in envs:
            close = getattr(env.env, "close", None)
            if callable(close):
                close()
    return {"rounds_completed": settings["rounds"], "samples": len(envs)}
