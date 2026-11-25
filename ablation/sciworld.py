"""ScienceWorld loop reusing the repository's action wrapper and score patch."""

import copy
import sys
import time
from pathlib import Path

from openai import APIStatusError

from .feedback import format_feedback
from .reward_polarity import shown_reward
from .sciworld_resume import (API400_FIRST_STEP_RETRY_DELAYS, api400_backtrack_limit,
                             backtrack_attempt, resume_state)
from .sciworld_checkpoint import Checkpointer

ROOT = Path(__file__).resolve().parents[1]
SCI_DIR = ROOT / "experiments" / "sciworld"


class ReplayMismatchError(ValueError):
    """A restarted simulator cannot reproduce the unfinished attempt exactly."""


def _install_scienceworld_compat(env_class):
    if str(SCI_DIR) not in sys.path:
        sys.path.insert(0, str(SCI_DIR))
    from sciworld_armap.utils.compat import install_scienceworld_compat
    install_scienceworld_compat(env_class)


def _load_envs_compat(count, config, base_class, wrapper_class, task_class):
    """Select the same sorted tasks as the original loader without api_key=None."""
    selector = base_class()
    envs = []
    bases = [selector]
    try:
        available = sorted(selector.get_task_names())
        task_names = getattr(config, "task_names", None) or available[1:]
        if len(set(task_names)) != len(task_names) or any(t not in available for t in task_names):
            raise ValueError("ScienceWorld task names must be unique and present in the environment")
        if count > len(task_names):
            raise ValueError(f"Requested {count} ScienceWorld tasks; only {len(task_names)} are available")
        variation = getattr(config, "variation_idx", 0)
        for name in task_names[:count]:
            if hasattr(selector, "getMaxVariations") and not 0 <= variation < selector.getMaxVariations(name):
                raise ValueError(f"Invalid variation {variation} for {name}")
        for index, task_name in enumerate(task_names[:count]):
            base_env = selector if index == 0 else base_class()
            if index:
                bases.append(base_env)
            task = task_class(
                task_id=f"{task_name}-{variation}", sub_task_name=task_name, variation_idx=variation,
            )
            split = getattr(config, "sciworld_split", None)
            if split:
                base_env.load(task_name, variation, simplificationStr=config.simplifications)
                variations = getattr(base_env, f"getVariations{split.title()}")()
                if variation not in variations:
                    raise ValueError(f"Variation {variation} for {task_name} is not in the {split} split; available: {variations}")
            options = {}
            if hasattr(config, "simplifications"):
                options["simplifications"] = config.simplifications
            env = wrapper_class(task=task, env=base_env, max_env_steps=config.max_env_steps, **options)
            envs.append(env)
            env.reset()
    except BaseException:
        for base in bases:
            close = getattr(base, "close", None)
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


def _history_block(history: list[dict], limit: int | None, *, include_rewards: bool = True) -> str:
    selected = history if limit is None else history[-limit:] if limit else []
    blocks = []
    for number, attempt in enumerate(selected, 1):
        steps = []
        for step in attempt["steps"]:
            line = f"{step['action']} -> {step['observation']}"
            if include_rewards:
                line += f" (reward={format_feedback(step.get('shown_reward', step['reward']))})"
            steps.append(line)
        body = "\n".join(steps)
        if include_rewards:
            body += f"\nTotal reward: {format_feedback(attempt.get('shown_reward', attempt['reward']))}"
        blocks.append(f"<Attempt {number}>\n{body}\n</Attempt>")
    return "\n".join(blocks)


def _generate_action(messages, settings, client, log, *, sample_id, round_idx, step, saved):
    """Retry a rejected first request before changing the simulator or prompt."""
    retry_delays = settings.get("sciworld_first_step_400_retry_delays", API400_FIRST_STEP_RETRY_DELAYS)
    while True:
        try:
            return client.generate(settings["model"], messages, temperature=1.0, max_tokens=settings["max_tokens"])
        except APIStatusError as exc:
            count = saved.get("api400_first_step_retries", 0)
            if step != 1 or exc.status_code != 400 or exc.code != "invalid_prompt" or count >= len(retry_delays):
                raise
            delay = retry_delays[count]
            log.emit("attempt_retry", sample_id=sample_id, round=round_idx, step=1,
                     status_code=400, error_code=exc.code, retry_count=count + 1,
                     retry_delay_seconds=delay, error_type=type(exc).__name__, error=str(exc))
            saved["api400_first_step_retries"] = count + 1
            print(f"ScienceWorld API 400: retrying step 1 in round {round_idx}, task {sample_id} "
                  f"in {delay}s ({count + 1}/{len(retry_delays)})", flush=True)
            time.sleep(delay)


def _attempt(env, history, config, settings, client, log, *, sample_id: int, round_idx: int, bootstrap: bool, raw_icl, saved=None):
    active = copy.deepcopy(saved) if saved else {"steps": [], "generations": {}}
    limit = api400_backtrack_limit(settings)
    while True:
        try:
            return _attempt_once(env, history, config, settings, client, log,
                                 sample_id=sample_id, round_idx=round_idx, bootstrap=bootstrap,
                                 raw_icl=raw_icl, saved=active)
        except APIStatusError as exc:
            count = active.get("api400_backtracks", 0)
            if exc.status_code != 400 or not active["steps"] or count >= limit:
                raise
            step = len(active["steps"])
            log.emit("attempt_backtrack", sample_id=sample_id, round=round_idx,
                     step=step, failed_step=step + 1, status_code=400,
                     backtrack_count=count + 1, error_type=type(exc).__name__, error=str(exc))
            backtrack_attempt(active, step)
            active["api400_backtracks"] = count + 1
            print(f"ScienceWorld API 400: backing up to step {step} in round {round_idx}, task {sample_id} ({count + 1}/{limit})", flush=True)
        except ReplayMismatchError as exc:
            if (not active["steps"] and not active["generations"]
                    or settings.get("_sciworld_resume_mode", "auto") == "strict"):
                raise
            log.emit("attempt_restart", sample_id=sample_id, round=round_idx, reason=str(exc))
            print(f"ScienceWorld replay differs: restarting unfinished round {round_idx}, task {sample_id}; completed attempts are retained", flush=True)
            active = {"steps": [], "generations": {},
                      "api400_backtracks": active.get("api400_backtracks", 0),
                      "api400_first_step_retries": active.get("api400_first_step_retries", 0)}


def _attempt_once(env, history, config, settings, client, log, *, sample_id: int, round_idx: int, bootstrap: bool, raw_icl, saved):
    env.reset()
    task_description = env.env.taskdescription()
    reward_polarity = settings.get("reward_polarity", "normal")
    include_rewards = reward_polarity != "empty"
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
        elif include_rewards:
            instruction = config.exploitation_instruction
        else:
            instruction = (
                "Look at the previous attempts and use their actions and observations to complete the task."
                + config.neutral_round_instruction
            )
        prompt = config.task_prompt_cot.format(available_actions=config.available_actions)
        previous = _history_block(history, settings["history"], include_rewards=include_rewards)
        if previous:
            prompt += "\n<Attempts>\n" + previous + "\n</Attempts>"
        prompt += f"\n<Instructions>{instruction}</Instructions>\n{task_description}"
        messages = [{"role": "user", "content": prompt}]

    steps = []
    for step_idx in range(settings["max_env_steps"]):
        raw_messages = copy.deepcopy(messages)
        generation = saved["generations"].get(step_idx + 1)
        if generation:
            if generation["raw_prompt"] != raw_messages:
                raise ReplayMismatchError(f"ScienceWorld replay prompt differs from saved generation at step {step_idx + 1}")
            output = generation["model_response"]
        else:
            output = _generate_action(messages, settings, client, log, sample_id=sample_id,
                                      round_idx=round_idx, step=step_idx + 1, saved=saved)
            generation = dict(sample_id=sample_id, round=round_idx, step=step_idx + 1,
                              raw_prompt=raw_messages, model_response=output)
            log.emit("generation", **generation)
            saved["generations"][step_idx + 1] = generation
        observation, state = env.step(output)
        reward = int(state.reward)
        shown = shown_reward("sciworld", reward, reward_polarity,
                             step=step_idx + 1, random_seed=settings.get("random_reward_seed", 0),
                             sample_id=sample_id, round_idx=round_idx)
        action = env.parse_action(output)
        step = {"step": step_idx + 1, "action": action, "observation": observation,
                "reward": reward, "shown_reward": shown}
        steps.append(step)
        evaluation = dict(sample_id=sample_id, round=round_idx, step=step_idx + 1,
            raw_prompt=raw_messages, model_response=output, action=action,
            observation=observation, reward=reward, shown_reward=shown,
            display_reward=format_feedback(shown),
            cumulative_reward=sum(item["reward"] for item in steps), finished=bool(state.finished),
        )
        if step_idx < len(saved["steps"]):
            previous = saved["steps"][step_idx]
            for field in ("action", "observation", "reward", "shown_reward", "finished"):
                if evaluation[field] != previous[field]:
                    raise ReplayMismatchError(f"ScienceWorld replay differs at step {step_idx + 1}: {field}")
        else:
            log.emit("eval_step", **evaluation)
            saved["steps"].append(evaluation)
        messages.append({"role": "assistant", "content": output})
        if state.finished:
            break
        feedback = f" (Reward: {format_feedback(shown)})" if include_rewards else ""
        messages.append({
            "role": "user", "content": observation + feedback + "\nAvailable objects: "
            + ", ".join(env.env.get_possible_objects()) + "\n" + config.available_actions,
        })
    total = sum(item["reward"] for item in steps)
    shown_total = sum(item["shown_reward"] for item in steps) if include_rewards else None
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
    config.task_names = settings.get("sciworld_tasks")
    config.variation_idx = settings.get("variation_idx", 0)
    config.simplifications = settings.get("simplifications", "easy")
    config.sciworld_split = settings.get("sciworld_split")
    saved, saved_rounds, saved_manifest = {}, set(), None
    if settings.get("_resume_path"):
        saved, saved_rounds, saved_manifest = resume_state(Path(settings["_resume_path"]) / "events.jsonl", settings)
        completed = sum(attempt["end"] is not None for attempt in saved.values())
        print(f"ScienceWorld resume: restoring {completed} completed attempts and {len(saved_rounds)} completed rounds", flush=True)
    checkpointer = Checkpointer(log.path, settings)
    envs = load_envs(settings["samples"], config)
    histories = [[] for _ in envs]
    try:
        manifest = [{"task": e.task.sub_task_name, "variation": e.task.variation_idx,
                     "simplifications": config.simplifications} for e in envs]
        if saved_manifest is not None and saved_manifest != manifest:
            raise ValueError("Installed ScienceWorld tasks differ from the saved manifest")
        if saved_manifest is None:
            log.emit("env_manifest", tasks=manifest)
        checkpointer.save(saved, saved_rounds, manifest)

        def attempt_for(sample_id, env, round_idx, bootstrap):
            key = (round_idx, sample_id)
            prior = saved.get(key)
            if prior and prior["end"]:
                end = prior["end"]
                return {"steps": prior["steps"], "reward": end["reward"],
                        "shown_reward": end["shown_reward"], "finished": end["finished"]}

            def execute(previous):
                return _attempt(env, histories[sample_id], config, settings, client, log,
                                sample_id=sample_id, round_idx=round_idx, bootstrap=bootstrap,
                                raw_icl=raw_icl, saved=previous)

            mode = settings.get("_sciworld_resume_mode", "auto")
            if prior and mode == "restart-attempt":
                log.emit("attempt_restart", sample_id=sample_id, round=round_idx,
                         reason="Requested restart of the unfinished attempt")
                prior = {"steps": [], "generations": {}, "end": None,
                         "api400_backtracks": prior.get("api400_backtracks", 0),
                         "api400_first_step_retries": prior.get("api400_first_step_retries", 0)}
            attempt = execute(prior)
            saved[key] = {"steps": attempt["steps"], "generations": {}, "end": {
                "round": round_idx, "sample_id": sample_id, "reward": attempt["reward"],
                "shown_reward": attempt["shown_reward"], "steps": len(attempt["steps"]),
                "finished": attempt["finished"], "bootstrap": bootstrap,
            }}
            checkpointer.save(saved, saved_rounds, manifest)
            return attempt

        for bootstrap_idx in range(settings["bootstrap_attempts"]):
            for sample_id, env in enumerate(envs):
                attempt = attempt_for(sample_id, env, -settings["bootstrap_attempts"] + bootstrap_idx, True)
                histories[sample_id].append(attempt)
        for round_idx in range(settings["rounds"]):
            rewards = []
            for sample_id, env in enumerate(envs):
                attempt = attempt_for(sample_id, env, round_idx, False)
                histories[sample_id].append(attempt)
                rewards.append(attempt["reward"])
            if round_idx not in saved_rounds:
                log.emit("round_end", round=round_idx, mean_reward=sum(rewards) / len(rewards), count=len(rewards))
                saved_rounds.add(round_idx)
                checkpointer.save(saved, saved_rounds, manifest)
    finally:
        for env in envs:
            close = getattr(env.env, "close", None)
            if callable(close):
                close()
    return {"rounds_completed": settings["rounds"], "samples": len(envs)}
