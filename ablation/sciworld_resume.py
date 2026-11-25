"""Validate ScienceWorld checkpoints before reusing model responses."""

import json

from .reward_polarity import shown_reward
from .sciworld_checkpoint import compact_attempt, load_checkpoint

REWARD_POLICY = "score_delta_failure_minus_100_v2"
API400_BACKTRACK_LIMIT = 5
API400_FIRST_STEP_RETRY_DELAYS = (5, 15, 30)


def api400_backtrack_limit(settings):
    """Upgrade older recovery allowances without rewriting saved configurations."""
    return max(API400_BACKTRACK_LIMIT, settings.get("sciworld_api400_backtracks", API400_BACKTRACK_LIMIT))


def backtrack_attempt(attempt, step):
    """Discard an action and its suffix, retaining the append-only audit log."""
    del attempt["steps"][step - 1:]
    attempt["generations"] = {index: event for index, event in attempt["generations"].items()
                              if index < step}


def resume_state(path, settings):
    attempts, rounds, manifest, offset = load_checkpoint(path, settings)
    last_order = max(((r + settings["bootstrap_attempts"], s) for r, s in attempts), default=(-1, -1))
    with path.open("rb") as stream:
        stream.seek(offset)
        return _read_events(stream, settings, attempts, rounds, manifest, last_order)


def _read_events(stream, settings, attempts, rounds, manifest, last_order):
    for line in stream:
        event = json.loads(line)
        kind = event.get("event")
        if kind == "env_manifest":
            if manifest is not None and manifest != event["tasks"]:
                raise ValueError("Conflicting ScienceWorld task manifests")
            manifest = event["tasks"]
        if kind == "round_end":
            round_idx = event["round"]
            if round_idx in rounds or not 0 <= round_idx < settings["rounds"]:
                raise ValueError("Invalid or duplicate ScienceWorld round summary")
            if any(not attempts.get((round_idx, s), {}).get("end") for s in range(settings["samples"])):
                raise ValueError("ScienceWorld round summary precedes completed attempts")
            mean = sum(attempts[(round_idx, s)]["end"]["reward"] for s in range(settings["samples"])) / settings["samples"]
            if event["count"] != settings["samples"] or event["mean_reward"] != mean:
                raise ValueError("ScienceWorld round summary disagrees with its attempts")
            rounds.add(round_idx)
        if kind in ("attempt_restart", "attempt_backtrack"):
            key = (event["round"], event["sample_id"])
            prior = attempts.get(key)
            if (prior is None or prior["end"] is not None
                    or (key[0] + settings["bootstrap_attempts"], key[1]) != last_order):
                raise ValueError("ScienceWorld restart/backtrack must refer to the current incomplete attempt")
            if kind == "attempt_restart":
                attempts[key] = {"steps": [], "generations": {}, "end": None,
                                 "api400_backtracks": prior.get("api400_backtracks", 0),
                                 "api400_first_step_retries": prior.get("api400_first_step_retries", 0)}
            else:
                step = event.get("step")
                count = prior.get("api400_backtracks", 0) + 1
                if (type(step) is not int or step < 1 or step != len(prior["steps"])
                        or prior["steps"][-1]["finished"]
                        or event.get("failed_step") != step + 1
                        or event.get("status_code") != 400 or event.get("backtrack_count") != count
                        or count > api400_backtrack_limit(settings)):
                    raise ValueError("Invalid ScienceWorld API 400 backtrack")
                backtrack_attempt(prior, step)
                prior["api400_backtracks"] = count
        if kind not in ("attempt_retry", "generation", "eval_step", "attempt_end"):
            continue
        round_idx, sample = event["round"], event["sample_id"]
        if not (-settings["bootstrap_attempts"] <= round_idx < settings["rounds"] and
                0 <= sample < settings["samples"]):
            raise ValueError("ScienceWorld checkpoint has an invalid attempt index")
        order = (round_idx + settings["bootstrap_attempts"], sample)
        expected_order = (0, 0) if last_order == (-1, -1) else last_order
        if order != expected_order:
            next_order = (last_order[0] + (last_order[1] + 1) // settings["samples"],
                          (last_order[1] + 1) % settings["samples"])
            if order != next_order or not attempts.get(
                    (last_order[0] - settings["bootstrap_attempts"], last_order[1]), {}).get("end"):
                raise ValueError("ScienceWorld checkpoint has missing or out-of-order attempts")
        last_order = order
        attempt = attempts.setdefault((round_idx, sample), {"steps": [], "generations": {}, "end": None})
        if attempt["end"] is not None:
            raise ValueError("Duplicate or late records for a completed ScienceWorld attempt")
        steps = attempt["steps"]
        if steps and steps[-1]["finished"]:
            if kind != "attempt_end":
                raise ValueError("ScienceWorld checkpoint continues after termination")
        if kind == "attempt_retry":
            count = attempt.get("api400_first_step_retries", 0) + 1
            delays = settings.get("sciworld_first_step_400_retry_delays", API400_FIRST_STEP_RETRY_DELAYS)
            if (steps or attempt["generations"] or event.get("step") != 1
                    or event.get("status_code") != 400 or event.get("error_code") != "invalid_prompt"
                    or event.get("retry_count") != count or count > len(delays)
                    or event.get("retry_delay_seconds") != delays[count - 1]):
                raise ValueError("Invalid ScienceWorld first-step API 400 retry")
            attempt["api400_first_step_retries"] = count
        elif kind == "generation":
            step = event["step"]
            if step != len(steps) + 1 or step in attempt["generations"] or step > settings["max_env_steps"]:
                raise ValueError("Duplicate or non-contiguous ScienceWorld generation")
            attempt["generations"][step] = event
        elif kind == "eval_step":
            step = event["step"]
            generation = attempt["generations"].get(step)
            if step != len(steps) + 1 or generation is None:
                raise ValueError("Missing generation or non-contiguous ScienceWorld evaluations")
            if generation["model_response"] != event["model_response"] or generation["raw_prompt"] != event["raw_prompt"]:
                raise ValueError("ScienceWorld generation and evaluation disagree")
            if type(event["reward"]) is not int or type(event["finished"]) is not bool:
                raise ValueError("Invalid ScienceWorld reward or terminal flag")
            expected = shown_reward("sciworld", event["reward"], settings.get("reward_polarity", "normal"),
                                    step=step, random_seed=settings.get("random_reward_seed", 0),
                                    sample_id=sample, round_idx=round_idx)
            if event["shown_reward"] != expected or event["cumulative_reward"] != sum(s["reward"] for s in steps) + event["reward"]:
                raise ValueError("ScienceWorld checkpoint rewards are inconsistent")
            steps.append(event)
        else:
            total = sum(s["reward"] for s in steps)
            shown = None if settings.get("reward_polarity") == "empty" else sum(s["shown_reward"] for s in steps)
            if (not steps or event["steps"] != len(steps) or len(attempt["generations"]) != len(steps)
                    or (not steps[-1]["finished"] and len(steps) != settings["max_env_steps"])
                    or event["reward"] != total or event["shown_reward"] != shown
                    or event["bootstrap"] != (round_idx < 0)
                    or event["finished"] != steps[-1]["finished"]):
                raise ValueError("ScienceWorld attempt summary disagrees with its steps")
            attempt["end"] = event
            attempts[(round_idx, sample)] = compact_attempt(attempt)
    if attempts and manifest is None:
        raise ValueError("ScienceWorld checkpoint is missing its task manifest")
    return attempts, rounds, manifest
