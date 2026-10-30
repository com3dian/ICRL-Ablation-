"""Run one or a matrix of numeric/word-feedback ICRL conditions on SURF."""

import argparse
import json
import traceback
from pathlib import Path

from .client import (CLIPROXY_MODELS, MODELS, SURF_MODELS, SurfClient,
                     generation_options, judge_options)
from .events import EventLog
from .feedback import LANGUAGES, feedback_language
from .plot import plot_run
from .reward_polarity import POLARITIES

TASKS = ("game24", "creative_writing", "sciworld")
DEFAULT_SAMPLES = {"game24": 100, "creative_writing": 100, "sciworld": 29}
DEFAULT_ROUNDS = {"game24": 51, "creative_writing": 51, "sciworld": 40}
DATA_ROOT = Path(__file__).resolve().parent / "data"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "all"))
    parser.add_argument("--model", choices=(*MODELS, "all"))
    parser.add_argument("--feedback", choices=(*LANGUAGES, "all"))
    parser.add_argument("--reward-polarity", choices=POLARITIES, default="normal",
                        help="Reverse model-visible rewards while retaining true scores for evaluation")
    parser.add_argument("--writing-seed-policy", choices=("fixed", "judged"),
                        help="Creative Writing round 0: default fixed for normal, judged for flipped")
    parser.add_argument("--game24-demo-policy", choices=("legacy", "unscored"),
                        help="Game24 examples: default legacy for normal, unscored for flipped")
    parser.add_argument("--provider", choices=("surf", "cliproxy"), default="surf")
    parser.add_argument("--resume", type=Path, help="Resume one unfinished Game24 run from its output directory")
    parser.add_argument("--game24-data", type=Path, help="Puzzle CSV; defaults to ablation/data/24.csv for new runs; also overrides the saved path with --resume")
    parser.add_argument("--writing-data", type=Path, default=DATA_ROOT / "data_100_random_text.txt", help="Writing prompts; defaults to ablation/data/data_100_random_text.txt")
    parser.add_argument("--start-index", type=int, default=900, help="Game24 offset, matching the original experiment")
    parser.add_argument("--samples", type=int, help="Samples per task; defaults are 100, 100, and 29")
    parser.add_argument("--rounds", type=int, help="Override task defaults: 51 for Game24/Writing, 40 for ScienceWorld")
    parser.add_argument("--history", type=int, help="Max prior attempts in prompt; default: original task setting")
    parser.add_argument("--bootstrap-attempts", type=int, default=2, help="ScienceWorld initial attempts")
    parser.add_argument("--max-env-steps", type=int, default=15, help="ScienceWorld action limit")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--judge-model", help="Fixed judge; defaults to GPT-OSS on SURF or GPT-6 Luna on Cliproxy")
    parser.add_argument("--judge-max-tokens", type=int, default=1024, help="Judge output budget, including reasoning tokens")
    parser.add_argument("--keep-going", action="store_true", help="Continue other conditions if one model/run fails")
    args = parser.parse_args()
    if args.samples is not None and args.samples < 1:
        parser.error("--samples must be positive")
    if (args.rounds is not None and args.rounds < 1) or args.max_env_steps < 1 or args.max_tokens < 1 or args.judge_max_tokens < 1:
        parser.error("--rounds, --max-env-steps, --max-tokens, and --judge-max-tokens must be positive")
    if args.history is not None and args.history < 0:
        parser.error("--history cannot be negative")
    if args.bootstrap_attempts < 0 or args.start_index < 0:
        parser.error("--bootstrap-attempts and --start-index cannot be negative")
    if args.resume:
        if any((args.task, args.model, args.feedback, args.judge_model, args.samples, args.rounds)) or args.reward_polarity != "normal" or args.writing_seed_policy is not None or args.game24_demo_policy is not None:
            parser.error("--resume reads the saved configuration; do not combine it with condition overrides")
        if not (args.resume / "config.json").is_file() or not (args.resume / "events.jsonl").is_file():
            parser.error("--resume must point to a run directory containing config.json and events.jsonl")
        return args
    if not all((args.task, args.model, args.feedback)):
        parser.error("--task, --model, and --feedback are required unless --resume is used")
    args.game24_data = args.game24_data or DATA_ROOT / "24.csv"
    available = SURF_MODELS if args.provider == "surf" else CLIPROXY_MODELS
    if args.model != "all" and args.model not in available:
        parser.error(f"--model {args.model} is not available for {args.provider}")
    tasks = TASKS if args.task == "all" else (args.task,)
    if args.writing_seed_policy is None:
        args.writing_seed_policy = "judged" if args.reward_polarity == "flipped" else "fixed"
    if args.game24_demo_policy is None:
        args.game24_demo_policy = "unscored" if args.reward_polarity == "flipped" else "legacy"
    if args.reward_polarity == "flipped" and "creative_writing" in tasks and args.writing_seed_policy != "judged":
        parser.error("Flipped Creative Writing requires --writing-seed-policy judged; use it for the matched normal run too")
    if args.reward_polarity == "flipped" and "game24" in tasks and args.game24_demo_policy != "unscored":
        parser.error("Flipped Game24 requires --game24-demo-policy unscored; use it for the matched normal run too")
    if "game24" in tasks and (not args.game24_data or not args.game24_data.is_file()):
        parser.error("Game24 puzzle file is missing; run 'python -m ablation.prepare_data' or pass --game24-data PATH")
    if "creative_writing" in tasks and (not args.writing_data or not args.writing_data.is_file()):
        parser.error("writing prompt file is missing; run 'python -m ablation.prepare_data' or pass --writing-data PATH")
    if args.judge_model is None:
        args.judge_model = MODELS["oss120b" if args.provider == "surf" else "gpt6luna"]
    elif args.judge_model in MODELS:
        if args.judge_model not in available:
            parser.error(f"--judge-model {args.judge_model} is not available for {args.provider}")
        args.judge_model = MODELS[args.judge_model]
    return args


def _run_task(task: str, settings: dict, client, log):
    if task == "game24":
        from .game24 import run
    elif task == "creative_writing":
        from .creative_writing import run
    else:
        from .sciworld import run
    return run(settings, client, log)


def _resume_settings(run_dir: Path, game24_data: Path | None = None) -> dict:
    settings = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if settings["task"] != "game24":
        raise ValueError("--resume currently supports Game24 only")
    data_path = game24_data or Path(settings["game24_data"])
    if not data_path.is_file():
        raise ValueError(
            f"Saved Game24 data is missing: {data_path}. "
            "Run 'python -m ablation.prepare_data', then resume with "
            "--game24-data ablation/data/24.csv (use the same dataset as the original run)."
        )
    settings["game24_data"] = str(data_path.resolve())
    settings["_resume_path"] = str(run_dir)
    return settings


def main():
    args = parse_args()
    if args.resume:
        run_dir = args.resume.resolve()
        try:
            settings = _resume_settings(run_dir, args.game24_data)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        client = SurfClient(provider=settings.get("provider", "surf"))
        log = EventLog.reopen(run_dir)
        try:
            with feedback_language(settings["feedback"]):
                log.emit(
                    "run_resume", game24_data=settings["game24_data"], invalid_generation_policy="zero_reward_no_judge",
                    judge_retry_budgets=[settings["judge_max_tokens"] * 2,
                                         settings["judge_max_tokens"] * 4],
                )
                summary = _run_task("game24", settings, client, log)
                log.finish(summary)
            print(f"Resumed and finished: {run_dir} (plot: {plot_run(run_dir)})", flush=True)
        except Exception as exc:
            log.emit("run_error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            log.close()
        return
    tasks = TASKS if args.task == "all" else (args.task,)
    provider_models = SURF_MODELS if args.provider == "surf" else CLIPROXY_MODELS
    models = tuple(provider_models) if args.model == "all" else (args.model,)
    feedbacks = LANGUAGES if args.feedback == "all" else (args.feedback,)
    client = SurfClient(provider=args.provider)
    failed = []
    for task in tasks:
        for model_alias in models:
            for feedback in feedbacks:
                history = args.history if args.history is not None else (25 if task == "creative_writing" else None)
                settings = {
                    "task": task, "model_alias": model_alias, "model": MODELS[model_alias],
                    "provider": args.provider,
                    "generation_options": generation_options(MODELS[model_alias]),
                    "judge_model": args.judge_model,
                    "judge_generation_options": generation_options(args.judge_model) | judge_options(args.judge_model),
                    "judge_max_tokens": args.judge_max_tokens,
                    "judge_retry_budgets": [args.judge_max_tokens * 2, args.judge_max_tokens * 4],
                    "invalid_generation_policy": "zero_reward_no_judge",
                    "feedback": feedback,
                    "reward_polarity": args.reward_polarity,
                    "writing_seed_policy": args.writing_seed_policy,
                    "game24_demo_policy": args.game24_demo_policy,
                    "rounds": args.rounds or DEFAULT_ROUNDS[task], "samples": args.samples or DEFAULT_SAMPLES[task],
                    "history": history, "max_tokens": args.max_tokens,
                    "game24_data": str(args.game24_data.resolve()) if args.game24_data else None,
                    "writing_data": str(args.writing_data.resolve()) if args.writing_data else None,
                    "start_index": args.start_index, "bootstrap_attempts": args.bootstrap_attempts,
                    "max_env_steps": args.max_env_steps,
                }
                log = EventLog(task, model_alias, feedback, settings)
                print(f"Starting {task} / {model_alias} / {feedback} / {args.reward_polarity}: {log.path}", flush=True)
                try:
                    with feedback_language(feedback):
                        log.emit("run_start", task=task, model=MODELS[model_alias], feedback=feedback)
                        summary = _run_task(task, settings, client, log)
                        log.finish(summary)
                    plot_path = plot_run(log.path)
                    print(f"Finished: {log.path} (plot: {plot_path})", flush=True)
                except Exception as exc:
                    log.emit("run_error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
                    print(f"Run failed; partial events are in {log.path}", flush=True)
                    if not args.keep_going:
                        raise
                    failed.append(str(log.path))
                finally:
                    log.close()
    if failed:
        raise SystemExit(f"{len(failed)} condition(s) failed; inspect run_error in their events.jsonl files")


if __name__ == "__main__":
    main()
