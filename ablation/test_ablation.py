"""Offline checks; these tests never call SURF or launch ScienceWorld."""

import json
import fcntl
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import APIStatusError

from . import dashboard, events
from . import run as ablation_run
from .client import MODELS, SurfClient, generation_options, judge_options
from .creative_writing import build_prompt as writing_prompt, evaluate as evaluate_writing
from .events import EventLog
from .feedback import feedback_language, format_feedback, spell_reward
from .game24 import (_JUDGE_ANSWER, _one_shot_with_feedback, build_prompt as game24_prompt,
                     evaluate as evaluate_game24, run as run_game24)
from .plot import plot_run
from .reward_polarity import shown_reward
from .sciworld import (_attempt as scienceworld_attempt,
                       _install_scienceworld_compat, _load_envs_compat)


class FeedbackTests(unittest.TestCase):
    def test_qwen_uses_non_thinking_only_for_qwen(self):
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="READY.", model_extra={}),
                                         finish_reason="stop")],
                usage=None,
            )

        client = SurfClient.__new__(SurfClient)
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        self.assertEqual(client.generate(MODELS["qwen35b"], "Reply READY.", temperature=0, max_tokens=64), "READY.")
        self.assertEqual(requests[-1]["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(generation_options(MODELS["qwen35b"]), {"extra_body": requests[-1]["extra_body"]})
        self.assertEqual(client.generate(MODELS["oss120b"], "Reply READY.", temperature=0, max_tokens=64), "READY.")
        self.assertNotIn("extra_body", requests[-1])
        self.assertNotIn("reasoning_effort", requests[-1])
        self.assertEqual(
            client.generate(MODELS["oss120b"], "Reply READY.", temperature=0,
                            max_tokens=1024, **judge_options(MODELS["oss120b"])),
            "READY.",
        )
        self.assertEqual(requests[-1]["reasoning_effort"], "low")
        self.assertEqual(requests[-1]["max_tokens"], 1024)
        self.assertEqual(judge_options(MODELS["qwen35b"]), {})

    def test_empty_content_error_reports_finish_reason_without_reasoning(self):
        def create(**kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None, model_extra={"reasoning": "private"}),
                                         finish_reason="length")],
                usage=SimpleNamespace(completion_tokens=64),
            )

        client = SurfClient.__new__(SurfClient)
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with self.assertRaisesRegex(RuntimeError, "finish_reason=length, completion_tokens=64, reasoning_present=True") as error:
            client.generate(MODELS["qwen35b"], "Reply READY.", temperature=0, max_tokens=64)
        self.assertNotIn("private", str(error.exception))

    def test_empty_judge_response_retries_with_larger_budgets_and_logs_usage(self):
        requests, reports = [], []

        def create(**kwargs):
            requests.append(kwargs)
            content = "**Answer**: 3" if len(requests) == 3 else None
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content, model_extra={"reasoning": "private"}),
                                         finish_reason="stop" if content else "length")],
                usage=SimpleNamespace(completion_tokens=kwargs["max_tokens"],
                                      completion_tokens_details=SimpleNamespace(reasoning_tokens=42)),
            )

        client = SurfClient.__new__(SurfClient)
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        answer = client.generate(MODELS["oss120b"], "Score this.", temperature=0,
                                 max_tokens=64, retry_empty_budgets=(128, 256),
                                 on_response=reports.append)
        self.assertEqual(answer, "**Answer**: 3")
        self.assertEqual([request["max_tokens"] for request in requests], [64, 128, 256])
        self.assertEqual([report["content_present"] for report in reports], [False, False, True])
        self.assertEqual(reports[-1]["reasoning_tokens"], 42)

    def test_cliproxy_uses_completion_limit_and_no_reasoning_for_luna(self):
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="READY", model_extra={}),
                                         finish_reason="stop")], usage=None,
            )

        client = SurfClient.__new__(SurfClient)
        client.provider = "cliproxy"
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        self.assertEqual(client.generate(MODELS["gpt6luna"], "Reply READY", temperature=0,
                                         max_tokens=64), "READY")
        self.assertEqual(requests[0]["max_completion_tokens"], 64)
        self.assertEqual(requests[0]["reasoning_effort"], "none")
        self.assertNotIn("max_tokens", requests[0])

    def test_request_timeout_is_retried(self):
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                response = httpx.Response(408, request=httpx.Request("POST", "https://example.test/v1/chat/completions"))
                raise APIStatusError("request timed out", response=response, body={"code": "request_timeout"})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="**Answer**: 3", model_extra={}),
                                         finish_reason="stop")], usage=None,
            )

        client = SurfClient.__new__(SurfClient)
        client.provider = "cliproxy"
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch("ablation.client.time.sleep"):
            answer = client.generate(MODELS["gpt6luna"], "Score this.", temperature=0, max_tokens=64)
        self.assertEqual(answer, "**Answer**: 3")
        self.assertEqual(len(requests), 2)

    def test_dowhen_changes_only_display(self):
        self.assertEqual(format_feedback(3), "3")
        for language, expected in (("en", "three"), ("fr", "trois"), ("zh", "三")):
            with self.subTest(language=language), feedback_language(language):
                self.assertEqual(format_feedback(3), expected)
                self.assertEqual(format_feedback(-2), spell_reward(-2, language))
        self.assertEqual(format_feedback(3), "3")

    def test_non_integer_is_rejected(self):
        with self.assertRaises(ValueError):
            spell_reward(1.5, "en")

    def test_flipped_reward_scales_and_failed_evaluations(self):
        self.assertEqual([shown_reward("game24", score, "flipped", step=1)
                          for score in (0, 1, 3)], [3, 2, 0])
        self.assertEqual(shown_reward("game24", 10, "flipped", step=4), 0)
        self.assertEqual(shown_reward("game24", 0, "flipped", step=4), 10)
        self.assertEqual([shown_reward("creative_writing", score, "flipped")
                          for score in (1, 5, 10)], [10, 6, 1])
        self.assertEqual(shown_reward("sciworld", 17, "flipped"), -17)
        self.assertEqual(shown_reward("sciworld", -3, "flipped"), 3)
        self.assertEqual(shown_reward("game24", 0, "flipped", step=1,
                                      status="invalid_judge_score"), 0)
        self.assertEqual(shown_reward("creative_writing", 0, "flipped",
                                      status="invalid_judge_score"), 0)

    def test_resume_relocates_data_without_changing_saved_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = {"task": "game24", "game24_data": "/missing/linux/24.csv",
                        "reward_polarity": "flipped"}
            config = root / "config.json"
            config.write_text(json.dumps(original))
            data = root / "24.csv"
            data.write_text("Puzzles\n1 2 3 4\n")
            with self.assertRaisesRegex(ValueError, "--game24-data"):
                ablation_run._resume_settings(root)
            settings = ablation_run._resume_settings(root, data)
            self.assertEqual(settings["game24_data"], str(data.resolve()))
            self.assertEqual(settings["reward_polarity"], "flipped")
            self.assertEqual(json.loads(config.read_text()), original)
            # Once the saved path is valid, it works without an override too.
            config.write_text(json.dumps(settings))
            self.assertEqual(ablation_run._resume_settings(root)["game24_data"], str(data.resolve()))

    def test_reward_control_cli_defaults_and_matched_normal_settings(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        data_root = Path(data_dir.name)
        (data_root / "24.csv").touch()
        (data_root / "data_100_random_text.txt").touch()
        data_patch = patch.object(ablation_run, "DATA_ROOT", data_root)
        data_patch.start()
        self.addCleanup(data_patch.stop)
        for polarity in ("flipped", "random", "empty"):
            with self.subTest(polarity=polarity), patch("sys.argv", [
                "ablation.run", "--task", "all", "--model", "gpt6luna",
                "--provider", "cliproxy", "--feedback", "numeric",
                "--reward-polarity", polarity,
            ]):
                args = ablation_run.parse_args()
            self.assertEqual((args.game24_demo_policy, args.writing_seed_policy),
                             ("unscored", "judged"))
            self.assertEqual(args.random_reward_seed, 0)
        with patch("sys.argv", ["ablation.run", "--task", "all", "--model", "gpt6luna",
                                "--provider", "cliproxy", "--feedback", "numeric",
                                "--game24-demo-policy", "unscored",
                                "--writing-seed-policy", "judged"]):
            args = ablation_run.parse_args()
        self.assertEqual(args.reward_polarity, "normal")
        self.assertEqual((args.game24_demo_policy, args.writing_seed_policy),
                         ("unscored", "judged"))

    def test_random_rewards_are_reproducible_and_independent_of_true_scores(self):
        for task, step, valid in (("game24", 1, {0, 1, 3}), ("game24", 4, {0, 10}),
                                  ("creative_writing", None, set(range(1, 11))),
                                  ("sciworld", 1, set(range(-100, 101)))):
            with self.subTest(task=task, step=step):
                def draws(reward, seed=17):
                    return [shown_reward(task, reward, "random", step=step,
                                         random_seed=seed, sample_id=sample, round_idx=2)
                            for sample in range(64)]
                values = draws(min(valid))
                self.assertEqual(values, draws(max(valid)))
                self.assertEqual(values, draws(min(valid)))
                self.assertTrue(set(values) <= valid)
                self.assertGreater(len(set(values)), 1)
                self.assertNotEqual(values, draws(min(valid), seed=18))
        for status in ("invalid_generation", "invalid_judge_score"):
            self.assertEqual(shown_reward("game24", 0, "random", step=4, status=status), 0)

    def test_random_run_saves_seed_for_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "24.csv"
            data.write_text("1 2 3 4\n", encoding="utf-8")
            with patch.object(events, "OUTPUT_ROOT", root / "outputs"), \
                 patch("ablation.run.SurfClient"), \
                 patch("ablation.run._run_task", return_value={}) as run_task, \
                 patch("ablation.run.plot_run"), patch("builtins.print"), \
                 patch("sys.argv", [
                     "ablation.run", "--task", "game24", "--model", "gpt6luna",
                     "--provider", "cliproxy", "--feedback", "numeric",
                     "--reward-polarity", "random", "--random-reward-seed", "17",
                     "--game24-data", str(data),
                 ]):
                ablation_run.main()
            config_path, = (root / "outputs").rglob("config.json")
            settings = ablation_run._resume_settings(config_path.parent)
            self.assertEqual(settings["random_reward_seed"], 17)
            self.assertEqual(settings["reward_polarity"], "random")
            self.assertEqual(run_task.call_args.args[1]["random_reward_seed"], 17)

    def test_flipped_game24_keeps_true_score_and_uses_shown_history(self):
        class FakeJudge:
            def generate(self, model, prompt, **kwargs):
                if "Solution:" in prompt:
                    return "**Answer**: 10"
                return f"**Answer**: {('0', '1', '3')[int(prompt.split('Step ')[1][0]) - 1]}"

        class FakeLog:
            def __init__(self):
                self.records = []

            def emit(self, event, **fields):
                self.records.append((event, fields))

        log = FakeLog()
        response = "<answer>Step 1: a Step 2: b Step 3: c **Answer**: d</answer>"
        with feedback_language("en"):
            result = evaluate_game24("1 2 3 4", response, FakeJudge(), MODELS["gpt6luna"],
                                     log, sample_id=0, round_idx=1, reward_polarity="flipped")
            prompt = game24_prompt("1 2 3 4", [result], 1, None, "unscored")
        self.assertEqual(result["step_rewards"], [0, 1, 3, 10])
        self.assertEqual(result["shown_step_rewards"], [3, 2, 0, 0])
        self.assertEqual((result["reward"], result["shown_reward"]), (14, 5))
        self.assertIn("<Reward: three> Step 2", prompt)
        self.assertIn("**Reward**: five", prompt)
        self.assertEqual(prompt.count("**Reward**:"), 1)
        first = next(fields for event, fields in log.records if event == "eval_step")
        self.assertEqual((first["reward"], first["shown_reward"], first["display_reward"]),
                         (0, 3, "three"))

    def test_writing_reward_controls_keep_true_score(self):
        class FakeJudge:
            def generate(self, *args, **kwargs):
                return "Coherency score: 8"

        class FakeLog:
            def __init__(self):
                self.records = []

            def emit(self, event, **fields):
                self.records.append((event, fields))

        for polarity in ("flipped", "random", "empty"):
            log = FakeLog()
            with feedback_language("fr"):
                result = evaluate_writing("<answer>Passage: Bonjour.</answer>", FakeJudge(),
                                          MODELS["gpt6luna"], log, sample_id=0, round_idx=0,
                                          reward_polarity=polarity, seed_policy="judged", random_seed=17)
                prompt = writing_prompt("End.", [result], 1, None, include_rewards=polarity != "empty")
            shown = shown_reward("creative_writing", 8, polarity, random_seed=17)
            self.assertEqual((result["reward"], result["shown_reward"]), (8, shown))
            self.assertIn("Passage: Bonjour.", prompt)
            self.assertEqual(log.records[0][1]["status"], "ok")
            if polarity == "empty":
                self.assertIsNone(shown)
                self.assertNotIn("reward", prompt.lower())
                self.assertEqual(log.records[0][1]["display_reward"], "")
            else:
                self.assertIn(f"**Reward**: {spell_reward(shown, 'fr')}", prompt)
                self.assertEqual(log.records[0][1]["display_reward"], spell_reward(shown, "fr"))

            class InvalidJudge:
                def generate(self, *args, **kwargs):
                    return "No numeric score"

            invalid = evaluate_writing("<answer>Passage: Bonjour.</answer>", InvalidJudge(),
                                       MODELS["gpt6luna"], log, sample_id=0, round_idx=1,
                                       reward_polarity=polarity)
            self.assertEqual((invalid["reward"], invalid["shown_reward"]),
                             (0, None if polarity == "empty" else 0))
            self.assertEqual(invalid["eval_status"], "invalid_judge_score")

    def test_game24_demo_and_history_rendered_as_words(self):
        history = [{"response": "Step 1: a\nStep 2: b\nStep 3: c\n**Answer**: d",
                    "step_rewards": [3, 1, 0, 10], "reward": 14}]
        with feedback_language("en"):
            demo = _one_shot_with_feedback()
            prompt = game24_prompt("1 2 3 4", history, 1, None)
        self.assertIn("**Reward**: zero", demo)
        self.assertNotIn("**Reward**: 0", demo)
        self.assertIn("<Reward: three> Step 2", prompt)
        self.assertIn("<Reward: one> Step 3", prompt)
        self.assertIn("**Reward**: fourteen", prompt)
        empty = game24_prompt("1 2 3 4", history, 1, None, include_rewards=False)
        self.assertIn(history[0]["response"], empty)
        self.assertIn("Input: 4 4 6 8", empty)
        self.assertNotIn("reward", empty.lower())

    def test_game24_accepts_plain_or_bold_judge_answer(self):
        for response in ("Answer: 0", "**Answer**: 0"):
            with self.subTest(response=response):
                match = _JUDGE_ANSWER.search(response)
                self.assertIsNotNone(match)
                self.assertEqual(int(match.group(1)), 0)

    def test_zero_history_excludes_prior_attempts(self):
        history = [{"answer": "old", "reward": 3}]
        prompt = writing_prompt("The end.", history, 1, 0)
        self.assertNotIn("old", prompt)

    def test_both_llm_judges_use_separate_reasoning_budget(self):
        class FakeJudge:
            def __init__(self):
                self.requests = []

            def generate(self, model, prompt, **kwargs):
                self.requests.append((model, kwargs))
                return "**Answer**: 3" if "Rule of Game of 24" in prompt else "Coherency score: 5"

        class FakeLog:
            def emit(self, *args, **kwargs):
                pass

        judge = FakeJudge()
        evaluate_game24("1 2 3 4", "<answer>Step 1: a Step 2: b Step 3: c **Answer**: d</answer>", judge,
                        MODELS["oss120b"], FakeLog(), sample_id=0, round_idx=0,
                        judge_max_tokens=1024)
        evaluate_writing("<answer>Passage: Hello.</answer>", judge, MODELS["oss120b"],
                         FakeLog(), sample_id=0, round_idx=1, judge_max_tokens=1024)
        self.assertEqual(len(judge.requests), 5)
        for model, options in judge.requests:
            self.assertEqual(model, MODELS["oss120b"])
            self.assertEqual(options["temperature"], 0)
            self.assertEqual(options["max_tokens"], 1024)
            self.assertEqual(options["reasoning_effort"], "low")
            self.assertEqual(options["retry_empty_budgets"], (2048, 4096))

    def test_malformed_game24_output_is_zero_without_judge_call(self):
        class FakeJudge:
            def generate(self, *args, **kwargs):
                raise AssertionError("Judge must not be called for malformed output")

        class FakeLog:
            def __init__(self):
                self.events = []

            def emit(self, event, **fields):
                self.events.append((event, fields))

        log = FakeLog()
        result = evaluate_game24("4 5 6 8", "<answer>Step 1: 6 - 4 = 2 (left:",
                                 FakeJudge(), MODELS["oss120b"], log,
                                 sample_id=0, round_idx=0, reward_polarity="flipped")
        self.assertEqual(result["step_rewards"], [0, 0, 0, 0])
        self.assertEqual(result["shown_reward"], 0)
        self.assertEqual(result["eval_status"], "invalid_generation")
        self.assertEqual(result["response"], "<answer>Step 1: 6 - 4 = 2 (left:")
        self.assertEqual([fields["status"] for event, fields in log.events if event == "eval_step"],
                         ["invalid_generation"] * 4)

    def test_game24_resume_reuses_pending_generation_without_repeating_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            puzzle_file = Path(temporary) / "24.csv"
            puzzle_file.write_text("1 2 3 4\n", encoding="utf-8")
            settings = {
                "game24_data": str(puzzle_file), "start_index": 0, "samples": 1,
                "rounds": 2, "history": None, "model": MODELS["qwen35b"],
                "judge_model": MODELS["oss120b"], "max_tokens": 1000,
                "judge_max_tokens": 1024, "_resume_path": str(run_dir),
            }
            (run_dir / "config.json").write_text(json.dumps({"task": "game24"}), encoding="utf-8")
            prior = [
                {"event": "generation", "round": 0, "sample_id": 0,
                 "model_response": "<answer>Step 1: a Step 2: b Step 3: c **Answer**: 1+2+3+4</answer>"},
                {"event": "attempt_end", "round": 0, "sample_id": 0, "answer": "1+2+3+4",
                 "step_rewards": [3, 3, 3, 0], "reward": 9},
                {"event": "round_end", "round": 0, "count": 1, "mean_reward": 9},
                {"event": "generation", "round": 1, "sample_id": 0,
                 "model_response": "<answer>Step 1: unfinished"},
                {"event": "run_error", "error_type": "EmptyResponseError"},
            ]
            event_path = run_dir / "events.jsonl"
            event_path.write_text("".join(json.dumps(event) + "\n" for event in prior), encoding="utf-8")

            class NoCallClient:
                def generate(self, *args, **kwargs):
                    raise AssertionError("Saved generation should be reused; invalid text needs no judge")

            log = EventLog.reopen(run_dir)
            try:
                with feedback_language("en"):
                    summary = run_game24(settings, NoCallClient(), log)
            finally:
                log.close()
            self.assertEqual(summary["rounds_completed"], 2)
            records = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event["event"] == "generation" for event in records), 2)
            self.assertEqual(sum(event["event"] == "attempt_end" for event in records), 2)
            self.assertEqual(sum(event["event"] == "round_end" for event in records), 2)
            self.assertEqual(records[-2]["eval_status"], "invalid_generation")

    def test_game24_resume_reuses_partial_judge_scores(self):
        for polarity, saved_count in ((p, n) for p in ("flipped", "random", "empty") for n in (2, 4)):
            with self.subTest(polarity=polarity, saved_count=saved_count), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                run_dir.mkdir()
                puzzle_file = Path(temporary) / "24.csv"
                puzzle_file.write_text("1 2 3 4\n", encoding="utf-8")
                settings = {
                    "game24_data": str(puzzle_file), "start_index": 0, "samples": 1,
                    "rounds": 1, "history": None, "model": MODELS["gpt6luna"],
                    "judge_model": MODELS["gpt6luna"], "max_tokens": 1000,
                    "judge_max_tokens": 1024, "_resume_path": str(run_dir),
                    "reward_polarity": polarity, "random_reward_seed": 17, "game24_demo_policy": "unscored",
                }
                (run_dir / "config.json").write_text(json.dumps({"task": "game24"}), encoding="utf-8")
                prior = [
                    {"event": "generation", "round": 0, "sample_id": 0,
                     "model_response": "<answer>Step 1: a Step 2: b Step 3: c **Answer**: 1+2+3+4</answer>"},
                ]
                for step, score in enumerate((3, 0, 1, 0)[:saved_count], start=1):
                    prior.append({"event": "eval_step", "round": 0, "sample_id": 0,
                                  "step": step, "reward": score, "status": "ok",
                                  "shown_reward": shown_reward("game24", score, polarity,
                                                               step=step, random_seed=17)})
                prior.append({"event": "run_error", "error_type": "APIStatusError"})
                event_path = run_dir / "events.jsonl"
                event_path.write_text("".join(json.dumps(event) + "\n" for event in prior), encoding="utf-8")

                class RemainingJudge:
                    def __init__(self):
                        self.prompts = []

                    def generate(self, model, prompt, **kwargs):
                        self.prompts.append(prompt)
                        return "**Answer**: 0" if "Solution:" in prompt else "**Answer**: 1"

                judge = RemainingJudge()
                log = EventLog.reopen(run_dir)
                try:
                    with feedback_language("fr"):
                        summary = run_game24(settings, judge, log)
                finally:
                    log.close()
                self.assertEqual(summary["rounds_completed"], 1)
                records = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
                evaluations = [event for event in records if event["event"] == "eval_step"]
                attempts = [event for event in records if event["event"] == "attempt_end"]
                self.assertEqual([event["step"] for event in evaluations], [1, 2, 3, 4])
                self.assertEqual(len([event for event in records if event["event"] == "generation"]), 1)
                self.assertEqual(len(judge.prompts), 4 - saved_count)
                self.assertEqual(attempts[0]["step_rewards"], [3, 0, 1, 0])
                self.assertEqual(attempts[0]["reward"], 4)
                expected = [shown_reward("game24", score, polarity, step=step, random_seed=17)
                            for step, score in enumerate((3, 0, 1, 0), start=1)]
                self.assertEqual(attempts[0]["shown_step_rewards"], expected)
                self.assertEqual(attempts[0]["shown_reward"],
                                 None if polarity == "empty" else sum(expected))
                if polarity == "empty":
                    self.assertEqual(attempts[0]["shown_step_rewards"], [None] * 4)
                    self.assertEqual(attempts[0]["display_reward"], "")
                self.assertEqual(len(attempts), 1)
                self.assertEqual(len([event for event in records if event["event"] == "round_end"]), 1)
                reused = next(event for event in records if event["event"] == "reuse_generation")
                self.assertEqual(reused["saved_step_evaluations"], saved_count)

    def test_game24_resume_rejects_out_of_order_partial_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            event_path = Path(temporary) / "events.jsonl"
            events = [
                {"event": "generation", "round": 0, "sample_id": 0, "model_response": "<answer>saved</answer>"},
                {"event": "eval_step", "round": 0, "sample_id": 0, "step": 2,
                 "reward": 3, "status": "ok"},
            ]
            event_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            from .game24 import _resume_state
            with self.assertRaisesRegex(ValueError, "non-contiguous step evaluations"):
                _resume_state(event_path, 1, 1)

    def test_scienceworld_immediate_feedback_is_text_but_logged_numeric(self):
        class FakeEnv:
            def __init__(self):
                self.env = SimpleNamespace(taskdescription=lambda: "Test task",
                                           get_possible_objects=lambda: ["cup"])
                self.calls = 0

            def reset(self):
                self.calls = 0

            def step(self, output):
                self.calls += 1
                return "Observation: moved", SimpleNamespace(reward=3 if self.calls == 1 else 0,
                                                                finished=self.calls == 2)

            def parse_action(self, output):
                return "move"

        class FakeClient:
            def __init__(self):
                self.prompts = []

            def generate(self, model, messages, **kwargs):
                self.prompts.append([dict(message) for message in messages])
                return "Action: move"

        class FakeLog:
            def __init__(self):
                self.records = []

            def emit(self, event, **fields):
                self.records.append((event, fields))

        config = SimpleNamespace(task_prompt_cot="Task {available_actions}", available_actions="move",
                                 exploration_instruction="Explore", exploitation_instruction="Exploit rewards",
                                 neutral_round_instruction="\nTry to complete the task.")
        settings = {"history": None, "max_env_steps": 2, "model": "fake", "max_tokens": 8}
        client, log = FakeClient(), FakeLog()
        with feedback_language("en"):
            scienceworld_attempt(FakeEnv(), [], config, settings, client, log,
                                 sample_id=0, round_idx=0, bootstrap=False, raw_icl=[])
        self.assertIn("(Reward: three)", client.prompts[1][-1]["content"])
        first_step = next(fields for event, fields in log.records if event == "eval_step")
        self.assertEqual(first_step["reward"], 3)
        self.assertEqual(first_step["display_reward"], "three")

        settings["reward_polarity"] = "flipped"
        flipped_client, flipped_log = FakeClient(), FakeLog()
        with feedback_language("en"):
            attempt = scienceworld_attempt(FakeEnv(), [], config, settings,
                                           flipped_client, flipped_log,
                                           sample_id=0, round_idx=0, bootstrap=False, raw_icl=[])
        self.assertIn("(Reward: minus three)", flipped_client.prompts[1][-1]["content"])
        self.assertEqual((attempt["reward"], attempt["shown_reward"]), (3, -3))
        flipped_step = next(fields for event, fields in flipped_log.records if event == "eval_step")
        self.assertEqual((flipped_step["reward"], flipped_step["shown_reward"],
                          flipped_step["display_reward"]), (3, -3, "minus three"))

        settings.update(reward_polarity="random", random_reward_seed=17)
        random_client, random_log = FakeClient(), FakeLog()
        with feedback_language("en"):
            attempt = scienceworld_attempt(FakeEnv(), [], config, settings,
                                           random_client, random_log,
                                           sample_id=0, round_idx=0, bootstrap=False, raw_icl=[])
        expected = [shown_reward("sciworld", score, "random", step=step, random_seed=17)
                    for step, score in enumerate((3, 0), start=1)]
        self.assertEqual(attempt["reward"], 3)
        self.assertEqual([step["shown_reward"] for step in attempt["steps"]], expected)
        self.assertEqual(attempt["shown_reward"], sum(expected))
        self.assertIn(f"(Reward: {spell_reward(expected[0], 'en')})",
                      random_client.prompts[1][-1]["content"])

        settings["reward_polarity"] = "empty"
        empty_client, empty_log = FakeClient(), FakeLog()
        with feedback_language("en"):
            empty_attempt = scienceworld_attempt(FakeEnv(), [attempt], config, settings,
                                                 empty_client, empty_log, sample_id=0,
                                                 round_idx=1, bootstrap=False, raw_icl=[])
        self.assertIn("move -> Observation: moved", empty_client.prompts[0][0]["content"])
        self.assertIn("Observation: moved\nAvailable objects: cup", empty_client.prompts[1][-1]["content"])
        self.assertNotIn("reward", str(empty_client.prompts).lower())
        self.assertEqual((empty_attempt["reward"], empty_attempt["shown_reward"]), (3, None))
        for _, fields in empty_log.records:
            self.assertIsNone(fields["shown_reward"])
            self.assertEqual(fields["display_reward"], "")

    def test_scienceworld_legacy_method_aliases(self):
        class CurrentEnv:
            def getTaskNames(self):
                return ["task"]

            def getNumMoves(self):
                return 2

            def getValidActionObjectCombinations(self):
                return ["look around"]

            def getPossibleObjects(self):
                return ["cup"]

        _install_scienceworld_compat(CurrentEnv)
        env = CurrentEnv()
        self.assertEqual(env.get_task_names(), ["task"])
        self.assertEqual(env.get_num_moves(), 2)
        self.assertEqual(env.get_valid_action_object_combinations(), ["look around"])
        self.assertEqual(env.get_possible_objects(), ["cup"])

    def test_scienceworld_loader_skips_first_sorted_task_without_api_key_kwarg(self):
        class Base:
            def get_task_names(self):
                return ["task-c", "task-a", "task-b"]

        class Wrapper:
            def __init__(self, *, task, env, max_env_steps):
                self.task, self.env, self.max_env_steps = task, env, max_env_steps
                self.reset_called = False

            def reset(self):
                self.reset_called = True

        def task_class(**kwargs):
            return SimpleNamespace(**kwargs)

        envs = _load_envs_compat(2, SimpleNamespace(max_env_steps=3),
                                 Base, Wrapper, task_class)
        self.assertEqual([env.task.sub_task_name for env in envs], ["task-b", "task-c"])
        self.assertTrue(all(env.reset_called and env.max_env_steps == 3 for env in envs))

    def test_dashboard_dropdown_shows_time_and_run_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for stamp, outcome in (("20261015T085703Z", "failed"),
                                   ("20261015T090000Z", "succeeded"),
                                   ("20261015T091000Z", "incomplete")):
                run_dir = root / "sciworld" / "gpt6luna" / "numeric" / f"{stamp}-test"
                run_dir.mkdir(parents=True)
                (run_dir / "config.json").write_text(json.dumps({
                    "task": "sciworld", "model": "gpt-6-luna", "model_alias": "gpt6luna",
                    "feedback": "numeric", "samples": 2,
                }), encoding="utf-8")
                event = "run_error" if outcome == "failed" else "run_start"
                (run_dir / "events.jsonl").write_text(
                    json.dumps({"event": event, "detail": "x" * 10000}) + "\n",
                    encoding="utf-8",
                )
                if outcome == "succeeded":
                    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
            with patch.object(dashboard, "OUTPUT_ROOT", root):
                client = dashboard.app.test_client()
                runs = client.get("/api/runs").json
                self.assertEqual([run["status"] for run in runs],
                                 ["incomplete", "succeeded", "failed"])
                self.assertEqual(runs[-1]["started_at"], "2026-10-15T08:57:03+00:00")
                active_path = root / runs[0]["id"] / "events.jsonl"
                with active_path.open("a", encoding="utf-8") as stream:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    active = client.get("/api/runs").json
                    self.assertEqual(active[0]["status"], "running")
                page = client.get("/").get_data(as_text=True)
                self.assertIn("new Date(r.started_at).toLocaleString()", page)
                self.assertIn("esc(r.status)", page)

    def test_dashboard_completion_filter_requires_completed_iteration_50(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "game24" / "test" / "numeric" / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(json.dumps({
                "task": "game24", "model": "test", "feedback": "numeric", "samples": 2,
                "rounds": 51,
            }), encoding="utf-8")
            summary = run_dir / "summary.json"
            with patch.object(dashboard, "OUTPUT_ROOT", root):
                client = dashboard.app.test_client()
                for completed_rounds in (1, 50, 51):
                    with self.subTest(completed_rounds=completed_rounds):
                        summary.write_text(json.dumps({"rounds_completed": completed_rounds}))
                        run = client.get("/api/runs").json[0]
                        self.assertTrue(run["finished"])
                        self.assertEqual(run["reached_iteration_50"], completed_rounds == 51)
                summary.unlink()
                event_path = run_dir / "events.jsonl"
                event_path.write_text(json.dumps({"event": "round_end", "round": 49}) + "\n"
                                      + json.dumps({"event": "attempt_end", "round": 50,
                                                    "sample_id": 0, "detail": "x" * 10000}) + "\n")
                self.assertFalse(client.get("/api/runs").json[0]["reached_iteration_50"])
                with event_path.open("a") as stream:
                    stream.write(json.dumps({"event": "round_end", "round": 50}) + "\n")
                self.assertTrue(client.get("/api/runs").json[0]["reached_iteration_50"])

    def test_reward_control_runs_have_separate_paths_and_dashboard_labels(self):
        for polarity in ("flipped", "random", "empty"):
            with tempfile.TemporaryDirectory() as temporary, patch.object(events, "OUTPUT_ROOT", Path(temporary)):
                config = {"task": "game24", "model": "test", "model_alias": "test",
                          "feedback": "numeric", "reward_polarity": polarity, "samples": 1}
                log = EventLog("game24", "test", "numeric", config)
                try:
                    self.assertEqual(log.path.parent.name, polarity)
                finally:
                    log.close()
                with patch.object(dashboard, "OUTPUT_ROOT", Path(temporary)):
                    listed = dashboard.app.test_client().get("/api/runs").json
                self.assertEqual(len(listed), 1)
                self.assertEqual(listed[0]["reward_polarity"], polarity)

    def test_plot_reads_numeric_rewards(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "config.json").write_text(
                json.dumps({"task": "game24", "model_alias": "test", "feedback": "fr"}),
                encoding="utf-8",
            )
            (run_dir / "events.jsonl").write_text(
                '\n'.join(json.dumps(event) for event in (
                    {"event": "attempt_end", "round": 0, "reward": 3, "display_reward": "trois"},
                    {"event": "attempt_end", "round": 1, "reward": 10, "display_reward": "dix"},
                )) + '\n', encoding="utf-8",
            )
            plot_path = plot_run(run_dir)
            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_dashboard_tracks_success_and_four_step_curves_incrementally(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "game24" / "test" / "en" / "run-1"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(json.dumps({
                "task": "game24", "model": "test", "feedback": "en", "samples": 2,
            }), encoding="utf-8")
            events = []
            for sample_id, scores, success in (
                (0, [3, 1, 0, 10], True), (1, [0, 0, 0, 0], False),
            ):
                events.extend({"event": "eval_step", "round": 0, "sample_id": sample_id,
                               "step": step, "reward": score}
                              for step, score in enumerate(scores, start=1))
                events.append({"event": "attempt_end", "round": 0, "sample_id": sample_id,
                               "reward": sum(scores), "accuracy_reward": 0 if success else -24,
                               "format_reward": 0 if success else -30})
            event_path = run_dir / "events.jsonl"
            event_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            with patch.object(dashboard, "OUTPUT_ROOT", root):
                client = dashboard.app.test_client()
                run_id = "game24/test/en/run-1"
                response = client.get("/api/metrics", query_string={"run": run_id})
                self.assertEqual(response.status_code, 200)
                first = response.json["rounds"][0]
                self.assertTrue(first["complete"])
                self.assertEqual(first["success"]["mean"], 0.5)
                self.assertEqual(first["success"]["sem"], 0.5)
                self.assertAlmostEqual(first["success"]["std"], 0.5 ** 0.5)
                self.assertEqual(first["reward"]["mean"], 7.0)
                self.assertAlmostEqual(first["reward"]["std"], 98 ** 0.5)
                self.assertEqual([part["mean"] for part in first["steps"]], [1.5, 0.5, 0.0, 5.0])
                with event_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"event": "attempt_end", "round": 1, "sample_id": 0,
                                             "reward": 3, "accuracy_reward": 0, "format_reward": 0}) + "\n")
                second = client.get("/api/metrics", query_string={"run": run_id}).json["rounds"]
                self.assertEqual(len(second), 2)
                self.assertFalse(second[1]["complete"])
                self.assertEqual(second[1]["reward"]["count"], 1)
                self.assertIsNone(second[1]["reward"]["std"])
                self.assertEqual(second[1]["success"]["mean"], 1.0)
                self.assertEqual(client.get("/api/metrics", query_string={"run": run_id})
                                 .json["rounds"][1]["reward"]["count"], 1)
                page = client.get("/").get_data(as_text=True)
                self.assertIn('id="step4"', page)
                self.assertIn("Game24 success rate by trial", page)
                self.assertIn('id="game24RewardChart"', page)
                self.assertIn("Game24 mean judge reward by trial", page)


if __name__ == "__main__":
    unittest.main()
