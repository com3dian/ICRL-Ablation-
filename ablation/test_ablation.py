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

from . import dashboard
from .client import MODELS, SurfClient, generation_options, judge_options
from .creative_writing import build_prompt as writing_prompt, evaluate as evaluate_writing
from .events import EventLog
from .feedback import feedback_language, format_feedback, spell_reward
from .game24 import (_JUDGE_ANSWER, _one_shot_with_feedback, build_prompt as game24_prompt,
                     evaluate as evaluate_game24, run as run_game24)
from .plot import plot_run
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
                                 sample_id=0, round_idx=0)
        self.assertEqual(result["step_rewards"], [0, 0, 0, 0])
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
        for saved_count in (2, 4):
            with self.subTest(saved_count=saved_count), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                run_dir.mkdir()
                puzzle_file = Path(temporary) / "24.csv"
                puzzle_file.write_text("1 2 3 4\n", encoding="utf-8")
                settings = {
                    "game24_data": str(puzzle_file), "start_index": 0, "samples": 1,
                    "rounds": 1, "history": None, "model": MODELS["gpt6luna"],
                    "judge_model": MODELS["gpt6luna"], "max_tokens": 1000,
                    "judge_max_tokens": 1024, "_resume_path": str(run_dir),
                }
                (run_dir / "config.json").write_text(json.dumps({"task": "game24"}), encoding="utf-8")
                prior = [
                    {"event": "generation", "round": 0, "sample_id": 0,
                     "model_response": "<answer>Step 1: a Step 2: b Step 3: c **Answer**: 1+2+3+4</answer>"},
                ]
                for step, score in enumerate((3, 0, 1, 0)[:saved_count], start=1):
                    prior.append({"event": "eval_step", "round": 0, "sample_id": 0,
                                  "step": step, "reward": score, "status": "ok"})
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
                                 exploration_instruction="Explore", exploitation_instruction="Exploit")
        settings = {"history": None, "max_env_steps": 2, "model": "fake", "max_tokens": 8}
        client, log = FakeClient(), FakeLog()
        with feedback_language("en"):
            scienceworld_attempt(FakeEnv(), [], config, settings, client, log,
                                 sample_id=0, round_idx=0, bootstrap=False, raw_icl=[])
        self.assertIn("(Reward: three)", client.prompts[1][-1]["content"])
        first_step = next(fields for event, fields in log.records if event == "eval_step")
        self.assertEqual(first_step["reward"], 3)
        self.assertEqual(first_step["display_reward"], "three")

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
                self.assertEqual(first["reward"]["mean"], 7.0)
                self.assertEqual([part["mean"] for part in first["steps"]], [1.5, 0.5, 0.0, 5.0])
                with event_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"event": "attempt_end", "round": 1, "sample_id": 0,
                                             "reward": 3, "accuracy_reward": 0, "format_reward": 0}) + "\n")
                second = client.get("/api/metrics", query_string={"run": run_id}).json["rounds"]
                self.assertEqual(len(second), 2)
                self.assertFalse(second[1]["complete"])
                self.assertEqual(second[1]["reward"]["count"], 1)
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
