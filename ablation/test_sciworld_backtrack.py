"""API 400 rollback, simulator consistency, and durable recovery without API calls."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import httpx
from openai import APIStatusError, BadRequestError

from . import events, sciworld
from .events import EventLog
from .reward_polarity import shown_reward
from .run import _resume_settings
from .sciworld_checkpoint import Checkpointer
from .sciworld_resume import API400_BACKTRACK_LIMIT, API400_FIRST_STEP_RETRY_DELAYS, resume_state
from . import test_sciworld as fixtures


def api_error(status=400, code=None):
    response = httpx.Response(status, request=httpx.Request("POST", "https://test.invalid"))
    error_class = BadRequestError if status == 400 else APIStatusError
    return error_class("rejected request", response=response, body={"code": code})


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, model, messages, **kwargs):
        self.prompts.append(copy.deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return "Action: " + response


class ActionEnv:
    def __init__(self):
        self.task = SimpleNamespace(sub_task_name="test", variation_idx=0)
        self.env = self
        self.resets = 0
        self.closed = False

    def reset(self):
        self.actions = []
        self.resets += 1

    def taskdescription(self):
        return "test"

    def get_possible_objects(self):
        return ["cup"]

    def parse_action(self, output):
        return output.removeprefix("Action: ")

    def step(self, output):
        action = self.parse_action(output)
        self.actions.append(action)
        return f"Observation: {action}", SimpleNamespace(
            reward={"keep": 10, "discard": 80, "replace": 20, "finish": 30}[action],
            finished=len(self.actions) == 3)

    def close(self):
        self.closed = True


class BacktrackTests(unittest.TestCase):
    def fixture(self, root, polarity="normal", **overrides):
        settings = fixtures.ScienceWorldTests().settings(polarity)
        settings.update(samples=1, rounds=1, bootstrap_attempts=0)
        settings.update(overrides)
        env = ActionEnv()
        components = (fixtures.config, lambda count, cfg: [env], [[{"role": "user", "content": "Example"}]])
        self.enterContext(patch.object(events, "OUTPUT_ROOT", root))
        self.enterContext(patch.object(sciworld, "_original_components", return_value=components))
        log = EventLog("sciworld", "fake", "numeric", settings)
        self.addCleanup(log.close)
        return settings, log, env

    def records(self, path):
        return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]

    def test_backtrack_restores_simulator_prompt_and_true_and_shown_rewards(self):
        for polarity in ("normal", "flipped", "random", "empty"):
            with self.subTest(polarity=polarity), tempfile.TemporaryDirectory() as tmp:
                settings, log, env = self.fixture(Path(tmp), polarity)
                client = ScriptedClient(["keep", "discard", api_error(), "replace", "finish"])
                sciworld.run(settings, client, log)
                self.assertEqual(env.resets, 2)
                self.assertEqual(env.actions, ["keep", "replace", "finish"])
                self.assertTrue(env.closed)
                self.assertIn("Action: discard", str(client.prompts[2]))
                self.assertNotIn("discard", str(client.prompts[3]))
                self.assertEqual(client.prompts[1], client.prompts[3])
                records = self.records(log.path)
                rollback = next(e for e in records if e["event"] == "attempt_backtrack")
                self.assertEqual((rollback["step"], rollback["failed_step"], rollback["backtrack_count"]), (2, 3, 1))
                ends = [e for e in records if e["event"] == "attempt_end"]
                self.assertEqual(len(ends), 1)
                self.assertEqual((ends[0]["reward"], ends[0]["steps"]), (60, 3))
                shown = None if polarity == "empty" else sum(
                    shown_reward("sciworld", reward, polarity, step=step,
                                 random_seed=settings["random_reward_seed"])
                    for step, reward in enumerate((10, 20, 30), 1))
                self.assertEqual(ends[0]["shown_reward"], shown)
                self.assertEqual(next(e for e in records if e["event"] == "round_end")["mean_reward"], 60)
                # Audit entries retain the discarded step; both restore paths
                # must recover only the three active steps.
                for use_snapshot in (True, False):
                    if not use_snapshot:
                        (log.path / "checkpoint.json").unlink()
                    attempts, rounds, _ = resume_state(log.path / "events.jsonl", settings)
                    self.assertEqual([s["action"] for s in attempts[(0, 0)]["steps"]], env.actions)
                    self.assertEqual(attempts[(0, 0)]["end"]["reward"], 60)
                    self.assertEqual(rounds, {0})

    def test_crash_at_each_backtrack_boundary_resumes_without_duplicate_calls(self):
        for boundary in ("attempt_backtrack", "generation", "eval_step"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                settings, log, _ = self.fixture(Path(tmp))
                client = ScriptedClient(["keep", "discard", api_error(), "replace", "finish"])
                emit, rolled_back = log.emit, False

                def interrupt(event, **fields):
                    nonlocal rolled_back
                    emit(event, **fields)
                    if event == "attempt_backtrack":
                        rolled_back = True
                    if rolled_back and event == boundary:
                        raise RuntimeError("process interrupted")

                log.emit = interrupt
                with self.assertRaisesRegex(RuntimeError, "process interrupted"):
                    sciworld.run(settings, client, log)
                log.close()
                saved, rounds, manifest = resume_state(log.path / "events.jsonl", settings)
                self.assertEqual(saved[(0, 0)]["api400_backtracks"], 1)
                # Exercise a snapshot containing an incomplete backtracked
                # attempt, including a pending replacement response.
                Checkpointer(log.path, settings).save(saved, rounds, manifest)
                resumed_client = ScriptedClient(["replace", "finish"] if boundary == "attempt_backtrack" else ["finish"])
                resumed = EventLog.reopen(log.path)
                try:
                    sciworld.run(_resume_settings(log.path), resumed_client, resumed)
                finally:
                    resumed.close()
                self.assertEqual(len(client.prompts) + len(resumed_client.prompts), 5)
                records = self.records(log.path)
                self.assertEqual(sum(e["event"] == "attempt_end" for e in records), 1)
                (log.path / "checkpoint.json").unlink()
                saved, _, _ = resume_state(log.path / "events.jsonl", settings)
                self.assertEqual(saved[(0, 0)]["end"]["reward"], 60)

    def test_repeated_rejections_are_bounded_across_resumes_and_explicit_restarts(self):
        for mode in ("auto", "restart-attempt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                settings, log, _ = self.fixture(Path(tmp), sciworld_api400_backtracks=3)
                failure = api_error()
                responses = [item for _ in range(API400_BACKTRACK_LIMIT + 1) for item in ("keep", failure)]
                client = ScriptedClient(responses)
                with self.assertRaises(APIStatusError) as caught:
                    sciworld.run(settings, client, log)
                self.assertIs(caught.exception, failure)
                self.assertEqual(len(client.prompts), 2 * (API400_BACKTRACK_LIMIT + 1))
                log.close()
                saved, rounds, manifest = resume_state(log.path / "events.jsonl", settings)
                self.assertEqual(saved[(0, 0)]["api400_backtracks"], API400_BACKTRACK_LIMIT)
                Checkpointer(log.path, settings).save(saved, rounds, manifest)
                resumed_settings = _resume_settings(log.path)
                resumed_settings["_sciworld_resume_mode"] = mode
                resumed = EventLog.reopen(log.path)
                resumed_client = ScriptedClient([failure] if mode == "auto" else ["keep", failure])
                try:
                    with self.assertRaises(APIStatusError):
                        sciworld.run(resumed_settings, resumed_client, resumed)
                finally:
                    resumed.close()
                self.assertEqual(sum(e["event"] == "attempt_backtrack" for e in self.records(log.path)), API400_BACKTRACK_LIMIT)
                (log.path / "checkpoint.json").unlink()
                saved, _, _ = resume_state(log.path / "events.jsonl", settings)
                self.assertEqual(saved[(0, 0)]["api400_backtracks"], API400_BACKTRACK_LIMIT)

    def test_legacy_three_backtrack_run_resumes_with_two_more_and_retains_completed_round(self):
        for mode in ("auto", "restart-attempt"):
            for use_snapshot in (True, False):
                with self.subTest(mode=mode, snapshot=use_snapshot), tempfile.TemporaryDirectory() as tmp:
                    settings, log, _ = self.fixture(Path(tmp), "random", rounds=2, sciworld_api400_backtracks=3)
                    original_config = (log.path / "config.json").read_bytes()
                    failure = api_error(code="invalid_prompt")
                    client = ScriptedClient(["keep", "replace", "finish"]
                                            + [item for _ in range(4) for item in ("keep", failure)])
                    with patch("ablation.sciworld_resume.API400_BACKTRACK_LIMIT", 3), self.assertRaises(BadRequestError):
                        sciworld.run(settings, client, log)
                    log.close()
                    saved, rounds, manifest = resume_state(log.path / "events.jsonl", settings)
                    self.assertEqual(rounds, {0})
                    self.assertEqual(saved[(1, 0)]["api400_backtracks"], 3)
                    Checkpointer(log.path, settings).save(saved, rounds, manifest)
                    if not use_snapshot:
                        (log.path / "checkpoint.json").unlink()
                    resumed_settings = _resume_settings(log.path)
                    resumed_settings["_sciworld_resume_mode"] = mode
                    responses = ([failure] if mode == "auto" else ["keep", failure])
                    resumed_client = ScriptedClient(responses + ["keep", failure, "keep", "replace", "finish"])
                    resumed = EventLog.reopen(log.path)
                    try:
                        sciworld.run(resumed_settings, resumed_client, resumed)
                    finally:
                        resumed.close()
                    self.assertEqual((log.path / "config.json").read_bytes(), original_config)
                    records = self.records(log.path)
                    self.assertEqual([e["backtrack_count"] for e in records if e["event"] == "attempt_backtrack"],
                                     [1, 2, 3, 4, 5])
                    self.assertEqual(sum(e["event"] == "attempt_end" for e in records), 2)
                    self.assertEqual(sum(e["event"] == "round_end" for e in records), 2)
                    for snapshot_after_resume in (True, False):
                        if not snapshot_after_resume:
                            (log.path / "checkpoint.json").unlink()
                        restored, restored_rounds, _ = resume_state(log.path / "events.jsonl", resumed_settings)
                        self.assertEqual(restored_rounds, {0, 1})
                        self.assertEqual([restored[(r, 0)]["end"]["reward"] for r in range(2)], [60, 60])

    def test_first_step_invalid_prompt_retries_without_duplicate_actions_or_rewards(self):
        for polarity in ("normal", "flipped", "random", "empty"):
            for bootstrap in (0, 1):
                with self.subTest(polarity=polarity, bootstrap=bootstrap), tempfile.TemporaryDirectory() as tmp:
                    settings, log, env = self.fixture(Path(tmp), polarity, bootstrap_attempts=bootstrap)
                    client = ScriptedClient([api_error(code="invalid_prompt")] * 3
                                            + ["keep", "replace", "finish"] * (bootstrap + 1))
                    with patch("ablation.sciworld.time.sleep") as sleep:
                        sciworld.run(settings, client, log)
                    self.assertEqual(sleep.call_args_list, [call(5), call(15), call(30)])
                    self.assertTrue(all(prompt == client.prompts[0] for prompt in client.prompts[:4]))
                    self.assertEqual(env.resets, bootstrap + 1)
                    self.assertEqual(env.actions, ["keep", "replace", "finish"])
                    self.assertTrue(env.closed)
                    records = self.records(log.path)
                    retries = [e for e in records if e["event"] == "attempt_retry"]
                    self.assertEqual([e["retry_count"] for e in retries], [1, 2, 3])
                    self.assertTrue(all(e["round"] == -bootstrap for e in retries))
                    self.assertEqual(sum(e["event"] == "generation" for e in records), 3 * (bootstrap + 1))
                    self.assertEqual(sum(e["event"] == "eval_step" for e in records), 3 * (bootstrap + 1))
                    ends = [e for e in records if e["event"] == "attempt_end"]
                    self.assertEqual([e["reward"] for e in ends], [60] * (bootstrap + 1))
                    self.assertFalse(any(e["event"] == "attempt_backtrack" for e in records))
                    for use_snapshot in (True, False):
                        if not use_snapshot:
                            (log.path / "checkpoint.json").unlink()
                        saved, rounds, _ = resume_state(log.path / "events.jsonl", settings)
                        self.assertEqual(len(saved), bootstrap + 1)
                        self.assertEqual(rounds, {0})

    def test_first_step_invalid_prompt_retries_are_bounded_across_resume_and_restart(self):
        for mode in ("auto", "restart-attempt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                settings, log, env = self.fixture(Path(tmp))
                failure = api_error(code="invalid_prompt")
                client = ScriptedClient([failure] * 4)
                with patch("ablation.sciworld.time.sleep") as sleep, self.assertRaises(BadRequestError):
                    sciworld.run(settings, client, log)
                self.assertEqual(len(client.prompts), 4)
                self.assertEqual(sleep.call_args_list, [call(d) for d in API400_FIRST_STEP_RETRY_DELAYS])
                self.assertEqual(env.actions, [])
                self.assertTrue(env.closed)
                log.close()
                saved, rounds, manifest = resume_state(log.path / "events.jsonl", settings)
                self.assertEqual(saved[(0, 0)]["api400_first_step_retries"], 3)
                Checkpointer(log.path, settings).save(saved, rounds, manifest)
                resumed_settings = _resume_settings(log.path)
                resumed_settings["_sciworld_resume_mode"] = mode
                resumed_client = ScriptedClient([failure])
                resumed = EventLog.reopen(log.path)
                try:
                    with patch("ablation.sciworld.time.sleep") as sleep, self.assertRaises(BadRequestError):
                        sciworld.run(resumed_settings, resumed_client, resumed)
                    sleep.assert_not_called()
                finally:
                    resumed.close()
                self.assertEqual(len(resumed_client.prompts), 1)
                self.assertEqual(sum(e["event"] == "attempt_retry" for e in self.records(log.path)), 3)
                (log.path / "checkpoint.json").unlink()
                saved, _, _ = resume_state(log.path / "events.jsonl", settings)
                self.assertEqual(saved[(0, 0)]["api400_first_step_retries"], 3)

    def test_crash_after_first_step_retry_preserves_count_and_completed_round(self):
        for mode in ("auto", "restart-attempt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                settings, log, _ = self.fixture(Path(tmp), rounds=2)
                failure = api_error(code="invalid_prompt")
                client = ScriptedClient(["keep", "replace", "finish", failure])
                emit = log.emit

                def interrupt(event, **fields):
                    emit(event, **fields)
                    if event == "attempt_retry":
                        raise RuntimeError("process interrupted")

                log.emit = interrupt
                with self.assertRaisesRegex(RuntimeError, "process interrupted"):
                    sciworld.run(settings, client, log)
                log.close()
                saved, rounds, manifest = resume_state(log.path / "events.jsonl", settings)
                self.assertEqual(rounds, {0})
                self.assertEqual(saved[(0, 0)]["end"]["reward"], 60)
                self.assertEqual(saved[(1, 0)]["api400_first_step_retries"], 1)
                Checkpointer(log.path, settings).save(saved, rounds, manifest)
                resumed_settings = _resume_settings(log.path)
                resumed_settings["_sciworld_resume_mode"] = mode
                resumed_client = ScriptedClient([failure, "keep", "replace", "finish"])
                resumed = EventLog.reopen(log.path)
                try:
                    with patch("ablation.sciworld.time.sleep") as sleep:
                        sciworld.run(resumed_settings, resumed_client, resumed)
                    sleep.assert_called_once_with(15)
                finally:
                    resumed.close()
                self.assertEqual(len(client.prompts) + len(resumed_client.prompts), 8)
                records = self.records(log.path)
                self.assertEqual(sum(e["event"] == "generation" for e in records), 6)
                self.assertEqual(sum(e["event"] == "attempt_end" for e in records), 2)
                for use_snapshot in (True, False):
                    if not use_snapshot:
                        (log.path / "checkpoint.json").unlink()
                    saved, rounds, _ = resume_state(log.path / "events.jsonl", settings)
                    self.assertEqual(rounds, {0, 1})
                    self.assertEqual([a["end"]["reward"] for a in saved.values()], [60, 60])

    def test_invalid_first_step_retry_markers_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, log, _ = self.fixture(Path(tmp))
            client = ScriptedClient([api_error(code="invalid_prompt"), "keep", "replace", "finish"])
            with patch("ablation.sciworld.time.sleep"):
                sciworld.run(settings, client, log)
            records = self.records(log.path)
            index = next(i for i, e in enumerate(records) if e["event"] == "attempt_retry")
            broken = Path(tmp) / "broken.jsonl"
            for fields in ({"step": 2}, {"status_code": 500}, {"error_code": "invalid_parameter"},
                           {"retry_count": 2}, {"retry_delay_seconds": 30}):
                changed = copy.deepcopy(records)
                changed[index].update(fields)
                broken.write_text("".join(json.dumps(e) + "\n" for e in changed))
                with self.assertRaisesRegex(ValueError, "Invalid ScienceWorld first-step API 400 retry"):
                    resume_state(broken, settings)
            broken.write_text("".join(json.dumps(e) + "\n" for e in records + [records[index]]))
            with self.assertRaisesRegex(ValueError, "Duplicate or late"):
                resume_state(broken, settings)

    def test_non_retryable_first_step_400_and_other_errors_propagate_without_rollback(self):
        for responses in ([api_error()], ["keep", api_error(401)], ["keep", api_error(500)],
                          ["keep", RuntimeError("offline")]):
            with self.subTest(responses=responses), tempfile.TemporaryDirectory() as tmp:
                settings, log, env = self.fixture(Path(tmp))
                with self.assertRaises(type(responses[-1])) as caught:
                    sciworld.run(settings, ScriptedClient(responses), log)
                self.assertIs(caught.exception, responses[-1])
                self.assertEqual(env.resets, 1)
                self.assertTrue(env.closed)
                self.assertFalse(any(e["event"] == "attempt_backtrack" for e in self.records(log.path)))

    def test_backtrack_replay_mismatch_obeys_resume_mode(self):
        for mode in ("auto", "strict"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                settings, log, env = self.fixture(Path(tmp))
                settings["_sciworld_resume_mode"] = mode
                original_step = env.step

                def changed_step(output):
                    observation, state = original_step(output)
                    if env.resets == 2:
                        observation += " changed"
                    return observation, state

                env.step = changed_step
                client = ScriptedClient(["keep", "discard", api_error(), "keep", "replace", "finish"])
                if mode == "strict":
                    with self.assertRaises(sciworld.ReplayMismatchError):
                        sciworld.run(settings, client, log)
                    self.assertEqual(len(client.prompts), 3)
                    attempts, _, _ = resume_state(log.path / "events.jsonl", settings)
                    self.assertEqual(attempts[(0, 0)]["api400_backtracks"], 1)
                else:
                    sciworld.run(settings, client, log)
                    self.assertEqual(env.resets, 3)
                    self.assertEqual(env.actions, ["keep", "replace", "finish"])
                    self.assertEqual(sum(e["event"] == "attempt_restart" for e in self.records(log.path)), 1)
                    (log.path / "checkpoint.json").unlink()
                    attempts, _, _ = resume_state(log.path / "events.jsonl", settings)
                    self.assertEqual(attempts[(0, 0)]["end"]["reward"], 60)

    def test_invalid_or_completed_attempt_backtrack_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings, log, _ = self.fixture(Path(tmp))
            sciworld.run(settings, ScriptedClient(["keep", "discard", api_error(), "replace", "finish"]), log)
            records = self.records(log.path)
            index = next(i for i, e in enumerate(records) if e["event"] == "attempt_backtrack")
            broken = Path(tmp) / "broken.jsonl"
            for fields in ({"step": 1}, {"failed_step": 2}, {"backtrack_count": 2}, {"status_code": 500}):
                changed = copy.deepcopy(records)
                changed[index].update(fields)
                broken.write_text("".join(json.dumps(e) + "\n" for e in changed))
                with self.assertRaisesRegex(ValueError, "Invalid ScienceWorld API 400 backtrack"):
                    resume_state(broken, settings)
            broken.write_text("".join(json.dumps(e) + "\n" for e in records + [records[index]]))
            with self.assertRaisesRegex(ValueError, "current incomplete attempt"):
                resume_state(broken, settings)


if __name__ == "__main__":
    unittest.main()
