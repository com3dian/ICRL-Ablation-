"""Checkpoint durability and recovery across repeated process interruptions."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import events, sciworld
from .events import EventLog
from .run import _resume_settings, parse_args
from .sciworld_resume import resume_state
from . import test_sciworld as fixtures

Client, FakeEnv = fixtures.Client, fixtures.FakeEnv


class CheckpointTests(unittest.TestCase):
    def fixture(self, root):
        fixture = fixtures.ScienceWorldTests()
        settings = fixture.settings()
        components, _ = fixture.components()
        self.enterContext(patch.object(events, "OUTPUT_ROOT", root))
        self.enterContext(patch.object(sciworld, "_original_components", return_value=components))
        return settings

    def interrupt(self, log, settings, event_kind, occurrence):
        emit, seen = log.emit, 0

        def emit_and_interrupt(event, **fields):
            nonlocal seen
            emit(event, **fields)
            if event == event_kind:
                seen += 1
                if seen == occurrence:
                    raise RuntimeError("process interrupted")

        log.emit = emit_and_interrupt
        with self.assertRaisesRegex(RuntimeError, "process interrupted"):
            sciworld.run(settings, Client(), log)
        log.close()

    def records(self, path):
        return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]

    def test_snapshot_plus_tail_matches_full_log_without_repeating_completed_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.fixture(Path(tmp))
            log = EventLog("sciworld", "fake", "numeric", settings)
            self.interrupt(log, settings, "eval_step", 4)
            snapshot = json.loads((log.path / "checkpoint.json").read_text())
            self.assertEqual(len(snapshot["attempts"]), 1)
            self.assertLess(snapshot["event_offset"], (log.path / "events.jsonl").stat().st_size)
            cached = resume_state(log.path / "events.jsonl", _resume_settings(log.path))
            (log.path / "checkpoint.json").unlink()
            rebuilt = resume_state(log.path / "events.jsonl", _resume_settings(log.path))
            for state in (cached, rebuilt):
                self.assertEqual(len(state[0]), 2)
                self.assertEqual(state[0][(-1, 0)]["end"]["reward"], -100)
                self.assertEqual(len(state[0][(-1, 1)]["steps"]), 1)
            client = Client()
            with_context = EventLog.reopen(log.path)
            try:
                sciworld.run(_resume_settings(log.path), client, with_context)
            finally:
                with_context.close()
            self.assertEqual(len(client.prompts), 14)  # 18 total responses, four already saved.
            snapshot = json.loads((log.path / "checkpoint.json").read_text())
            self.assertEqual(len(snapshot["attempts"]), 6)
            self.assertEqual(snapshot["rounds"], [0, 1])

    def test_auto_restarts_only_partial_attempt_and_survives_another_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.fixture(Path(tmp))
            log = EventLog("sciworld", "fake", "numeric", settings)
            self.interrupt(log, settings, "eval_step", 4)
            before = self.records(log.path)
            original_step = FakeEnv.step

            def changed_step(env, output):
                observation, state = original_step(env, output)
                return observation + " changed", state

            # The second interruption happens after the fallback decision but
            # before a fresh response. A third process must accept that marker.
            resumed = EventLog.reopen(log.path)
            with patch.object(FakeEnv, "step", changed_step):
                self.interrupt(resumed, _resume_settings(log.path), "attempt_restart", 1)
            state = resume_state(log.path / "events.jsonl", _resume_settings(log.path))
            self.assertIsNone(state[0][(-1, 1)]["end"])
            self.assertEqual(state[0][(-1, 1)]["steps"], [])
            client = Client()
            resumed = EventLog.reopen(log.path)
            try:
                with patch.object(FakeEnv, "step", changed_step):
                    sciworld.run(_resume_settings(log.path), client, resumed)
            finally:
                resumed.close()
            after = self.records(log.path)
            self.assertEqual(after[:len(before)], before)
            self.assertEqual(len(client.prompts), 15)
            self.assertEqual(sum(e["event"] == "attempt_end" for e in after), 6)
            self.assertEqual(sum(e["event"] == "attempt_restart" for e in after), 1)
            self.assertEqual(sum(e["event"] == "round_end" for e in after), 2)
            resume_state(log.path / "events.jsonl", _resume_settings(log.path))
            (log.path / "checkpoint.json").unlink()
            resume_state(log.path / "events.jsonl", _resume_settings(log.path))

    def test_explicit_restart_discards_pending_response_without_replaying_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.fixture(Path(tmp))
            log = EventLog("sciworld", "fake", "numeric", settings)
            self.interrupt(log, settings, "generation", 4)
            resumed_settings = _resume_settings(log.path)
            resumed_settings["_sciworld_resume_mode"] = "restart-attempt"
            resumed = EventLog.reopen(log.path)
            client = Client()
            try:
                sciworld.run(resumed_settings, client, resumed)
            finally:
                resumed.close()
            self.assertEqual(len(client.prompts), 15)
            self.assertEqual(sum(e["event"] == "attempt_restart" for e in self.records(log.path)), 1)
            (log.path / "checkpoint.json").unlink()
            resume_state(log.path / "events.jsonl", resumed_settings)

    def test_pending_replay_state_survives_two_successive_process_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.fixture(Path(tmp))
            log = EventLog("sciworld", "fake", "numeric", settings)
            self.interrupt(log, settings, "eval_step", 1)
            resumed = EventLog.reopen(log.path)
            self.interrupt(resumed, _resume_settings(log.path), "generation", 1)
            resumed = EventLog.reopen(log.path)
            client = Client()
            try:
                sciworld.run(_resume_settings(log.path), client, resumed)
            finally:
                resumed.close()
            self.assertEqual(len(client.prompts), 16)
            records = self.records(log.path)
            self.assertEqual(sum(e["event"] == "generation" for e in records), 18)
            self.assertEqual(sum(e["event"] == "eval_step" for e in records), 18)
            self.assertEqual(sum(e["event"] == "attempt_end" for e in records), 6)
            (log.path / "checkpoint.json").unlink()
            resume_state(log.path / "events.jsonl", settings)

    def test_corrupt_snapshot_is_rebuilt_but_inconsistent_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.fixture(Path(tmp))
            log = EventLog("sciworld", "fake", "numeric", settings)
            self.interrupt(log, settings, "eval_step", 4)
            checkpoint = log.path / "checkpoint.json"
            original = checkpoint.read_bytes()
            for invalid in ('{"version":', 'null'):
                checkpoint.write_text(invalid)
                with self.assertWarnsRegex(RuntimeWarning, "rebuilding"):
                    attempts, _, _ = resume_state(log.path / "events.jsonl", settings)
                self.assertEqual(len(attempts), 2)
            checkpoint.write_bytes(original)
            records = self.records(log.path)
            next(e for e in records if e["event"] == "eval_step")["reward"] = 999
            (log.path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in records))
            with self.assertWarnsRegex(RuntimeWarning, "rebuilding"), self.assertRaisesRegex(ValueError, "inconsistent"):
                resume_state(log.path / "events.jsonl", settings)

    def test_interrupted_atomic_replace_leaves_prior_snapshot_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.fixture(Path(tmp))
            log = EventLog("sciworld", "fake", "numeric", settings)
            replace = events.os.replace
            calls = 0

            def crash_replace(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("interrupted checkpoint write")
                return replace(source, target)

            try:
                with patch.object(events.os, "replace", side_effect=crash_replace):
                    with self.assertRaisesRegex(OSError, "interrupted checkpoint"):
                        sciworld.run(settings, Client(), log)
            finally:
                log.close()
            attempts, _, _ = resume_state(log.path / "events.jsonl", settings)
            self.assertIsNotNone(attempts[(-1, 0)]["end"])
            self.assertEqual(list(log.path.glob(".checkpoint-*")), [])

    def test_interrupted_summary_write_does_not_mark_run_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.fixture(Path(tmp))
            log = EventLog("sciworld", "fake", "numeric", settings)
            try:
                sciworld.run(settings, Client(), log)
                with patch.object(events.os, "replace", side_effect=OSError("interrupted summary")):
                    with self.assertRaisesRegex(OSError, "interrupted summary"):
                        log.finish({"rounds_completed": 2, "samples": 2})
            finally:
                log.close()
            self.assertFalse((log.path / "summary.json").exists())
            resumed = EventLog.reopen(log.path)
            client = Client()
            try:
                summary = sciworld.run(_resume_settings(log.path), client, resumed)
                resumed.finish(summary)
            finally:
                resumed.close()
            self.assertEqual(client.prompts, [])
            self.assertEqual(json.loads((log.path / "summary.json").read_text()), summary)

    def test_torn_final_record_is_archived_and_valid_unterminated_record_is_kept(self):
        for fragment in (b'{"event":"generation","model_response":"partial',
                         b'{"event":"generation","model_response":"\xf0\x9f',
                         b'{"event":"run_error","error":"offline"}'):
            with self.subTest(fragment=fragment), tempfile.TemporaryDirectory() as tmp:
                settings = self.fixture(Path(tmp))
                log = EventLog("sciworld", "fake", "numeric", settings)
                self.interrupt(log, settings, "eval_step", 4)
                with (log.path / "events.jsonl").open("ab") as stream:
                    stream.write(fragment)
                resumed = EventLog.reopen(log.path)
                resumed.close()
                archives = list(log.path.glob("events-truncated-*.bin"))
                if fragment.endswith(b"}"):
                    self.assertEqual(archives, [])
                    self.assertEqual(self.records(log.path)[-1]["event"], "run_error")
                else:
                    self.assertEqual(len(archives), 1)
                    self.assertEqual(archives[0].read_bytes(), fragment)
                    self.assertEqual(self.records(log.path)[-1]["event"], "log_tail_recovered")
                resume_state(log.path / "events.jsonl", settings)

    def test_restart_cannot_remove_a_completed_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.fixture(Path(tmp))
            log = EventLog("sciworld", "fake", "numeric", settings)
            self.interrupt(log, settings, "attempt_end", 1)
            with (log.path / "events.jsonl").open("a") as stream:
                stream.write(json.dumps({"event": "attempt_restart", "round": -1, "sample_id": 0}) + "\n")
            with self.assertRaisesRegex(ValueError, "current incomplete attempt"):
                resume_state(log.path / "events.jsonl", settings)

    def test_resume_mode_requires_resume(self):
        with patch.object(sys, "argv", ["run", "--sciworld-resume-mode", "strict"]):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
