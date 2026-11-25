"""Opt-in local simulator checks; cached MiniLM and Java required, no model API.

SCIWORLD_LIVE=1 python -m unittest ablation.test_sciworld_live
"""

import os
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import sciworld
from . import events
from .events import EventLog
from .run import _resume_settings
from .sciworld_resume import REWARD_POLICY, resume_state


@unittest.skipUnless(os.environ.get("SCIWORLD_LIVE") == "1", "set SCIWORLD_LIVE=1 to run Java checks")
class ScienceWorldLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
        except PermissionError as exc:
            raise unittest.SkipTest("ScienceWorld requires local TCP sockets; this sandbox blocks them") from exc
        environment = patch.dict(os.environ, {
            "PATH": f"{sys.prefix}/bin:{os.environ.get('PATH', '')}",
            "JAVA_HOME": sys.prefix, "HF_HUB_OFFLINE": "1",
            "JAVA_TOOL_OPTIONS": "-XX:ActiveProcessorCount=1 -Xmx1024m",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
        })
        environment.start()
        cls.addClassCleanup(environment.stop)
        # Reuse one matcher and one JVM to keep this diagnostic small.
        sciworld._original_components()
        from scienceworld import ScienceWorldEnv
        from sentence_transformers import SentenceTransformer
        from sciworld_armap.envs import sciworld_env
        from sciworld_armap.tasks.sciworld import SciWorldTask

        cls.wrapper_module = sciworld_env
        cls.task_class = SciWorldTask
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu",
                                    local_files_only=True)
        cls.matcher_patch = patch.object(sciworld_env, "SentenceTransformer", return_value=model)
        cls.matcher_patch.start()
        cls.addClassCleanup(cls.matcher_patch.stop)
        cls.base = ScienceWorldEnv()
        cls.addClassCleanup(cls.base.close)

    def wrapper(self, task, max_steps=15):
        env = self.wrapper_module.SciWorldEnv(
            task=self.task_class(task_id=f"{task}-0", sub_task_name=task, variation_idx=0),
            env=self.base, max_env_steps=max_steps, simplifications="easy")
        env.reset()
        return env

    def test_all_tasks_reset_invalid_action_and_step_limit(self):
        tasks = sorted(self.base.get_task_names())
        self.assertGreaterEqual(len(tasks), 29)
        for task in tasks:
            with self.subTest(task=task):
                env = self.wrapper(task, max_steps=3)
                observation, state = env.step("Action: look around")
                self.assertIsInstance(observation, str)
                self.assertIsInstance(state.reward, int)
                self.assertFalse(state.finished)
                _, state = env.step("Action:")
                self.assertEqual((state.reward, state.invalid_action_count), (0, 1))
                observation, state = env.step("Action: look around")
                self.assertTrue(state.finished)
                self.assertIn("maximum number of steps", observation)
                env.reset()
                self.assertEqual(env.state.steps, 0)
                self.assertEqual(env.state.history, [])
                observation, state = env.step("Action: invent an impossible object")
                self.assertEqual((state.reward, state.invalid_action_count), (0, 1))
                print(f"PASS {task}: reset, step, invalid actions, timeout", flush=True)

    def test_grow_plant_seed_clarification(self):
        command = "focus on apple seed"
        cases = [(response, confidence)
                 for response in ("Action: focus on apple seed (in seed jar, in kitchen)", "Action: 0", "Action:")
                 for confidence in (1.0, 0.1)]
        for response, confidence in cases:
            with self.subTest(response=response, confidence=confidence):
                env = self.wrapper("grow-plant")
                env.step("Action: teleport to kitchen")
                # Force the exact command that triggers the simulator's dialog,
                # independent of similarity ties between seed referents.
                with patch.object(self.wrapper_module, "get_best_matched_action_using_sent_transformer",
                                  return_value=(command, [(command, confidence)])):
                    observation, _ = env.step(f"Action: {command}")
                self.assertIn("Ambiguous request", observation)
                observation, state = env.step(response)
                self.assertNotIn("Ambiguous request", observation)
                self.assertIsInstance(state.reward, int)
                print(f"PASS grow-plant clarification (match={confidence}): {response!r} -> {observation!r}", flush=True)

    def test_checkpoint_resumes_with_a_new_jvm_without_model_api(self):
        class OfflineClient:
            def __init__(self):
                self.calls = 0

            def generate(self, *args, **kwargs):
                self.calls += 1
                return "Action: look around"

        settings = dict(task="sciworld", model="offline", samples=1, rounds=2,
                        bootstrap_attempts=0, max_env_steps=2, max_tokens=16,
                        history=None, reward_polarity="normal", sciworld_reward_policy=REWARD_POLICY,
                        sciworld_tasks=["chemistry-mix-paint-tertiary-color"])
        with tempfile.TemporaryDirectory() as tmp, patch.object(events, "OUTPUT_ROOT", Path(tmp)):
            log = EventLog("sciworld", "offline", "numeric", settings)
            emit, seen = log.emit, 0

            def interrupt(event, **fields):
                nonlocal seen
                emit(event, **fields)
                if event == "eval_step":
                    seen += 1
                    if seen == 3:
                        raise RuntimeError("simulated process crash")

            log.emit = interrupt
            try:
                with self.assertRaisesRegex(RuntimeError, "simulated process crash"):
                    sciworld.run(settings, OfflineClient(), log)
            finally:
                log.close()
            before = (log.path / "events.jsonl").read_bytes()
            resumed, client = EventLog.reopen(log.path), OfflineClient()
            try:
                summary = sciworld.run(_resume_settings(log.path), client, resumed)
                resumed.finish(summary)
            finally:
                resumed.close()
            self.assertTrue((log.path / "events.jsonl").read_bytes().startswith(before))
            records = [json.loads(line) for line in (log.path / "events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(e["event"] == "attempt_end" for e in records), 2)
            self.assertIn(client.calls, (1, 2))  # Exact replay or one restarted unfinished attempt.
            self.assertEqual(summary, {"rounds_completed": 2, "samples": 1})
            resume_state(log.path / "events.jsonl", _resume_settings(log.path))
            print("PASS checkpoint recovery across JVM restart: completed score retained, no model API", flush=True)


if __name__ == "__main__":
    unittest.main()
