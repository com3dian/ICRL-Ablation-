"""ScienceWorld correctness and crash recovery checks; no JVM or API calls."""

import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from . import events, sciworld
from . import run as runner
from .client import SurfClient
from .events import EventLog
from .feedback import feedback_language
from .run import _resume_settings, parse_args
from .sciworld_resume import REWARD_POLICY, resume_state


def config():
    return SimpleNamespace(task_prompt_cot="Task {available_actions}", available_actions="look around",
                           exploration_instruction="Explore", exploitation_instruction="Exploit",
                           neutral_round_instruction="Complete the task")


class FakeEnv:
    def __init__(self, sample):
        self.task = SimpleNamespace(sub_task_name=f"task-{sample}", variation_idx=0)
        self.env = self
        self.closed = False

    def taskdescription(self):
        return self.task.sub_task_name

    def get_possible_objects(self):
        return ["cup"]

    def reset(self):
        self.moves = 0

    def step(self, output):
        self.moves += 1
        return f"Observation {self.moves}", SimpleNamespace(
            reward=(30, 0, -130)[self.moves - 1], finished=self.moves == 3)

    def parse_action(self, output):
        return output.removeprefix("Action: ")

    def close(self):
        self.closed = True


class Client:
    def __init__(self):
        self.prompts = []

    def generate(self, model, messages, **kwargs):
        self.prompts.append(copy.deepcopy(messages))
        return "Action: look around"


class ScienceWorldTests(unittest.TestCase):
    def settings(self, polarity="normal"):
        return dict(task="sciworld", samples=2, rounds=2, bootstrap_attempts=1, max_env_steps=3,
                    model="fake", max_tokens=16, history=None, reward_polarity=polarity,
                    random_reward_seed=3, sciworld_reward_policy=REWARD_POLICY)

    def components(self):
        created = []

        def load(count, cfg):
            envs = [FakeEnv(i) for i in range(count)]
            created.extend(envs)
            return envs

        demo = [[{"role": "user", "content": "Example"}]]
        return (config, load, demo), created

    def records(self, log):
        return [json.loads(line) for line in (log.path / "events.jsonl").read_text().splitlines()]

    def test_empty_response_retries_current_step_without_duplicate_actions_or_rewards(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings("empty")
            settings.update(samples=1, rounds=1, bootstrap_attempts=0)
            components, created = self.components()
            requests = []
            responses = iter(("Action: look around", None, "Action: look around", "Action: look around"))

            def create(**kwargs):
                requests.append(copy.deepcopy(kwargs))
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses), model_extra={}),
                                             finish_reason="stop")], usage=None)

            client = SurfClient.__new__(SurfClient)
            client.provider = "cliproxy"
            client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            with patch.object(events, "OUTPUT_ROOT", Path(tmp)), \
                 patch.object(sciworld, "_original_components", return_value=components), \
                 patch("ablation.client.time.sleep") as sleep, \
                 self.assertLogs("ablation.client", level="WARNING"), feedback_language("numeric", empty=True):
                log = EventLog("sciworld", "fake", "numeric", settings)
                try:
                    summary = sciworld.run(settings, client, log)
                    records = self.records(log)
                    resume_state(log.path / "events.jsonl", settings)
                finally:
                    log.close()
            self.assertEqual(summary, {"rounds_completed": 1, "samples": 1})
            self.assertEqual(len(requests), 4)
            self.assertEqual(requests[1], requests[2])
            sleep.assert_called_once_with(5)
            self.assertEqual(created[0].moves, 3)
            self.assertTrue(created[0].closed)
            generations = [e for e in records if e["event"] == "generation"]
            evaluations = [e for e in records if e["event"] == "eval_step"]
            self.assertEqual([e["step"] for e in generations], [1, 2, 3])
            self.assertTrue(all(e["model_response"] == "Action: look around" for e in generations))
            self.assertEqual([e["reward"] for e in evaluations], [30, 0, -130])
            self.assertTrue(all(e["shown_reward"] is None for e in evaluations))
            ends = [e for e in records if e["event"] == "attempt_end"]
            self.assertEqual(len(ends), 1)
            self.assertEqual(ends[0]["reward"], -100)

    def test_resume_matches_uninterrupted_run_at_each_persisted_boundary(self):
        for polarity in ("normal", "flipped", "random", "empty"):
            with self.subTest(polarity=polarity), tempfile.TemporaryDirectory() as tmp:
                settings = self.settings(polarity)
                components, created = self.components()
                with patch.object(events, "OUTPUT_ROOT", Path(tmp)), patch.object(
                        sciworld, "_original_components", return_value=components), feedback_language("fr"):
                    baseline = EventLog("sciworld", "fake", "fr", settings)
                    baseline_client = Client()
                    sciworld.run(settings, baseline_client, baseline)
                    expected = [{k: v for k, v in e.items() if k != "time"} for e in self.records(baseline)]
                    baseline.close()
                    # Includes a pending generation, partial/terminal evaluations,
                    # completed attempts, and a persisted round summary.
                    for crash_event, occurrence in (("generation", 2), ("eval_step", 1),
                                                     ("eval_step", 3), ("attempt_end", 1),
                                                     ("round_end", 1)):
                        with self.subTest(crash=crash_event, occurrence=occurrence):
                            log = EventLog("sciworld", "fake", "fr", settings)
                            emit = log.emit
                            seen = 0

                            def crash(event, **fields):
                                nonlocal seen
                                emit(event, **fields)
                                if event == crash_event:
                                    seen += 1
                                    if seen == occurrence:
                                        raise RuntimeError("simulated interruption")

                            client = Client()
                            log.emit = crash
                            with self.assertRaisesRegex(RuntimeError, "simulated"):
                                sciworld.run(settings, client, log)
                            log.close()
                            resumed = EventLog.reopen(log.path)
                            resumed_settings = _resume_settings(log.path)
                            try:
                                sciworld.run(resumed_settings, client, resumed)
                                actual = [{k: v for k, v in e.items() if k != "time"} for e in self.records(resumed)]
                                self.assertEqual(actual, expected)
                                self.assertEqual(client.prompts, baseline_client.prompts)
                                resume_state(resumed.path / "events.jsonl", resumed_settings)
                            finally:
                                resumed.close()
                    self.assertTrue(all(env.closed for env in created))

    def test_resume_rejects_changed_observation_before_new_api_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings()
            components, created = self.components()
            with patch.object(events, "OUTPUT_ROOT", Path(tmp)), patch.object(
                    sciworld, "_original_components", return_value=components):
                log = EventLog("sciworld", "fake", "numeric", settings)
                original_emit = log.emit

                def interrupt(event, **fields):
                    original_emit(event, **fields)
                    if event == "eval_step":
                        raise RuntimeError("interrupt")

                log.emit = interrupt
                with self.assertRaises(RuntimeError):
                    sciworld.run(settings, Client(), log)
                log.close()
                client = Client()
                resumed = EventLog.reopen(log.path)
                original_step = FakeEnv.step

                def changed_step(env, output):
                    _, state = original_step(env, output)
                    return "Different observation", state

                try:
                    resume_settings = _resume_settings(log.path)
                    resume_settings["_sciworld_resume_mode"] = "strict"
                    with patch.object(FakeEnv, "step", changed_step), self.assertRaisesRegex(ValueError, "replay differs"):
                        sciworld.run(resume_settings, client, resumed)
                    self.assertEqual(client.prompts, [])
                    self.assertTrue(all(env.closed for env in created))
                finally:
                    resumed.close()

    def test_legacy_reward_policy_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(json.dumps({"task": "sciworld"}))
            with self.assertRaisesRegex(ValueError, "corrected reward policy"):
                _resume_settings(path)

    def test_cli_resumes_scienceworld_with_saved_selection_and_provider(self):
        class InterruptedClient(Client):
            def generate(self, model, messages, **kwargs):
                if self.prompts:
                    raise RuntimeError("provider unavailable")
                return super().generate(model, messages, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            components, _ = self.components()
            argv = ["run", "--task", "sciworld", "--provider", "cliproxy", "--model", "gpt6luna",
                    "--feedback", "numeric", "--samples", "1", "--rounds", "1",
                    "--bootstrap-attempts", "0", "--max-env-steps", "3"]
            first_client, resumed_client = InterruptedClient(), Client()
            with patch.object(events, "OUTPUT_ROOT", Path(tmp)), patch.object(
                    sciworld, "_original_components", return_value=components), patch.object(
                    sys, "stdout", io.StringIO()), patch.object(runner, "plot_run", return_value="test.png"):
                with patch.object(sys, "argv", argv), patch.object(runner, "SurfClient", return_value=first_client):
                    with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                        runner.main()
                run_dir = next(Path(tmp).rglob("config.json")).parent
                with patch.object(sys, "argv", ["run", "--resume", str(run_dir)]), patch.object(
                        runner, "SurfClient", return_value=resumed_client) as factory:
                    runner.main()
                    factory.assert_called_once_with(provider="cliproxy")
                summary = json.loads((run_dir / "summary.json").read_text())
                self.assertEqual(summary, {"rounds_completed": 1, "samples": 1})
                self.assertEqual(len(first_client.prompts) + len(resumed_client.prompts), 3)
                records = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
                self.assertEqual(sum(e["event"] == "eval_step" for e in records), 3)
                self.assertEqual(sum(e["event"] == "run_resume" for e in records), 1)

    def test_cli_task_selection_and_resume_overrides(self):
        argv = ["run", "--task", "sciworld", "--model", "qwen35b", "--feedback", "numeric",
                "--sciworld-tasks", "boil", "freeze", "--variation-idx", "2", "--simplifications", ""]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual((args.sciworld_tasks, args.variation_idx, args.simplifications), (["boil", "freeze"], 2, ""))

    def test_resume_rejects_duplicate_and_missing_step_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings()
            components, _ = self.components()
            with patch.object(events, "OUTPUT_ROOT", Path(tmp)), patch.object(
                    sciworld, "_original_components", return_value=components):
                log = EventLog("sciworld", "fake", "numeric", settings)
                sciworld.run(settings, Client(), log)
                records = self.records(log)
                log.close()
            index = next(i for i, event in enumerate(records) if event["event"] == "eval_step")
            broken = Path(tmp) / "broken.jsonl"
            for changed in (records[:index] + [records[index]] + records[index:],
                            records[:index] + records[index + 1:]):
                broken.write_text("".join(json.dumps(e) + "\n" for e in changed))
                with self.assertRaises(ValueError):
                    resume_state(broken, settings)

    def test_exact_task_variation_and_split_selection(self):
        bases = []

        class Base:
            def __init__(self):
                self.closed = False
                bases.append(self)

            def get_task_names(self):
                return ["b", "a", "c"]

            def getMaxVariations(self, name):
                return 4

            def load(self, name, variation, simplificationStr):
                self.loaded = (name, variation, simplificationStr)

            def getVariationsTest(self):
                return [2, 3]

            def close(self):
                self.closed = True

        def wrapper(**kwargs):
            return SimpleNamespace(**kwargs, reset=lambda: None)

        cfg = SimpleNamespace(max_env_steps=3, task_names=["a", "c"], variation_idx=2,
                              simplifications="", sciworld_split="test")
        envs = sciworld._load_envs_compat(2, cfg, Base, wrapper, lambda **kw: SimpleNamespace(**kw))
        self.assertEqual([e.task.sub_task_name for e in envs], ["a", "c"])
        self.assertEqual([e.task.variation_idx for e in envs], [2, 2])
        self.assertEqual([e.env.loaded for e in envs], [("a", 2, ""), ("c", 2, "")])
        for env in envs:
            env.env.close()
        cfg.variation_idx = 0
        with self.assertRaisesRegex(ValueError, "not in the test split"):
            sciworld._load_envs_compat(2, cfg, Base, wrapper, lambda **kw: SimpleNamespace(**kw))
        self.assertTrue(all(base.closed for base in bases))

    def test_compat_cleanup_is_idempotent(self):
        shutdowns = []

        class Base:
            def __init__(self):
                self._gateway = SimpleNamespace(shutdown=lambda: shutdowns.append(True))

        for name in sciworld_aliases():
            setattr(Base, name, lambda self: [])
        sciworld._install_scienceworld_compat(Base)
        sciworld._install_scienceworld_compat(Base)
        env = Base()
        self.assertIn(env, Base._icrl_instances)
        env.close()
        env.close()
        self.assertEqual(shutdowns, [True])
        self.assertEqual(Base._icrl_instances, set())

    def test_loader_closes_all_jvms_when_wrapper_initialization_fails(self):
        bases = []

        class Base:
            def __init__(self):
                self.closed = False
                bases.append(self)

            def get_task_names(self):
                return ["a", "b", "c"]

            def close(self):
                self.closed = True

        def wrapper(**kwargs):
            if len(bases) == 2:
                raise RuntimeError("matcher initialization failed")
            return SimpleNamespace(reset=lambda: None)

        with self.assertRaisesRegex(RuntimeError, "matcher"):
            sciworld._load_envs_compat(2, SimpleNamespace(max_env_steps=3), Base, wrapper,
                                      lambda **kwargs: SimpleNamespace(**kwargs))
        self.assertEqual(len(bases), 2)
        self.assertTrue(all(base.closed for base in bases))

    def test_failure_reward_telescopes_to_minus_100(self):
        sciworld._install_scienceworld_compat(type("Aliases", (), {
            name: lambda self: [] for name in sciworld_aliases()}))
        from sciworld_armap.utils.replace_sciworld_score import step
        env = SimpleNamespace(server=SimpleNamespace(step=lambda _: "failed", getScore=lambda: -1,
                                                     getCompleted=lambda: False), lastStepScore=30,
                              get_num_moves=lambda: 3, look=lambda: "", inventory=lambda: "",
                              taskdescription=lambda: "", get_valid_action_object_combinations=lambda: [],
                              variationIdx=0, taskName="task", simplificationStr="easy")
        observation, reward, finished, info = step(env, "focus on wrong object")
        self.assertEqual(30 + reward, -100)
        self.assertEqual(info["reward"], -130)
        self.assertTrue(finished)


def sciworld_aliases():
    return ("getTaskNames", "getNumMoves", "getValidActionObjectCombinations", "getPossibleObjects")


class ScienceWorldActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(sciworld.SCI_DIR) not in sys.path:
            sys.path.insert(0, str(sciworld.SCI_DIR))
        from sciworld_armap.envs import sciworld_env
        from sciworld_armap.utils.datatypes import State
        cls.wrapper_module = sciworld_env
        cls.State = State

    def wrapper(self, valid=(), observation=""):
        env = self.wrapper_module.SciWorldEnv.__new__(self.wrapper_module.SciWorldEnv)
        env.state = self.State()
        env.max_steps = 15
        env.sent_transformer_model = None
        env.env = SimpleNamespace(
            step=Mock(return_value=("resolved", 0, False, {"reward": 0, "valid": ["look around"]})),
            get_valid_action_object_combinations=Mock(return_value=list(valid)),
            get_num_moves=Mock(return_value=1),
        )
        if observation:
            env.state.history.append({"role": "user", "content": observation})
            env.state.valid_actions_list = list(valid)
        return env

    def ambiguity(self):
        return ("Observation: Ambiguous request: Please enter the number for the action you intended "
                "(or blank to cancel):\n0:\tfocus on apple seed (in seed jar, in kitchen)\n"
                "1:\tfocus on apple seed (in seed jar, in kitchen)\n")

    def test_repeated_focus_after_ambiguity_executes_number_instead_of_none(self):
        env = self.wrapper(["0", "1"], self.ambiguity())
        observation, state = env.step("Action: focus on apple seed (in seed jar, in kitchen)")
        env.env.step.assert_called_once_with("0")
        env.env.get_valid_action_object_combinations.assert_not_called()
        self.assertEqual(observation, "Observation: resolved")
        self.assertEqual((state.steps, state.reward), (1, 0))

    def test_numbered_choices_are_not_limited_to_twenty(self):
        env = self.wrapper(["25"], self.ambiguity() + "25:\tfocus on another seed\n")
        with patch.object(self.wrapper_module, "get_best_matched_action_using_sent_transformer") as matcher:
            env.step("Thought: choose the last option\nAction: 25")
        env.env.step.assert_called_once_with("25")
        matcher.assert_not_called()

    def test_invalid_choice_preserves_clarification_for_next_response(self):
        env = self.wrapper(["0", "1"], self.ambiguity())
        observation, state = env.step("Action: 9")
        env.env.step.assert_not_called()
        self.assertIn(self.ambiguity(), observation)
        self.assertEqual((state.invalid_action_count, state.reward), (1, 0))
        env.step("Action: 1")
        env.env.step.assert_called_once_with("1")

    def test_unrelated_action_does_not_silently_choose_a_number(self):
        env = self.wrapper(["0", "1"], self.ambiguity())
        with patch.object(self.wrapper_module, "get_best_matched_action_using_sent_transformer",
                          return_value=("focus on apple seed (in seed jar, in kitchen)", [("seed", 0.1)])):
            observation, state = env.step("Action: go to outside")
        env.env.step.assert_not_called()
        self.assertIn("Ambiguous request", observation)
        self.assertEqual(state.invalid_action_count, 1)

    def test_blank_response_cancels_ambiguity(self):
        for response in ("", "Action:", "Thought: cancel this selection\nAction: "):
            with self.subTest(response=response):
                env = self.wrapper(["0", "1"], self.ambiguity())
                env.step(response)
                env.env.step.assert_called_once_with("")

    def test_empty_action_space_and_blank_actions_do_not_crash(self):
        for valid, action in (([], "Action: focus on seed"), ([], "Action: look around"),
                              (["look around"], ""), (["look around"], "Action:")):
            with self.subTest(valid=valid, action=action):
                env = self.wrapper(valid)
                env.max_steps = 1
                observation, state = env.step(action)
                env.env.step.assert_not_called()
                self.assertTrue(state.finished)
                self.assertEqual((state.steps, state.reward, state.invalid_action_count), (1, 0, 1))
                self.assertIn("maximum number of steps", observation)

    def test_invalid_raw_action_still_counts_toward_step_budget(self):
        env = self.wrapper(["look around"])
        with patch.object(self.wrapper_module, "get_best_matched_action_using_sent_transformer",
                          return_value=("look around", [("look around", 0.1)])):
            observation, state = env.step("Action: invent an object")
        env.env.step.assert_called_once_with("invent an object")
        self.assertIn("cannot be matched", observation)
        self.assertEqual((state.steps, state.reward, state.invalid_action_count), (1, 0, 1))

    def test_raw_action_preserves_clarification_reward_and_termination_without_a_move(self):
        for observation, done, reward in ((self.ambiguity(), False, 0),
                                          ("Task Failed.", True, -100),
                                          ("Subgoal achieved.", False, 10)):
            with self.subTest(observation=observation):
                env = self.wrapper(["focus on seed jar"])
                env.env.step.return_value = (observation, reward, done, {"reward": reward, "valid": ["0", "1"]})
                with patch.object(self.wrapper_module, "get_best_matched_action_using_sent_transformer",
                                  return_value=("focus on seed jar", [("focus on seed jar", 0.1)])):
                    result, state = env.step("Action: focus on seed")
                self.assertIn(observation, result)
                self.assertEqual((state.reward, state.finished, state.invalid_action_count), (reward, done, 0))
                self.assertEqual(state.valid_actions_list, ["0", "1"])

    def test_success_on_last_allowed_step_is_not_labeled_as_timeout(self):
        env = self.wrapper(["look around"])
        env.max_steps = 1
        env.env.step.return_value = ("Task Successfully Completed.", 100, True,
                                     {"reward": 100, "valid": []})
        observation, state = env.step("Action: look around")
        self.assertEqual((state.steps, state.reward, state.finished), (1, 100, True))
        self.assertNotIn("Task Failed", observation)

    def test_java_error_remains_readable_after_gateway_closes(self):
        from sciworld_armap.utils import replace_sciworld_score as scoring
        closed = False

        class LazyJavaError(Exception):
            def __str__(self):
                if closed:
                    raise RuntimeError("Gateway is not connected")
                return "java.lang.NullPointerException: original simulator error"

        env = SimpleNamespace(server=SimpleNamespace(step=Mock(side_effect=LazyJavaError())))
        with patch.object(scoring, "Py4JJavaError", LazyJavaError):
            with self.assertRaises(RuntimeError) as raised:
                scoring.step(env, "focus on seed")
        closed = True
        self.assertIn("java.lang.NullPointerException", str(raised.exception))
        self.assertIn("focus on seed", str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)

    def test_non_string_actions_never_reach_java(self):
        from sciworld_armap.utils.replace_sciworld_score import step
        env = SimpleNamespace(server=SimpleNamespace(step=Mock()))
        with self.assertRaisesRegex(TypeError, "must be strings"):
            step(env, None)
        env.server.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
