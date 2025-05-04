import re
import json
import logging
from typing import Tuple

from scienceworld import ScienceWorldEnv

from ..envs import BaseEnv
from ..tasks import SciWorldTask
# from eval_agent.prompt import prompt_with_icl
from ..utils.datatypes import State


logger = logging.getLogger("agent_frame")

task_all = []
# randomly selected intents for data generation stage 1
import random
# file_path = '/nobackup/users/zfchen/cdl/ScienceWorld_Planning/eval_agent/data/sciworld/test_indices.json'
import pdb

file_path = "/nobackup/users/zfchen/cdl/ScienceWorld_Planning/eval_agent/data/sciworld/dev_sampled_50.json"
# max_steps_dict = {
#     "task-1-boil": 100,
#     "task-1-change-the-state-of-matter-of": 80,
#     "task-1-freeze": 80,
#     "task-1-melt": 80,
#     "task-10-measure-melting-point-(known-substance)": 120,
#     "task-10-use-thermometer": 30,
#     "task-2-power-component": 20,
#     "task-2-power-component-(renewable-vs-nonrenewable-energy)": 30,
#     "task-2a-test-conductivity": 30,
#     "task-2a-test-conductivity-of-unknown-substances": 30,
#     "task-3-find-animal": 15,
#     "task-3-find-living-thing": 15,
#     "task-3-find-non-living-thing": 15,
#     "task-3-find-plant": 15,
#     "task-4-grow-fruit": 60,
#     "task-4-grow-plant": 30,
#     "task-5-chemistry-mix": 60,
#     "task-5-chemistry-mix-paint-(secondary-color)": 15,
#     "task-5-chemistry-mix-paint-(tertiary-color)": 30,
#     "task-6-lifespan-(longest-lived)": 10,
#     "task-6-lifespan-(longest-lived-then-shortest-lived)": 12,
#     "task-6-lifespan-(shortest-lived)": 10,
#     "task-7-identify-life-stages-1": 30,
#     "task-7-identify-life-stages-2": 30
# }
max_steps_dict = {
    "boil": 100,
    "change-the-state-of-matter-of": 80,
    "freeze": 80,
    "melt": 80,
    "measure-melting-point-known-substance": 120,
    "use-thermometer": 30,
    "power-component": 20,
    "power-component-renewable-vs-nonrenewable-energy": 30,
    "test-conductivity": 30,
    "test-conductivity-of-unknown-substances": 30,
    "find-animal": 15,
    "find-living-thing": 15,
    "find-non-living-thing": 15,
    "find-plant": 15,
    "grow-fruit": 60,
    "grow-plant": 30,
    "chemistry-mix": 60,
    "chemistry-mix-paint-secondary-color": 15,
    "chemistry-mix-paint-tertiary-color": 30,
    "lifespan-longest-lived": 10,
    "lifespan-longest-lived-then-shortest-lived": 12,
    "lifespan-shortest-lived": 10,
    "identify-life-stages-1": 30,
    "identify-life-stages-2": 30,
    "inclined-plane-determine-angle": 40,
    "inclined-plane-friction-named-surfaces": 40,
    "inclined-plane-friction-unnamed-surfaces": 40,
    "measure-melting-point-unknown-substance": 120,
    "mendelian-genetics-known-plant": 50,
    "mendelian-genetics-unknown-plant": 50
}

# with open(file_path, 'r') as file:
#     train_indices_sampled_taskDesc = json.load(file) # 440 samples -> 1483 samples

class SciWorldEnv(BaseEnv):
    def __init__(
        self,
        task: SciWorldTask,
        env: ScienceWorldEnv,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: SciWorldTask = task
        self.env = env
        # f = open("data/sciworld/max_steps.json")
        self.max_steps_dict = max_steps_dict
        # self.max_steps_dict = json.load(open("data/sciworld/max_steps.json"))
        
        self.state = State()
        
        # self.task_all = []
    
    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        assert action is not None
        return action
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })

        try:
            action = self.parse_action(llm_output)
        except:
            observation = f"Observation: Invalid format. The input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            self.state.reward = 0
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
            return observation, self.state
        try:
            observation, _, done, info = self.env.step(action)
            reward = info['raw_score']
            observation = f"Observation: {observation}"
            if self.state.reward is None or reward > self.state.reward:
                self.state.reward = reward
            # available_actions = self.env.get_available_actions()
            # observation = f"Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}"
        except AssertionError:
            observation = 'Observation: Invalid action!'
            done = False

        self.state.history.append({
            "role": "user",
            "content": f"{observation}",
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            # self.state.reward = reward

        return observation, self.state

    def reset(self) -> Tuple[str, State]:
        self.state = State()
        self.max_steps = self.max_steps_dict[self.task.sub_task_name]
        self.env.load(self.task.sub_task_name, self.task.variation_idx, simplificationStr="easy", generateGoldPath=False)
        obs, info = self.env.reset()
        # ! NOTE need to revise when sampling for data-fake or test-set generation !! NOTE
        cur_task = info['taskDesc'] # it is intent, Task Description:\nYour task is to boil lead. For compounds without a boiling point, ...
        # import pdb; pdb.set_trace()
        #################################
        # task_all.append(cur_task)
        # if len(task_all) == 1483: # 440 before
        #     # save file
        #     save_path = '/home/rs4110/code/ScienceWorld_Planning/eval_agent/data/sciworld/train_indices_sampled_taskDesc2.json'
        #     with open(save_path, "w") as fh:
        #         json.dump(task_all, fh)
        # self.state.finished = True
        # cur_task = random.choice(train_indices_sampled_taskDesc)
        # import pdb; pdb.set_trace()
        #################################
        # observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 1)
        # if self.icl_format == 'first':
        #     self.state.history.append({
        #         "role": "user",
        #         "content": observation,
        #     })
        # elif self.icl_format == 'conversation':
        #     self.state.history = messages
        # return observation, self.state
        return obs, info
