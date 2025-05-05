import re
import json
import logging
from typing import Tuple

from scienceworld import ScienceWorldEnv

from ..envs import BaseEnv
from ..tasks import SciWorldTask
# from eval_agent.prompt import prompt_with_icl
from ..utils.datatypes import State
from ..utils.clin_utils import get_best_matched_action_using_sent_transformer
from sentence_transformers import SentenceTransformer
logger = logging.getLogger("agent_frame")

task_all = []
# randomly selected intents for data generation stage 1
import random
import pdb

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


class SciWorldEnv(BaseEnv):
    def __init__(
        self,
        task: SciWorldTask,
        env: ScienceWorldEnv,
        gold_path: bool,
        max_env_steps: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: SciWorldTask = task
        self.env = env
        self.max_steps_dict = max_steps_dict
        self.gold_path = gold_path
        self.state = State()
        self.max_steps = max_env_steps
        self.sent_transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="cpu")
    
    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"(?:.*Action: )?(.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        assert action is not None
        return action
    
    def _check_max_steps(self, observation: str) -> str:
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            observation += "\nTask Failed. You have exceeded the maximum number of steps."
        return observation
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        llm_output = self.parse_action(llm_output)

        valid_actions_list = getattr(self.state, 'valid_actions_list', self.env.get_valid_action_object_combinations())
        valid_actions_list = [x for x in valid_actions_list if 'reset' not in x] # remove reset from valid actions

        if "focus" in llm_output.lower():
            valid_actions_list = [x for x in valid_actions_list if 'focus' in x]
        else:
            valid_actions_list = [x for x in valid_actions_list if 'focus' not in x]

        if 'teleport' in llm_output.lower():
            valid_actions_list = [f"teleport to {x}" for x in ["kitchen", "foundry", "workshop", "bathroom", "outside", "living room", "bedroom", "greenhouse", "art studio", "hallway"]] + \
                [vaction for vaction in valid_actions_list if 'teleport' in vaction]
        best_match_score = 0.0
        action = None
        if len(valid_actions_list) == 0:
            # check "Ambiguous request in observation"
            if "Ambiguous request" in self.state.history[-1]['content']:
                valid_actions_list = [str(x) for x in range(len(self.state.history[-2]['content'].split('\n')[1:]))]

        if len(valid_actions_list) > 0:
            # Time how long it takes to map generated next_action to one of the valid_actions?
            # start = time.time()
            action, topN = get_best_matched_action_using_sent_transformer(
                allowed_actions=valid_actions_list,
                query=llm_output,
                model=self.sent_transformer_model,
                device="cpu"
            )
            # end = time.time()
            # sentenceTransformerRuntimes.append(round(end - start, 2))

            # print("Sentence transformer runtimes: " + str(sentenceTransformerRuntimes))

            # Check top-1 action match score, and if the score < threshold then 
            best_match_score = topN[0][1]

        if not (best_match_score > 0.9 or \
            action in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'] or \
            (len(valid_actions_list) == 0) ):
            old_num_moves = self.env.get_num_moves()
            obs_candidate, *_ = self.env.step(llm_output)
            new_num_moves = self.env.get_num_moves()
            if old_num_moves == new_num_moves:
                action_text = llm_output if len(llm_output) <= 20 else f"{llm_output[:20]}..."
                observation = f"Your generated action '{action_text}' cannot be matched to a valid action."
            else:
                observation = obs_candidate
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.invalid_action_count += 1
            self.state.steps += 1
            self.state.reward = 0
            self.state.reward_history.append(0)
            observation = self._check_max_steps(observation)
            return observation, self.state
        
        assert action is not None, f"Action is None for {llm_output}, valid_actions_list: {valid_actions_list}, best_match_score: {best_match_score}, all_possible_actions: {self.state.valid_actions_list}, previous_observation: {self.state.history[-1]['content']}"
        
        observation, _, done, info = self.env.step(action)
        reward = info['raw_score']
        self.state.valid_actions_list = info['valid']
        observation = f"Observation: {observation}"
        # if self.state.reward is None or reward > self.state.reward:
        self.state.reward = reward
        self.state.reward_history.append(reward)
        self.state.history.append({
            "role": "user",
            "content": f"{observation}",
        })

        self.state.steps += 1
        observation = self._check_max_steps(observation)
        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            observation += "\nTask Successfully Completed."

        return observation, self.state
    
    def legacy_step(self, llm_output: str) -> Tuple[str, State]:
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
            self.state.invalid_action_count += 1
            self.state.steps += 1
            observation = self._check_max_steps(observation)
            return observation, self.state
        try:
            # assert action in [aa['action'] for aa in self.env.get_possible_action_object_combinations()[0]]
            observation, _, done, info = self.env.step(action)
            reward = info['raw_score']
            observation = f"Observation: {observation}"
            # available_actions = self.env.get_available_actions()
            # observation = f"Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}"
        except AssertionError:
            observation = f'Observation: Invalid action! Your action: {action}'
            self.state.invalid_action_count += 1
            done = False

        self.state.history.append({
            "role": "user",
            "content": f"{observation}",
        })

        self.state.steps += 1
        observation = self._check_max_steps(observation)
        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            observation += "\nTask Successfully Completed."

        return observation, self.state


    def reset(self) -> Tuple[str, State]:
        self.state = State()
        if self.max_steps == -1:
            self.max_steps = self.max_steps_dict[self.task.sub_task_name]
        else:
            self.max_steps = self.max_steps
        self.env.load(self.task.sub_task_name, self.task.variation_idx, simplificationStr="easy", generateGoldPath=self.gold_path)
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
