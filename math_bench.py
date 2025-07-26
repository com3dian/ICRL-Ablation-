import pickle
import glob
import time
import os
import re
import json
import sys
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import anyio
from openai import AsyncOpenAI
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from typing import Literal, Dict, List, Any, Optional
# from scienceworld import ScienceWorldEnv as ScienceWorldEnvBase
# from sciworld_armap.utils.replace_sciworld_score import sciworld_monkey_patch
# sciworld_monkey_patch()
# from sciworld_armap.envs.sciworld_env import SciWorldEnv
# from sciworld_armap.tasks.sciworld import SciWorldTask
from omegaconf import OmegaConf
from enum import Enum
import colorama
import copy
import dotenv
import pdb
import traceback
from datasets import load_dataset
from unittest.mock import MagicMock, AsyncMock
import tiktoken
dotenv.load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path to find eval_agent
script_path = Path(__file__).resolve()
parent_dir = str(script_path.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

base_path = os.getcwd()


class Methods(Enum):
    ICRL = "icrl"
    RANDOM_SAMPLING = "random_sampling"
    REFLEXION = "reflexion"
    REACT = "react"
    SELFREFINE = "selfrefine"
    COT = "cot"

@dataclass
class MathConfig:
    output_path: str = "ICL/math/"
    postfix: str = ""
    commit_message: str = os.popen("git log -1 --pretty=%B").read().strip()
    
    # Experiment modes
    icrl_mode: Methods = Methods.ICRL
    debug_run: bool = False
    max_attempts_in_context: Optional[int] = None # Ablation
    zero_out_rewards: bool = False # Ablation
    no_rewards: bool = False # Ablation
    explore_only: bool = False # Ablation
    exploit_only: bool = False # Ablation
    explore_and_exploit: bool = False # Ablation
    neutral_prompt: bool = False # Ablation
    max_reflections_in_context: Optional[int] = None
    react: bool = False
    selfrefine: bool = False
    cot: bool = False
    high_reward_only: bool = False

    # Experiment parameters
    dataset_name: str = "HuggingFaceH4/MATH-500"
    split_name: str = "test"
    num_initial_attempts: int = 2
    num_problems: int = -1 # -1 means all problems
    rounds: int = 40
    
    # Model configuration
    model_name: str = "Qwen/Qwen3-32B"
    vllm_address: str = "http://localhost:11435/v1"
    score_model_name: str = "virtuoussy/Qwen2.5-7B-Instruct-RLVR"
    score_vllm_address: str = "http://localhost:11436/v1"
    score_vllm_context_size: int = 2048
    disable_reasoning: bool = True
    temperature: float = 1.0  

    checkpoint_path: Optional[str] = None
    
    # Prompt templates

    exploration_instruction: str = """
Look at the previous attempts, try to construct a new answer that is different from all of them.
Write your final answer in the format of <answer>...</answer>.
"""

    exploitation_instruction: str = """
Look at the previous attempts and their rewards. Try to construct a new answer that scores higher than all of them.
Write your final answer in the format of <answer>...</answer>.
"""

    explore_and_exploit_instruction: str = """
You get multiple attempts to complete the task. You can see the previous attempts and their rewards.
For this attempt, decide whether to try a completely different approach or to learn and improve on the previous attempts. Then, continue to answer the question.
Write your final answer in the format of <answer>...</answer>.
"""

    do_reflexion_instruction: str = """
"You will be given the history of a past experience in which you encountered a task that required you to provide a response to a prompt aiming to maximize a reward, and you attempted a response. You were unsuccessful in providing an answer that successfully completed the task. Instead of recounting the details of the task itself, focus on analyzing the approach you took and the specific actions or steps you attempted. Based on this reflection, devise a concise, revised plan of action that acknowledges your error and details the exact measures or methods you should have employed. For example, if you attempted steps A and B but overlooked step C, construct a plan that explicitly incorporates step C into your approach. This self-reflection and plan will be essential for when you reattempt the task.
"""

    use_reflexion_instruction: str = """
Your location and the environment is reset now. It's your turn.
Consider the previous reflections about doing the task and try to complete the task.
After thinking, make sure to write your action **exactly** in the "Action: single_action" format. **You can only do one action at a time.**
"""

    do_selfrefine_instruction: str = """
Review your completed attempt for this scientific task. Now, provide detailed feedback on what went wrong:
1. Identify any specific errors or misunderstandings in your approach
2. Analyze which actions were ineffective and why they failed
3. Determine what key steps or objects you missed or used incorrectly

Put your feedback within <feedback>...</feedback> tags.

Then, briefly outline an improved approach that would address these issues for a future attempt. What would you do differently to successfully complete the task?
"""

    use_selfrefine_instruction: str = """
Your location and the environment is reset now. It's your turn.

Consider the feedback provided on previous attempts for this scientific task. Apply the insights from this feedback to improve your approach. Pay special attention to:
1. Correcting the specific errors identified in previous attempts
2. Using more effective actions in the right sequence
3. Focusing on key objects and steps that were missed before

Develop a clear plan that addresses the issues highlighted in the feedback and follows the task instructions correctly.

After thinking through your approach, write your action **exactly** in the "Action: single_action" format. You can only do one action at a time.
"""

    react_instruction: str = """
Your location and the environment is reset now. It's your turn.

Before each action, think through your process step by step. Enclose your reasoning within `<thought>...</thought>` tags so that only you can see it.

After thinking, make sure to write your action **exactly** in the "Action: single_action" format. **You can only do one action at a time.**
"""

    reward_model_instruction: str = """
Given a problem, determine whether the final answer in the provided (incomplete) solution process matches the reference answer.  
The reference answer may be one single option character (e.g., A, B, C, D), a numerical value, an expression, or a list of answers if multiple questions are involved.  
**The reference answer may be in Chinese or another language, but your evaluation should be language-agnostic.**  

Your task:  
- Compare the final output of the solution process with the reference answer.  
- If they **match exactly**, output **YES**.  
- If they **do not match**, output **NO**.  
- If the solution process is unclear, incomplete, or ambiguous, assume it is incorrect and output **NO**.  

Your output must be strictly **'YES'** or **'NO'**, with no additional words, punctuation, or explanation.  

---

**Question:**  
{question}  

**Solution Process (Final Step Only):**  
{response}  

**Reference Answer:**  
{reference}  

**Output:**  
"""

def parse_args():
    # Parse command line arguments
    default_config = OmegaConf.structured(MathConfig)  # Start with code defaults
    cli_conf = OmegaConf.from_cli()
    config = OmegaConf.merge(default_config, cli_conf)
    OmegaConf.set_struct(config, False)
    
    if config.checkpoint_path:
        path = Path(config.checkpoint_path)
        config_path = path / "config.yaml"
        config = OmegaConf.load(config_path)
        OmegaConf.set_struct(config, False)
        config.icrl_mode = Methods(config.icrl_mode.lower())
        config.checkpoint_path = path
        return config
        
    # Runtime modifications

    if config.debug_run:
        # config.num_problems = 1
        config.rounds = 10
        # config.num_initial_attempts = 1
        logger.debug(colorama.Fore.RED + "*"*100)
        logger.debug("Debug run")
        logger.debug("*"*100 + colorama.Fore.RESET)
        
    if config.icrl_mode == Methods.RANDOM_SAMPLING:
        config.num_initial_attempts = 0
        config.max_attempts_in_context = 0
    elif config.icrl_mode == Methods.REFLEXION:
        config.num_initial_attempts = 0
    elif config.icrl_mode == Methods.REACT:
        config.num_initial_attempts = 0
        config.max_attempts_in_context = 0
        config.react = True
    elif config.icrl_mode == Methods.SELFREFINE:
        config.num_initial_attempts = 0
        config.selfrefine = True
        config.no_rewards = True
    elif config.icrl_mode == Methods.COT:
        config.num_initial_attempts = 0
        config.max_attempts_in_context = 0
        config.cot = True

    postfix = datetime.now().strftime("%Y%m%d_%H%M")
    if config.postfix:
        postfix = postfix + "_" + config.postfix
    output_path = Path(base_path) / config.output_path / config.icrl_mode.value / postfix
    config.sw_output_path = str(output_path)

    config.is_openrouter = '/' in config.model_name

    # sanity checks
    assert sum([config.explore_only, config.explore_and_exploit]) <= 1, "Only one of explore_only or explore_and_exploit can be true"
    assert sum([config.no_rewards, config.zero_out_rewards]) <= 1, "Only one of positive_only, no_rewards, or zero_out_rewards can be true"

    # save config
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "config.yaml", "w") as f:
        OmegaConf.save(config, f)
    
    return config

@dataclass
class DataStore:
    @dataclass
    class Attempt:
        raw_prompt: list[dict]
        model_output: str
        reward: float
        round_idx: int
        extra_fields: dict = field(default_factory=dict)

    @dataclass
    class Problem:
        problem: str
        answer: str

    @dataclass
    class ProblemHistory:
        problem: 'DataStore.Problem'
        attempts: list['DataStore.Attempt'] = field(default_factory=list)
    
    problem_histories: list['DataStore.ProblemHistory'] = field(default_factory=list)
    
    def init_problems(self, config):
        dataset = load_dataset(config.dataset_name, split=config.split_name)
        if config.num_problems != -1:
            dataset = dataset.select(range(config.num_problems))
        for i in range(len(dataset)):
            self.problem_histories.append(self.ProblemHistory(
                problem=self.Problem(
                    problem=dataset[i]["problem"],
                    answer=dataset[i]["answer"],
                ),
            ))

    def save_data_snapshot(self, config, filename, delete=None):
        if config.debug_run:
            return
        output_path = Path(config.output_path)
        with open(output_path / filename, "wb") as f:
            pickle.dump(self, f)
        
        # Delete the previous file if specified
        if delete:
            delete_path = output_path / delete
            if delete_path.exists():
                delete_path.unlink()
                logger.info(f"Deleted previous snapshot: {delete}")
    
    @staticmethod
    def load_data_snapshot(config, checkpoint_path): #! todo needs features
        if config.debug_run:
            return None
        filename = find_math_file(checkpoint_path)
        with open(filename, "rb") as f:
            return pickle.load(f)

def mock_reward_client(client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].logprobs = MagicMock()
    mock_response.choices[0].logprobs.token_logprobs = [0.2]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "YES"
    client.chat.completions.create = AsyncMock(return_value=mock_response) 

def mock_client(client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "This is a fake response for debugging. <answer>1</answer>"
    client.chat.completions.create = AsyncMock(return_value=mock_response) 

def extract_answer(output):
    """
    Extract the answer from the output.
    """
    match = re.search(r"<answer>(.*?)</answer>", output)
    if match:
        return match.group(1)
    else:
        # logger.warning(f"{colorama.Fore.YELLOW}No answer found in {colorama.Fore.RESET} {output}")
        return 0

score_client = None
async def get_reward_for_answer(model_output, problem_instance: DataStore.Problem, config: MathConfig):
    global score_client
    if score_client is None:
        score_client = AsyncOpenAI(base_url=config.score_vllm_address)
        if config.debug_run:
            mock_reward_client(score_client)

    reference = problem_instance.answer
    model_answer = extract_answer(model_output)
    if model_answer == reference:
        return 1
    else:
        # truncate the model output to max_model_output_tokens
        
        messages = [{
            "role": "user",
            "content": config.reward_model_instruction.format(
                question=problem_instance.problem,
                response=model_output,
                reference=problem_instance.answer,
            ),
        }]
        
        encoding = tiktoken.encoding_for_model('gpt-4o')
        input_tokens = encoding.encode(messages[0]['content'])
        diff = config.score_vllm_context_size - len(input_tokens)
        print(diff)
        if diff < 100:
            truncated_model_output = encoding.decode(encoding.encode(model_output)[-diff + 100:])
            messages = [{
                "role": "user",
                "content": config.reward_model_instruction.format(
                    question=problem_instance.problem,
                    response=truncated_model_output,
                    reference=problem_instance.answer,
                ),
            }]

        reward_output = await generate_model_output(score_client, config.score_model_name, messages, config, logprobs=True)

        reward_answer = reward_output.choices[0].message.content
        if reward_answer == "YES":
            return np.exp(reward_output.choices[0].logprobs.content[0].logprob)
        elif reward_answer == "NO":
            return 1 - np.exp(reward_output.choices[0].logprobs.content[0].logprob)
        else:
            logger.warning(
                f"{colorama.Fore.YELLOW}Invalid reward answer:{colorama.Fore.RESET} {reward_answer} \n for problem {problem_instance['problem']} \n model output: {model_output}")
            return 0

def merge_same_role_messages(messages):
    merged_messages = []
    for message in messages:
        if merged_messages and merged_messages[-1]["role"] == message["role"]:
            merged_messages[-1]["content"] += "\n" + message["content"]
        else:
            merged_messages.append(message)
    return merged_messages

async def generate_model_output(client: AsyncOpenAI, model_name: str, messages: list[dict], config: MathConfig, **kwargs):
    if config.disable_reasoning:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_reasoning": False,
            },
        }

    output = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=config.temperature,
        **kwargs,
    )
    return output

async def run_evaluation(config: MathConfig, data: DataStore = None):
    """
    This function is the main entry point for the evaluation.
    The previous code is for the sciworld environment. this code is for the math problems. a lot of the code is different. none of the code in the previous run_evaluation function is used.
    """
    if data is None:
        data = DataStore()
        data.init_problems(config)
    
    client = AsyncOpenAI(base_url=config.vllm_address)
    if config.debug_run:
        mock_client(client)

    async def initial_interaction(problem_idx):
        problem_instance = data.problem_histories[problem_idx].problem
        for _ in range(config.num_initial_attempts):
            messages = [
                {"role": "user", "content": f"{problem_instance.problem}"},
            ]
            output = await generate_model_output(client, config.model_name, messages, config)
            model_output = output.choices[0].message.content
            
            reward = await get_reward_for_answer(model_output, problem_instance, config)
            
            attempt = DataStore.Attempt(
                raw_prompt=messages,
                model_output=model_output,
                reward=reward,
                round_idx=-1,
            )
            data.problem_histories[problem_idx].attempts.append(attempt)
    
    async with anyio.create_task_group() as tg:
        for i in range(len(data.problem_histories)):
            tg.start_soon(initial_interaction, i)
    
    data.save_data_snapshot(config, f"data_initial_attempts.json")
    
    rewards = []
    for i in range(len(data.problem_histories)):
        rewards.extend([attempt.reward for attempt in data.problem_histories[i].attempts])
    print(np.percentile(rewards, 25), np.percentile(rewards, 50), np.percentile(rewards, 75))
    
    start_round = 0
    for i in range(len(data.problem_histories)):
        if len(data.problem_histories[i].attempts) > 0:
            start_round = max(start_round, data.problem_histories[i].attempts[-1].round_idx + 1)
    
    for round_idx in range(start_round, config.rounds):
        async def ICRL_interaction(i):
            messages = []
            messages.append({"role": "user", "content": f"{data.problem_histories[i].problem.problem}\n\n"})
            for attempt in data.problem_histories[i].attempts:
                messages.append({"role": "user", "content": f"<Attempt>\n{attempt.model_output}\n**Reward:** {attempt.reward}\n</Attempt>"})
            instruction = config.exploration_instruction if round_idx % 2 == 0 else config.exploitation_instruction
            messages.append({"role": "user", "content": f"\n\n{instruction}"})
            messages = merge_same_role_messages(messages)
            
            output = await generate_model_output(client, config.model_name, messages, config)
            model_output = output.choices[0].message.content
            
            reward = await get_reward_for_answer(model_output, data.problem_histories[i].problem, config)
            
            data.problem_histories[i].attempts.append(DataStore.Attempt(
                raw_prompt=messages,
                model_output=model_output,
                reward=reward,
                round_idx=round_idx,
            ))

        async def reflexion_interaction(question_idx):
            messages = []
            messages.append({"role": "user", "content": f"{data.problem_histories[question_idx].problem.problem}\n\n"})
            for attempt in data.problem_histories[question_idx].attempts:
                messages.append({"role": "user", "content": f"<Reflection>\n{attempt.extra_fields['reflection']}\n**Reward:** {attempt.reward}\n</Reflection>"})
            instruction = config.use_reflexion_instruction if not config.selfrefine else config.use_selfrefine_instruction
            messages.append({"role": "user", "content": f"\n\n{instruction}"})
            messages = merge_same_role_messages(messages)
            
            output = await generate_model_output(client, config.model_name, messages, config)
            model_output = output.choices[0].message.content
            
            reward = await get_reward_for_answer(model_output, data.problem_histories[question_idx].problem, config)
            
            current_attempt = DataStore.Attempt(
                raw_prompt=copy.deepcopy(messages),
                model_output=model_output,
                reward=reward,
                round_idx=round_idx,
            )
            data.problem_histories[question_idx].attempts.append(current_attempt)

            # reflection
            messages.append({"role": "assistant", "content": f"{model_output}\n**Reward:** {reward}\n"})
            instruction = config.do_reflexion_instruction if not config.selfrefine else config.do_selfrefine_instruction
            messages.append({"role": "user", "content": f"{instruction}"})
            current_attempt.extra_fields['reflection_raw_prompt'] = copy.deepcopy(messages)
            
            output = await generate_model_output(client, config.model_name, messages, config)
            model_output = output.choices[0].message.content
            
            if config.selfrefine:
                reflection = "<Attempt>\n"
                reflection += current_attempt.extra_fields['reflection_raw_prompt'][-1]['content']
                reflection += "\n</Attempt>"
                reflection += "\n" + model_output
            else:
                reflection = model_output
            current_attempt.extra_fields['reflection'] = reflection

        
        async with anyio.create_task_group() as tg:
            for i in range(len(data.problem_histories)):
                if config.icrl_mode == Methods.ICRL:
                    f = ICRL_interaction
                elif config.icrl_mode == Methods.REFLEXION or config.icrl_mode == Methods.SELFREFINE:
                    f = reflexion_interaction
                else:
                    raise ValueError(f"Invalid ICRL mode: {config.icrl_mode}")
                tg.start_soon(f, i)
                
        data.save_data_snapshot(
            config,
            f"data_round_{round_idx}_final.json",
            delete=(
                f"data_round_{round_idx-1}_final.json"
                if round_idx > 0
                else "data_initial_attempts.json"
            ),
        )
        
        # print the 25 50 75 percentile of the rewards
        rewards = []
        for i in range(len(data.problem_histories)):
            rewards.extend([attempt.reward for attempt in data.problem_histories[i].attempts])
        print('Round', round_idx, np.percentile(rewards, 25), np.percentile(rewards, 50), np.percentile(rewards, 75))
        

def find_math_file(folder_path):
    """Find the math data file in a given folder."""
    # Look for the most recent round file first
    pattern = os.path.join(folder_path, "data_round_*_final.json")
    files = glob.glob(pattern)
    
    if files:
        # Sort by round number to get the latest
        def extract_round_num(filepath):
            match = re.search(r"data_round_(\d+)_final\.json", os.path.basename(filepath))
            return int(match.group(1)) if match else -1
        
        files.sort(key=extract_round_num, reverse=True)
        return files[0]
    
    # If no round files found, look for initial attempts file
    initial_file = os.path.join(folder_path, "data_initial_attempts.json")
    if os.path.exists(initial_file):
        return initial_file
    
    raise FileNotFoundError(f"No math data file found in {folder_path}")

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    logger.setLevel(logging.INFO)
    
    # Parse command line arguments and get a config object
    config = parse_args()
    data = None
    if config.checkpoint_path:
        data = DataStore.load_data_snapshot(config, config.checkpoint_path)
    await run_evaluation(config, data)

if __name__ == "__main__":
    anyio.run(main)