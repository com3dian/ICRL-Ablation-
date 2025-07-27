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
import openai
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
from transformers import AutoTokenizer

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
    dataset_name: str = "MathArena/aime_2025"
    split_name: str = "train"
    num_initial_attempts: int = 2
    num_problems: int = -1 # -1 means all problems
    rounds: int = 40
    
    # Model configuration
    model_name: str = "Qwen/Qwen3-32B"
    # vllm_address: str = "http://localhost:11435/v1"
    vllm_address: str = "https://openrouter.ai/api/v1"
    vllm_context_size: int = 32768
    score_model_name: str = "virtuoussy/Qwen2.5-7B-Instruct-RLVR"
    score_vllm_address: str = "http://localhost:11436/v1"
    score_vllm_context_size: int = 4096
    disable_reasoning: bool = True
    temperature: float = 1.0  
    max_completion_tokens: int = 4096
    context_size_safety_margin: int = 75

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
        logger.info("*"*100)
        logger.info("Debug run")
        logger.info("*"*100)
        
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
    config.output_path = str(output_path)

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
    def load_data_snapshot(checkpoint_path, is_debug=False): #! todo needs features
        if is_debug:
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
        # logger.warning(f"No answer found in {output}")
        return 0

class RewardModel:
    def __init__(self, config: MathConfig):
        self.score_client = get_clinet(config.score_vllm_address, config.score_model_name)
        if config.debug_run:
            mock_reward_client(self.score_client)

    async def get_reward_for_answer(self, model_output, problem_instance: DataStore.Problem, config: MathConfig):
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
            
            input_tokens = self.score_client.encoder.encode(messages[0]['content'])
            diff = config.score_vllm_context_size - len(input_tokens)
            if diff < config.context_size_safety_margin:
                truncated_model_output = self.score_client.encoder.decode(self.score_client.encoder.encode(model_output)[-diff + config.context_size_safety_margin:])
                messages = [{
                    "role": "user",
                    "content": config.reward_model_instruction.format(
                        question=problem_instance.problem,
                        response=truncated_model_output,
                        reference=problem_instance.answer,
                    ),
                }]


            try:
                reward_output = await self.score_client.chat.completions.create(
                    model=config.score_model_name,
                    messages=messages,
                    temperature=config.temperature,
                    logprobs=True,
                    max_completion_tokens=10,
                )
            except openai.BadRequestError as e:
                import pprint
                logger.warning(f"length of input tokens: {len(input_tokens)}")
                logger.warning("messages:")
                logger.warning(str(messages))
                logger.warning("model_output:")
                logger.warning(str(model_output))
                raise e

            reward_answer = reward_output.choices[0].message.content
            if reward_answer == "YES":
                return np.exp(reward_output.choices[0].logprobs.content[0].logprob)
            elif reward_answer == "NO":
                return 1 - np.exp(reward_output.choices[0].logprobs.content[0].logprob)
            else:
                logger.warning(
                    f"Invalid reward answer: {reward_answer} \n for problem {problem_instance['problem']} \n model output: {model_output}")
                return 0

def merge_same_role_messages(messages):
    merged_messages = []
    for message in messages:
        if merged_messages and merged_messages[-1]["role"] == message["role"]:
            merged_messages[-1]["content"] += "\n" + message["content"]
        else:
            merged_messages.append(message)
    return merged_messages

def get_clinet(base_url, model_name):
    api_key = os.getenv("OPENROUTER_API_KEY") if 'openrouter' in base_url else None
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    client.encoder = AutoTokenizer.from_pretrained(model_name)
    return client
    

async def generate_model_output(client: AsyncOpenAI, model_name: str, messages: list[dict], config: MathConfig, **kwargs):
    kwargs["extra_body"] = kwargs.get("extra_body", {})
    if config.disable_reasoning:
        kwargs["extra_body"]["chat_template_kwargs"] = {
            "enable_reasoning": False,
        }
    if 'openrouter' in str(client.base_url):
        kwargs["extra_body"]["provider"] = {
            "only": ["chutes"]
        }

    input_text = [m['role'] + ": " + m['content'] for m in messages]
    input_text = "\n".join(input_text)
    num_input_tokens = len(client.encoder.encode(input_text))
    adjusted_max_completion_tokens = min(
        config.max_completion_tokens,
        config.vllm_context_size - num_input_tokens - config.context_size_safety_margin,
    )
    assert adjusted_max_completion_tokens > 0, "adjusted_max_completion_tokens is not positive"

    while True:
        try:
            output = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=config.temperature,
                max_completion_tokens=adjusted_max_completion_tokens,
                **kwargs,
            )
            output.choices[0].message.content = f"<think> {output.choices[0].message.reasoning} </think> {output.choices[0].message.content}"
            break
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit error: {e}")
        except openai.APIConnectionError as e:
            logger.warning(f"API connection error: {e}")
        except Exception as e:
            logger.warning(f"Error: {e}")
    return output

class LengthTracker:
    def __init__(self, length_limit: int, encoder: AutoTokenizer, config: MathConfig):
        self.length_limit = length_limit - config.context_size_safety_margin
        self.current_length = 0
        self.encoder = encoder
    
    def can_i_add_this_message(self, message: dict):
        new_text = message['role'] + ": " + message['content']
        new_tokens = len(self.encoder.encode(new_text))
        if self.current_length + new_tokens > self.length_limit:
            return False
        self.current_length += new_tokens
        return True

    

async def run_evaluation(config: MathConfig, data: DataStore = None):
    """
    This function is the main entry point for the evaluation.
    The previous code is for the sciworld environment. this code is for the math problems. a lot of the code is different. none of the code in the previous run_evaluation function is used.
    """
    if data is None:
        data = DataStore()
        data.init_problems(config)
    
    client = get_clinet(config.vllm_address, config.model_name)
    if config.debug_run:
        mock_client(client)
    reward_model = RewardModel(config)

    async def initial_interaction(problem_idx):
        problem_instance = data.problem_histories[problem_idx].problem
        for _ in range(config.num_initial_attempts):
            messages = [
                {"role": "user", "content": f"{problem_instance.problem}"},
            ]
            output = await generate_model_output(client, config.model_name, messages, config)
            model_output = output.choices[0].message.content
            
            reward = await reward_model.get_reward_for_answer(model_output, problem_instance, config)
            
            attempt = DataStore.Attempt(
                raw_prompt=messages,
                model_output=model_output,
                reward=reward,
                round_idx=-1,
            )
            data.problem_histories[problem_idx].attempts.append(attempt)
            
            if problem_idx == 0:
                logger.info(f"\n\nInitial attempt: {model_output}\n\n")
    
    async with anyio.create_task_group() as tg:
        for i in range(len(data.problem_histories)):
            tg.start_soon(initial_interaction, i)
    
    data.save_data_snapshot(config, f"data_initial_attempts.pkl")
    
    rewards = []
    for i in range(len(data.problem_histories)):
        rewards.extend([attempt.reward for attempt in data.problem_histories[i].attempts if attempt.round_idx == -1])
    logger.info(f"Initial rewards - 25th: {np.percentile(rewards, 25):.3f}, 50th: {np.percentile(rewards, 50):.3f}, 75th: {np.percentile(rewards, 75):.3f}")
    
    start_round = 0
    for i in range(len(data.problem_histories)):
        if len(data.problem_histories[i].attempts) > 0:
            start_round = max(start_round, data.problem_histories[i].attempts[-1].round_idx + 1)
    
    for round_idx in range(start_round, config.rounds):
        async def ICRL_interaction(problem_idx):
            messages = []
            messages.append({"role": "user", "content": f"{data.problem_histories[problem_idx].problem.problem}\n\n"})
            sorted_attempts = sorted(data.problem_histories[problem_idx].attempts, key=lambda x: x.reward, reverse=True)
            length_tracker = LengthTracker(config.vllm_context_size, client.encoder, config)
            for attempt in sorted_attempts:
                message = {"role": "user", "content": f"<Attempt>\n{attempt.model_output}\n**Reward:** {attempt.reward}\n</Attempt>"}
                if not length_tracker.can_i_add_this_message(message):
                    break
                messages.append(message)
            instruction = config.exploration_instruction if round_idx % 2 == 0 else config.exploitation_instruction
            messages.append({"role": "user", "content": f"\n\n{instruction}"})
            messages = merge_same_role_messages(messages)
            
            output = await generate_model_output(client, config.model_name, messages, config)
            model_output = output.choices[0].message.content
            
            reward = await reward_model.get_reward_for_answer(model_output, data.problem_histories[problem_idx].problem, config)
            
            data.problem_histories[problem_idx].attempts.append(DataStore.Attempt(
                raw_prompt=messages,
                model_output=model_output,
                reward=reward,
                round_idx=round_idx,
            ))

            if problem_idx == 0:
                logger.info(f"\n\nRound {round_idx} attempt: {model_output}\n\n")

        async def reflexion_interaction(problem_idx):
            messages = []
            messages.append({"role": "user", "content": f"{data.problem_histories[problem_idx].problem.problem}\n\n"})
            for attempt in data.problem_histories[problem_idx].attempts:
                messages.append({"role": "user", "content": f"<Reflection>\n{attempt.extra_fields['reflection']}\n**Reward:** {attempt.reward}\n</Reflection>"})
            instruction = config.use_reflexion_instruction if not config.selfrefine else config.use_selfrefine_instruction
            messages.append({"role": "user", "content": f"\n\n{instruction}"})
            messages = merge_same_role_messages(messages)
            
            output = await generate_model_output(client, config.model_name, messages, config)
            model_output = output.choices[0].message.content
            
            reward = await reward_model.get_reward_for_answer(model_output, data.problem_histories[problem_idx].problem, config)
            
            current_attempt = DataStore.Attempt(
                raw_prompt=copy.deepcopy(messages),
                model_output=model_output,
                reward=reward,
                round_idx=round_idx,
            )
            data.problem_histories[problem_idx].attempts.append(current_attempt)

            if problem_idx == 0:
                logger.info(f"\n\nRound {round_idx} reflection: {model_output}\n\n")

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
            f"data_round_{round_idx}_final.pkl",
            delete=(
                f"data_round_{round_idx-1}_final.pkl"
                if round_idx > 0
                else "data_initial_attempts.pkl"
            ),
        )
        
        # Log the 25 50 75 percentile of the rewards
        rewards = []
        for i in range(len(data.problem_histories)):
            rewards.extend([attempt.reward for attempt in data.problem_histories[i].attempts if attempt.round_idx == round_idx])
        logger.info(f"Round {round_idx} rewards - 25th: {np.percentile(rewards, 25):.3f}, 50th: {np.percentile(rewards, 50):.3f}, 75th: {np.percentile(rewards, 75):.3f}")
        

def find_math_file(folder_path):
    """Find the math data file in a given folder."""
    # Look for the most recent round file first
    pattern = os.path.join(folder_path, "data_round_*_final.pkl")
    files = glob.glob(pattern)
    
    if files:
        # Sort by round number to get the latest
        def extract_round_num(filepath):
            match = re.search(r"data_round_(\d+)_final\.pkl", os.path.basename(filepath))
            return int(match.group(1)) if match else -1
        
        files.sort(key=extract_round_num, reverse=True)
        return files[0]
    
    # If no round files found, look for initial attempts file
    initial_file = os.path.join(folder_path, "data_initial_attempts.pkl")
    if os.path.exists(initial_file):
        return initial_file
    
    raise FileNotFoundError(f"No math data file found in {folder_path}")

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels"""
    
    def format(self, record):
        # Apply color based on log level
        if record.levelno == logging.INFO:
            record.msg = f"{colorama.Fore.GREEN}{record.msg}{colorama.Fore.RESET}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{colorama.Fore.YELLOW}{record.msg}{colorama.Fore.RESET}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{colorama.Fore.RED}{record.msg}{colorama.Fore.RESET}"
        elif record.levelno == logging.CRITICAL:
            record.msg = f"{colorama.Fore.RED}{colorama.Style.BRIGHT}{record.msg}{colorama.Style.RESET_ALL}"
        
        return super().format(record)

async def main():
    # Initialize colorama
    colorama.init()
    
    # Parse command line arguments and get a config object
    config = parse_args()
    
    # Configure logging with custom formatter
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(message)s'))
    
    # Add file logging
    log_file = Path(config.output_path) / "output.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler, file_handler]
    )
    logger.setLevel(logging.INFO)
    
    # Suppress INFO logs from openai module
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    data = None
    if config.checkpoint_path:
        data = DataStore.load_data_snapshot(config.checkpoint_path, config.debug_run)
    await run_evaluation(config, data)

if __name__ == "__main__":
    anyio.run(main)