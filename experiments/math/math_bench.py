import pickle
import re
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
import uuid

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
    # zero_out_rewards: bool = False # Ablation
    random_rewards: float = 0.0 # Ablation - 0.0 = never, 1.0 = always
    # no_rewards: bool = False # Ablation
    # explore_only: bool = False # Ablation
    # exploit_only: bool = False # Ablation
    explore_and_exploit: bool = False # Ablation
    neutral_prompt: bool = False # Ablation
    alternate_prompt: Optional[int] = None # Ablation
    # max_reflections_in_context: Optional[int] = None 
    react: bool = False
    selfrefine: bool = False
    # cot: bool = False
    # high_reward_only: bool = False

    # Experiment parameters
    dataset_name: str = "MathArena/aime_2025"
    split_name: str = "train"
    num_initial_attempts: int = 2
    num_problems: int = -1 # -1 means all problems
    rounds: int = 40
    
    # Model configuration
    model_name: str = "Qwen/Qwen3-32B"
    vllm_address: str = "http://localhost:11435/v1"
    # vllm_address: str = "https://openrouter.ai/api/v1"
    vllm_context_size: int = 32768
    score_model_name: str = "virtuoussy/Qwen2.5-7B-Instruct-RLVR"
    score_vllm_address: str = "http://localhost:11436/v1"
    score_vllm_context_size: int = 2048
    disable_reasoning: bool = False
    temperature: float = 1.0  
    max_completion_tokens: int = 4096 * 4
    context_size_safety_margin: int = 75
    max_attempt_length: int = 512
    reflection_max_completion_tokens: int = 512
    label_name: str = "Achieved score"
    reward_first: bool = False   

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

    exploration_alternates: list[str] = (
        "Review the prior attempts. Produce a fresh solution that is clearly different from every previous one.\nWrite your final answer in the format of <answer>...</answer>.",
        "Consider all earlier tries and generate a novel answer that does not repeat or closely paraphrase them.\nWrite your final answer in the format of <answer>...</answer>.",
        "Inspect the earlier answers and craft a distinct solution that meaningfully diverges from each of them.\nWrite your final answer in the format of <answer>...</answer>.",
        "After reading the previous attempts, propose an alternative approach that explores a new line of reasoning and avoids overlap with any prior answer.\nWrite your final answer in the format of <answer>...</answer>.",
        "After reading the previous attempts, propose an alternative approach that explores a new line of reasoning and avoids overlap with any prior answer.\nWrite your final answer in the format of <answer>...</answer>.", # the no reference to reward prompt
    )

    exploitation_alternates: list[str] = (
        "Review the previous attempts and their reward scores. Produce an answer expected to outperform all prior scores.\nWrite your final answer in the format of <answer>...</answer>.",
        "Use the earlier attempts and rewards as guidance to optimize your response; aim to exceed the current best reward.\nWrite your final answer in the format of <answer>...</answer>.",
        "Analyze the rewards/feedback from previous answers, then craft a response that maximizes the objective and surpasses the best so far.\nWrite your final answer in the format of <answer>...</answer>.",
        "Given the earlier attempts and their rewards, write an improved answer designed to beat the top score.\nWrite your final answer in the format of <answer>...</answer>.",
        "Given the earlier attempts and the number after each attempt, write an improved answer designed to beat that number. \nWrite your final answer in the format of <answer>...</answer>." # the no reference to reward prompt
    )

    explore_and_exploit_instruction: str = """
You get multiple attempts to complete the task. You can see the previous attempts and their rewards.
For this attempt, decide whether to try a completely different approach or to improve on a previous attempt. Then, continue to answer the question.
Write your final answer in the format of <answer>...</answer>.
"""

    neutral_round_instruction: str = """
Try to correctly answer the math problem presented to you.
Write your final answer in the format of <answer>...</answer>.
"""

    cot_instruction: str = """
Before giving your final answer, carefully reason step-by-step:
1. Understand and restate clearly what the math problem is asking.
2. Identify the mathematical concepts or formulas needed to solve the problem.
3. Plan the precise steps required to arrive at the correct solution.
4. Verify your solution for accuracy and logical consistency.

Then, clearly present your final solution in the format of <answer>...</answer>.
"""

    do_reflexion_instruction: str = """
Explain what went wrong in your previous solution attempt and identify the specific mistake. Then, describe briefly how you will change your approach to avoid that error.
"""

    use_reflexion_instruction: str = """
Consider your reflections on past mistakes for this math problem. Apply those insights now to solve the problem correctly.
Write your corrected final answer in the format of <answer>...</answer>.
"""

    do_selfrefine_instruction: str = """
Review your completed solution to this math problem. Provide detailed feedback clearly enclosed within <feedback>...</feedback> tags:
1. Identify precisely which steps or calculations were incorrect.
2. Analyze why each incorrect step or calculation failed.
3. Specify the exact mathematical concepts, rules, or formulas that should have been applied differently.

Then, briefly outline a refined and correct approach that you would follow next time to avoid these mistakes.
"""

    use_selfrefine_instruction: str = """
Consider the detailed feedback from your previous solution attempts on this math problem. Use this feedback explicitly to guide your current attempt:
1. Correctly apply the identified mathematical concepts and formulas.
2. Carefully avoid repeating previously made calculation errors.
3. Clearly follow the refined approach outlined earlier.

Write your improved final answer in the format of <answer>...</answer>.
"""

    react_instruction: str = """
Think through each step of the math problem-solving process carefully before finalizing your answer. Clearly outline your reasoning internally within `<thought>...</thought>` tags.

After careful reasoning, present your final answer in the format of <answer>...</answer>.
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
        config.num_problems = 1
        config.rounds = 10
        config.num_initial_attempts = 1
        config.max_completion_tokens = 200
        logger.info("*"*100)
        logger.info("Debug run")
        logger.info("*"*100)
    
    if config.alternate_prompt is not None:
        config.exploration_instruction = config.exploration_alternates[config.alternate_prompt]
        config.exploitation_instruction = config.exploitation_alternates[config.alternate_prompt]
        
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

    postfix = datetime.now().strftime("%Y%m%d_%H%M") + "_" + str(uuid.uuid4())[:3]
    if config.postfix:
        postfix = postfix + "_" + config.postfix
    output_path = Path(base_path) / config.output_path / config.icrl_mode.value / postfix
    config.output_path = str(output_path)

    config.is_openrouter = '/' in config.model_name

    # sanity checks
    # assert sum([config.explore_only, config.explore_and_exploit]) <= 1, "Only one of explore_only or explore_and_exploit can be true"
    # assert sum([config.no_rewards, config.zero_out_rewards]) <= 1, "Only one of positive_only, no_rewards, or zero_out_rewards can be true"

    # save config
    if not config.debug_run:
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
    def load_data_snapshot(checkpoint_path):
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
        return 0

class RewardModel:
    def __init__(self, config: MathConfig):
        self.score_client = get_clinet(config.score_vllm_address, config.score_model_name)
        # if config.debug_run:
        #     mock_reward_client(self.score_client)

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

def format_attempt_content(attempt_content: str, reward: float, config: MathConfig, tokenizer: AutoTokenizer, tag_name: str = "Attempt") -> str:
    """Format attempt content with improved formatting."""
    # remove the content between <think> and </think>
    attempt_content = re.sub(r'<think>.*?</think>', '', attempt_content, flags=re.DOTALL)
    # Truncate to last N tokens instead of characters
    tokens = tokenizer.encode(attempt_content)
    if len(tokens) > config.max_attempt_length:
        # truncated_tokens = tokens[:config.max_attempt_length // 2] + tokenizer.encode("...") + tokens[-config.max_attempt_length // 2:]
        truncated_tokens = tokens[:config.max_attempt_length] + tokenizer.encode("...")
        attempt_content = tokenizer.decode(truncated_tokens)
    
    # Replace multiple newlines with single newline
    attempt_content = re.sub(r'\n+', '\n', attempt_content)

    # Format with reward in front and human readable reward format
    if reward is None:
        formatted_content = f"<{tag_name}>\n{attempt_content}\n</{tag_name}>"
    elif config.reward_first:
        formatted_content = f"<{tag_name}>\n**{config.label_name}: {int(reward * 100)}/100**\n{attempt_content}\n</{tag_name}>"
    else:
        formatted_content = f"<{tag_name}>\n{attempt_content}\n</{tag_name}>**{config.label_name}: {int(reward * 100)}/100**"
    
    return formatted_content

def get_clinet(base_url, model_name):
    api_key = os.getenv("OPENROUTER_API_KEY") if 'openrouter' in base_url else None
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    client.encoder = AutoTokenizer.from_pretrained(model_name)
    return client
    

async def generate_model_output(client: AsyncOpenAI, model_name: str, messages: list[dict], config: MathConfig, **kwargs):
    kwargs["extra_body"] = kwargs.get("extra_body", {})
    if kwargs.pop("disable_reasoning", config.disable_reasoning):
        messages.insert(0, {"role": "system", "content": "/no_think"})
    # if 'openrouter' in str(client.base_url):
    #     kwargs["extra_body"]["provider"] = {
    #         "only": ["chutes"]
    #     }

    input_text = [m['role'] + ": " + m['content'] for m in messages]
    input_text = "\n".join(input_text)
    num_input_tokens = len(client.encoder.encode(input_text))
    max_completion_tokens = kwargs.pop("max_completion_tokens", config.max_completion_tokens)
    adjusted_max_completion_tokens = min(
        max_completion_tokens,
        config.vllm_context_size - num_input_tokens - config.context_size_safety_margin,
    )
    assert adjusted_max_completion_tokens > 0, f"adjusted_max_completion_tokens is not positive: {adjusted_max_completion_tokens}"
    if adjusted_max_completion_tokens < max_completion_tokens:
        logger.warning(f"had to truncate the max_completion_tokens from {max_completion_tokens} to {adjusted_max_completion_tokens}")

    while True:
        try:
            output = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=config.temperature,
                max_completion_tokens=adjusted_max_completion_tokens,
                **kwargs,
            )
            if not config.disable_reasoning:
                if hasattr(output.choices[0].message, 'reasoning') and output.choices[0].message.reasoning is not None and len(output.choices[0].message.reasoning) > 20:
                    output.choices[0].message.content = f"<think>\n{output.choices[0].message.reasoning}\n</think>\n\n{output.choices[0].message.content}"
                else:
                    # remove any <think> or </think> tags
                    # then add empty think tags in the beginning
                    cleaned_content = re.sub(r'<think>|</think>', '', output.choices[0].message.content)
                    output.choices[0].message.content = f"<think></think>\n\n{cleaned_content}"
            else:
                output.choices[0].message.content = re.sub(r'<think>.*?</think>\s*', '', output.choices[0].message.content, flags=re.DOTALL)
            assert len(output.choices[0].message.content) > 0, "Output is empty"
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
    # if config.debug_run:
    #     mock_client(client)
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

            real_reward = reward
            if config.random_rewards > 0 and np.random.random() < config.random_rewards:
                reward = 1 - reward
            
            attempt = DataStore.Attempt(
                raw_prompt=messages,
                model_output=model_output,
                reward=reward,
                round_idx=-1,
                extra_fields={"real_reward": real_reward},
            )
            data.problem_histories[problem_idx].attempts.append(attempt)
            
            if problem_idx == 0:
                logger.info(f"\n{'-'*100}\nInitial attempt: {model_output}\n{'-'*100}\n")
    
    async with anyio.create_task_group() as tg:
        for i in range(len(data.problem_histories)):
            tg.start_soon(initial_interaction, i)
    
    data.save_data_snapshot(config, f"data_initial_attempts.pkl")
    
    rewards = []
    for i in range(len(data.problem_histories)):
        rewards.extend([attempt.reward for attempt in data.problem_histories[i].attempts if attempt.round_idx == -1])
    if config.num_initial_attempts > 0:
        logger.info(f"Initial rewards - 25th: {np.percentile(rewards, 25):.3f}, 50th: {np.percentile(rewards, 50):.3f}, 75th: {np.percentile(rewards, 75):.3f}")
    
    start_round = 0
    for i in range(len(data.problem_histories)):
        if len(data.problem_histories[i].attempts) > 0:
            start_round = max(start_round, data.problem_histories[i].attempts[-1].round_idx + 1)
    
    for round_idx in range(start_round, config.rounds):
        async def ICRL_interaction(problem_idx):
            messages = []
            length_tracker = LengthTracker(config.vllm_context_size - config.max_completion_tokens, client.encoder, config)

            # Add instruction based on round type and task description
            if config.icrl_mode == Methods.ICRL:
                if config.explore_and_exploit:
                    instruction = config.explore_and_exploit_instruction
                elif config.neutral_prompt:
                    instruction = config.neutral_round_instruction
                else:
                    instruction = config.exploration_instruction if round_idx % 2 == 0 else config.exploitation_instruction
            elif config.icrl_mode == Methods.RANDOM_SAMPLING:
                instruction = config.neutral_round_instruction
            elif config.react:
                instruction = config.react_instruction
            else:
                raise ValueError(f"Invalid Method: {config.icrl_mode}")
            instruction_message = {"role": "user", "content": f"\n\n{instruction}"}
            assert length_tracker.can_i_add_this_message(instruction_message), f"Instruction message is too long!! {instruction_message}"
            message = {"role": "user", "content": f"{data.problem_histories[problem_idx].problem.problem}\n\n"}
            assert length_tracker.can_i_add_this_message(message), f"Initial message is too long!! {message}"
            messages.append(message)
            sorted_attempts = sorted(data.problem_histories[problem_idx].attempts, key=lambda x: x.reward, reverse=True)
            if config.max_attempts_in_context is not None:
                sorted_attempts = sorted_attempts[:config.max_attempts_in_context]
            for i, attempt in enumerate(sorted_attempts):
                formatted_attempt = format_attempt_content(attempt.model_output, attempt.reward, config, client.encoder, "Attempt")
                # Add double newline between attempts (except for the first one)
                if i > 0:
                    formatted_attempt = "\n\n" + formatted_attempt
                message = {"role": "user", "content": formatted_attempt}
                if not length_tracker.can_i_add_this_message(message):
                    break
                messages.append(message)
            messages.append(instruction_message)
            messages = merge_same_role_messages(messages)
            
            output = await generate_model_output(client, config.model_name, messages, config)
            model_output = output.choices[0].message.content
            
            reward = await reward_model.get_reward_for_answer(model_output, data.problem_histories[problem_idx].problem, config)

            real_reward = reward
            if config.random_rewards > 0 and np.random.random() < config.random_rewards:
                reward = 1 - reward
            
            data.problem_histories[problem_idx].attempts.append(DataStore.Attempt(
                raw_prompt=messages,
                model_output=model_output,
                reward=reward,
                round_idx=round_idx,
                extra_fields={"real_reward": real_reward},
            ))

            if problem_idx == 0:
                logger.info(f"\n{'-'*100}\nRound {round_idx} attempt: {model_output}\n{'-'*100}\n")

        async def reflexion_interaction(problem_idx):
            messages = []
            length_tracker = LengthTracker(config.vllm_context_size - config.max_completion_tokens - config.reflection_max_completion_tokens, client.encoder, config)
            
            instruction = config.use_reflexion_instruction if not config.selfrefine else config.use_selfrefine_instruction
            instruction_message = {"role": "user", "content": f"\n\n{instruction}"}
            assert length_tracker.can_i_add_this_message(instruction_message), f"Instruction message is too long!! {instruction_message}"
            
            # Also check the reflection instruction that will be added later
            reflection_instruction = config.do_reflexion_instruction if not config.selfrefine else config.do_selfrefine_instruction
            reflection_instruction_message = {"role": "user", "content": f"{reflection_instruction}"}
            assert length_tracker.can_i_add_this_message(reflection_instruction_message), f"Reflection instruction message is too long!! {reflection_instruction_message}"
            
            message = {"role": "user", "content": f"{data.problem_histories[problem_idx].problem.problem}\n\n"}
            assert length_tracker.can_i_add_this_message(message), f"Initial message is too long!! {message}"
            messages.append(message)
            for i, attempt in enumerate(data.problem_histories[problem_idx].attempts):
                formatted_reflection = format_attempt_content(attempt.extra_fields['reflection'], None, config, client.encoder, "Reflection")
                # Add double newline between reflections (except for the first one)
                if i > 0:
                    formatted_reflection = "\n\n" + formatted_reflection
                message = {"role": "user", "content": formatted_reflection}
                if not length_tracker.can_i_add_this_message(message):
                    break
                messages.append(message)
            messages.append(instruction_message)
            messages = merge_same_role_messages(messages)
            
            output = await generate_model_output(client, config.model_name, messages, config)
            model_output = output.choices[0].message.content
            
            reward = await reward_model.get_reward_for_answer(model_output, data.problem_histories[problem_idx].problem, config)
            
            real_reward = reward
            if config.random_rewards > 0 and np.random.random() < config.random_rewards:
                reward = 1 - reward
            
            current_attempt = DataStore.Attempt(
                raw_prompt=copy.deepcopy(messages),
                model_output=model_output,
                reward=reward,
                round_idx=round_idx,
                extra_fields={"real_reward": real_reward},
            )
            data.problem_histories[problem_idx].attempts.append(current_attempt)

            if problem_idx == 0:
                logger.info(f"\n{'-'*100}\nRound {round_idx} reflection: {model_output}\n{'-'*100}\n")

            # reflection
            messages.append({"role": "assistant", "content": f"{model_output}\n**Reward:** {reward}\n"})
            model_solution_output = model_output
            messages.append({"role": "user", "content": f"{reflection_instruction}"})
            current_attempt.extra_fields['reflection_raw_prompt'] = copy.deepcopy(messages)
            
            output = await generate_model_output(client, config.model_name, messages, config, max_completion_tokens=config.reflection_max_completion_tokens, disable_reasoning=True)
            model_output = output.choices[0].message.content
            
            if config.selfrefine:
                reflection = "<Attempt>\n"
                reflection += model_solution_output
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
    
    handlers = []
    # Configure logging with custom formatter
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(message)s'))
    handlers.append(handler)
    
    if not config.debug_run:
        # Add file logging
        log_file = Path(config.output_path) / "output.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers
    )
    logger.setLevel(logging.INFO)
    
    # Suppress INFO logs from openai module
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    data = None
    if config.checkpoint_path:
        data = DataStore.load_data_snapshot(config.checkpoint_path)
    await run_evaluation(config, data)

if __name__ == "__main__":
    anyio.run(main)