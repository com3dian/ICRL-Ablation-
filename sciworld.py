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
from scienceworld import ScienceWorldEnv as ScienceWorldEnvBase
from sciworld_armap.utils.replace_sciworld_score import sciworld_monkey_patch
sciworld_monkey_patch()
from sciworld_armap.envs.sciworld_env import SciWorldEnv
from sciworld_armap.tasks.sciworld import SciWorldTask
from omegaconf import OmegaConf
from enum import Enum
import colorama
import copy
import dotenv
import pdb
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

@dataclass
class SciWorldConfig:
    sw_output_path: str = "ICL/sw/"  # ScienceWorld output path
    postfix: str = ""
    
    # Experiment modes
    icrl_mode: Methods = Methods.ICRL
    debug_run: bool = False
    concise_attempts: bool = True
    positive_only: bool = False
    max_attempts_in_context: Optional[int] = None
    zero_out_rewards: bool = False
    no_rewards: bool = False
    explore_only: bool = False
    exploit_only: bool = False
    explore_and_exploit: bool = False
    neutral_prompt: bool = False
    # Shorthands
    # no_rewards: bool = False # shorthand
    # is_openrouter: bool = False # shorthand

    # Experiment parameters
    num_initial_attempts: int = 2
    max_env_steps: int = 15
    num_envs: int = 4
    rounds: int = 40
    
    # Model configuration
    model_name: str = "gpt-4.1-mini"
    # model_name: str = "gpt-4.1"
    # model_name: str = "gpt-4.1-nano-2025-04-14"
    # model_name: str = "google/gemini-2.0-flash-001"
    use_openai_embedding: bool = True
    exploration_temperature: float = 1.0  
    exploitation_temperature: float = 1.0

    checkpoint_path: Optional[str] = None
    
    # OpenAI API key
    api_key: str = "sk-C8z62BDhmo4EW1bqOn2TTmdFR29ocUeZXLExkdmGS1T3BlbkFJQcA3zOug-aNTm98KC0Wjsv549b3OgxEGn9TKJknXMA"

    # Prompt templates
    task_prompt_cot: str = """
You are a helpful assistant to do some scientific experiment in an environment.

<Environment description>
In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway  
You should explore the environment and find the items you need to complete the experiment.  
{available_actions}

FOCUS is a extremely critical action that can be only used the number of times 'focus' is mentioned in the task description. Using it more than that or inappropiately (such as on a wrong object) will terminate the session and the task WILL FAIL.

</Environment description>
"""
    
#     exploration_instruction: str = """
# Your location and the environment is reset now. It's your turn.
# Look at the previous attempts and try to construct a plan that is different from every single one of the previous attempts, while making sure it is feasible as well. 
# After thinking, make sure to write your action in the "Action: single_action" format. It is parsed by a script.
# """

    exploration_instruction: str = """
Your location and the environment is reset now. It's your turn.
Look at the previous attempts and try to construct a plan for doing the task that is different from every single one of the previous attempts. Try new approaches to the task, don't follow the previous attempts.
After thinking, make sure to write your action **exactly** in the "Action: single_action" format. **You can only do one action at a time.**
"""

#     exploitation_instruction: str = """
# Your location and the environment is reset now. It's your turn.
# Based on the previous high reward attempts, try to construct a higher scoring plan while making sure it is feasible as well. 
# After thinking, make sure to write your action in the "Action: single_action" format. It is parsed by a script.
# """
#     exploitation_instruction: str = """
# Your location and the environment is reset now. It's your turn.
# Look at the previous high reward attempts and inspired by what they're doing right, try to construct a plan that successfully completes the task. If any of them successfully completed the task, try to do the same, if not, try to stitch together their best parts somehow.
# After thinking, make sure to write your action in the "Action: single_action" format. It is parsed by a script.
# """
#     exploitation_instruction: str = """
# Your location and the environment is reset now. It's your turn.
# Look at the previous high reward attempts and inspired by what they're doing right, try to construct a plan that successfully completes the task. If any of them successfully completed the task, try to do the same, if not, try to stitch together their best parts somehow. Similarly, specifically avoid the actions with negative rewards.
# After thinking, make sure to write your action in the "Action: single_action" format. It is parsed by a script.
# """
#     exploitation_instruction: str = """
# Your location and the environment is reset now. It's your turn.
# Look at the previous high reward attempts/actions and based on what they're doing right, stitch together an improved action sequence. Obviously, if any of them successfully completed the task, simply copy it.
# After thinking, make sure to write your action in the "Action: single_action" format. It is parsed by a script.
# """
    exploitation_instruction: str = """
Your location and the environment is reset now. It's your turn.
Look at the previous attempts. The steps with positive rewards are the ones that have achieved a subgoal successfully. Try to feasibly chain together the positive reward steps to achieve all subgoals and complete the task.
Obviously, if any of the previous attempts successfully completed the task, you should simply imitate the steps one by one.
After thinking, make sure to write your action **exactly** in the "Action: single_action" format. **You can only do one action at a time.**
"""

    neutral_round_instruction: str = """
Your location and the environment is reset now. It's your turn.
Try to complete the task.
After thinking, make sure to write your action **exactly** in the "Action: single_action" format. **You can only do one action at a time.**
"""

    explore_and_exploit_instruction: str = """
Your location and the environment is reset now. It's your turn.
You get multiple attempts to complete the task.
At the beginning of each attempt, decide whether to explore or exploit.
To explore, look at the previous attempts and try to construct a plan that is different from every single one of the previous attempts, while making sure it is feasible as well.
To exploit, look at the previous high reward attempts and based on what they're doing right, stitch together an improved action sequence. Obviously, if any of them successfully completed the task, you should simply copy all the steps.
After thinking, make sure to write your action **exactly** in the "Action: single_action" format. **You can only do one action at a time.**
"""

#     do_reflexion_instruction: str = """
# Look at the previous attempt and reflect on:
# 1. What actions worked well and why
# 2. What actions didn't work and why
# 3. What you learned about the task requirements
# 4. What you would do differently
# This reflection should help the agent for its next attempt. Write in a single paragraph.
# """

    do_reflexion_instruction: str = """
"You will be given the history of a past experience in which you encountered a task that required you to provide a response to a prompt aiming to maximize a reward, and you attempted a response. You were unsuccessful in providing an answer that successfully completed the task. Instead of recounting the details of the task itself, focus on analyzing the approach you took and the specific actions or steps you attempted. Based on this reflection, devise a concise, revised plan of action that acknowledges your error and details the exact measures or methods you should have employed. For example, if you attempted steps A and B but overlooked step C, construct a plan that explicitly incorporates step C into your approach. This self-reflection and plan will be essential for when you reattempt the task.
"""

    use_reflexion_instruction: str = """
Your location and the environment is reset now. It's your turn.
Consider the previous reflections about doing the task and try to complete the task.
After thinking, make sure to write your action **exactly** in the "Action: single_action" format. **You can only do one action at a time.**
"""

    available_actions: str = """
The available actions are: (OBJ is placeholder for object name so for example "Action: look at picture" is valid)
open OBJ: open a container  
close OBJ: close a container  
activate OBJ: activate a device  
deactivate OBJ: deactivate a device  
connect OBJ to OBJ: connect electrical components  
disconnect OBJ: disconnect electrical components  
use OBJ [on OBJ]: use a device/item  
look around: describe the current room  
examine OBJ: describe an object in detail  
look at OBJ: describe a container's contents  
read OBJ: read a note or book  
move OBJ to OBJ: move an object to a container  
pick up OBJ: move an object to the inventory  
pour OBJ into OBJ: pour a liquid into a container  
mix OBJ: chemically mix a container (here, OBJ should be the container the items to be mixed are in)
teleport to LOC: teleport to a specific room in one step  
focus on OBJ: signal intent on a task object  
eat OBJ: eat a food
go to OBJ: move to a new location
dunk OBJ into OBJ: dunk a container into a liquid
inventory: list agent's inventory
wait: task no action for 10 steps  
wait1: task no action for a step
"""

    # no_reward_exploration_instruction: str = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response. Provide a response that is different from every single one of the previous attempts demonstrated in the context, while making sure it correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

    # evaluation_prompt_template: str = """Please evaluate whether the proposed solution to the 24‐game is correct for this input:
# Question: {sample_question}
# Solution: {full_answer}

# – If the solution is correct, return **Answer**: 10.  
# – If it is incorrect, look at the numbers in each "left: ..." step and judge how likely they are to reach 24:
#    • sure → 3  
#    • likely → 2  
#    • impossible → 1  
# - If the step is invalid, such as using numbers outside of the 4 input numbers or not using all numbers at the last step, the score is 0. 

# Add up those step‐scores and return **Answer**: <sum of likeliness scores>."""

def parse_args():
    """
    Parse command line arguments and create a configuration object using OmegaConf
    
    Returns:
        SciWorldConfig: The configuration object with all parameters
    """
    # Parse command line arguments
    default_config = OmegaConf.structured(SciWorldConfig)  # Start with code defaults
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
    config.task_prompt_cot = config.task_prompt_cot.format(available_actions=config.available_actions)

    if config.debug_run:
        config.num_envs = 1
        config.rounds = 10
        config.num_initial_attempts = 1
        config.model_name = "gpt-4.1-nano-2025-04-14"
        config.judge_model_name = "gpt-4.1-nano-2025-04-14"
        logger.debug("*"*100)
        logger.debug("Debug run")
        logger.debug("*"*100)
        
    if config.icrl_mode == Methods.RANDOM_SAMPLING:
        config.num_initial_attempts = 0
        config.max_attempts_in_context = 0
    elif config.icrl_mode == Methods.REFLEXION:
        config.num_initial_attempts = 0
    elif config.icrl_mode == Methods.REACT:
        config.num_initial_attempts = 0
        config.max_attempts_in_context = 0
        config.react = True

    postfix = datetime.now().strftime("%Y%m%d_%H%M")
    if config.postfix:
        postfix = postfix + "_" + config.postfix
    output_path = Path(base_path) / config.sw_output_path / config.icrl_mode.value / postfix
    config.sw_output_path = str(output_path)

    config.is_openrouter = '/' in config.model_name

    # sanity checks
    assert sum([config.explore_only, config.explore_and_exploit]) <= 1, "Only one of explore_only or explore_and_exploit can be true"
    assert sum([config.positive_only, config.no_rewards, config.zero_out_rewards]) <= 1, "Only one of positive_only, no_rewards, or zero_out_rewards can be true"

    # save config
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "config.yaml", "w") as f:
        OmegaConf.save(config, f)
    
    return config


def load_envs(num_envs, config, gold_path=False, micro_repeat=1):
    # Initialize base ScienceWorld environment
    base_env = ScienceWorldEnvBase()
    
    # Get available tasks
    task_names = base_env.get_task_names()
    task_names.sort()
    if gold_path:
        task_names = task_names[:1]
        num_envs = 1
    else:
        task_names = task_names[1:]
    
    # Create a list to store environment instances
    envs = []
    
    # Create environments
    for i in range(num_envs):
        # Randomly select task and variation
        # task_num = random.randint(0, len(task_names) - 1)
        task_num = i 
        task_name = task_names[task_num]
        # var_num = random.randint(0, 9)  # Most tasks have 10 variations
        var_num = 0
        
        for _ in range(micro_repeat):
            while True:
                try:
                    base_env = ScienceWorldEnvBase()
                    # Create SciWorldTask instance
                    task = SciWorldTask(
                        task_id=f"{task_name}-{var_num}",
                        sub_task_name=task_name,
                        variation_idx=var_num
                    )

                    # Initialize SciWorldEnv from eval_agent
                    sciworld_env = SciWorldEnv(
                        task=task,
                        env=base_env,
                        gold_path=gold_path,
                        max_env_steps=config.max_env_steps,
                        api_key=config.api_key if config.use_openai_embedding else None
                    )
                    
                    sciworld_env.reset()
                    break
                except Exception as e:
                    if not isinstance(e, KeyboardInterrupt):
                        logger.info(f"Error: {e}")
                        continue
                    else:
                        raise e
            
            # Add to list of environments
            envs.append(sciworld_env)
    
    # logger.debug(f"Created {len(envs)} ScienceWorld environments with random tasks")
    return envs

async def converse(client: AsyncOpenAI, f, messages, config, temperature: float = 1.0):
    """
    Converse with the model.
    
    Args:
        client: AsyncOpenAI client
        f: function that
            takes messages on each call (initial argument on first call)
            outputs messages and done flag on each call (return value and done flag on last call)
        messages: initial input to f
        config: configuration object
        temperature: temperature parameter for model generation (default: 1.0)
    """
    while True:
        # t0 = time.perf_counter()
        messages, done = f(messages)
        # t1 = time.perf_counter()
        # logger.info(f"Environment step took {t1 - t0} seconds")
        if not done:
            # t0 = time.perf_counter()
            response = await client.chat.completions.create(
                model=config.model_name, 
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=temperature,
            )
            # t1 = time.perf_counter()
            # logger.info(f"API call took {t1 - t0} seconds")
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
        else:
            break
    return messages

def merge_same_role_messages(messages):
    merged_messages = []
    for message in messages:
        if merged_messages and merged_messages[-1]["role"] == message["role"]:
            merged_messages[-1]["content"] += "\n" + message["content"]
        else:
            merged_messages.append(message)
    return merged_messages

@dataclass
class Attempt:
    raw_prompts: list[dict] = field(default_factory=list)  # the whole messages objects given to the api at each step
    rewards: list[float] = field(default_factory=list)     # rewards from the model
    attempt_prompts: list[dict] = field(default_factory=list)  # has rewards embedded
    extra_fields: dict = field(default_factory=dict)
    reflexion: bool = False

    def get_processed_attempt_prompts(self, config):
        if self.reflexion:
            return self.attempt_prompts

        def predicate(reward):
            if config.no_rewards:
                return False
            if config.positive_only:
                return reward > 0
            return True
        
        attempt_prompts_copy = copy.deepcopy(self.attempt_prompts)
        modified_rewards = copy.deepcopy(self.rewards)
        if config.positive_only:
            if modified_rewards[-1] < 0:
                modified_rewards[-1] = 0
        if config.zero_out_rewards:
            modified_rewards = [0] * len(modified_rewards)

        if not config.concise_attempts:
            if not config.no_rewards:
                attempt_prompts_copy[-1]['content'] += f" (Total reward: {sum(modified_rewards)})"
            return attempt_prompts_copy
        else: 
            """
            (Interaction summary)
            Task description: blah blah blah
            Actions and respective rewards: action1 (reward1) -> action2 (reward2) -> ... -> actionN (rewardN)
            sum of rewards: 10, final observation: blah blah blah
            """
            # content = "(Interaction summary)\n"
            # action_idx = 0
            # for attempt_prompt in attempt_prompts_copy:
            #     if attempt_prompt['role'] == "assistant":
            #         action = SciWorldEnv.parse_action(attempt_prompt['content'])
            #         action = re.sub(r'\s+', ' ', action)
            #         if len(action) > 100:
            #             action = action[:100] + "..."
            #         if action_idx == 0:
            #             if predicate(modified_rewards[0]):
            #                 content += f"{action} (reward={modified_rewards[0]})"
            #             else:
            #                 content += f"{action}"
            #         else:
            #             if predicate(modified_rewards[action_idx]):
            #                 content += f" -> {action} (reward={modified_rewards[action_idx]})"
            #             else:
            #                 content += f" -> {action}"
            #         action_idx += 1
            # outcome = attempt_prompts_copy[-1]['content'].split("\n")[-1]
            # content += f"\nTotal reward: {sum(modified_rewards)}, Outcome: {outcome}" \
            #     if not config.no_rewards else f"\nOutcome: {outcome}"
            # return [{"role": "user", "content": content}]
            
            
            content = "(Interaction summary)\n"
            action_idx = 0
            for i, attempt_prompt in enumerate(attempt_prompts_copy):
                if attempt_prompt['role'] == "assistant":
                    action = SciWorldEnv.parse_action(attempt_prompt['content'])
                    action = re.sub(r'\s+', ' ', action)
                    if len(action) > 100:
                        action = action[:100] + "..."
                    if action_idx == 0:
                        if predicate(modified_rewards[0]):
                            content += f"{action} -> {attempt_prompts_copy[i+1]['content']} (reward={modified_rewards[0]})\n"
                        else:
                            content += f"{action} -> {attempt_prompts_copy[i+1]['content']}\n"
                    else:
                        if predicate(modified_rewards[action_idx]):
                            content += f" -> {action} -> {attempt_prompts_copy[i+1]['content']} (reward={modified_rewards[action_idx]})\n"
                        else:
                            content += f" -> {action} -> {attempt_prompts_copy[i+1]['content']}\n"
                    action_idx += 1
            if not config.no_rewards:
                content += f"\nTotal reward: {sum(modified_rewards)}"
            return [{"role": "user", "content": content}]

def save_data_snapshot(data, config, filename, delete=None):
    """
    Save a snapshot of the data to a file
    
    Args:
        data: Data dictionary to save
        config: Configuration object
        filename: Name of the file to save to
        delete: Name of the file to delete if it exists
    """
    # Convert data to serializable format
    serializable_data = {}
    raw_prompts_data = {}
    
    for env_id, env_data in data.items():
        serializable_data[env_id] = {
            'bootstrap_attempts': {
                attempt_id: {
                    'rewards': attempt.rewards,
                    'attempt_prompts': attempt.attempt_prompts,
                    'extra_fields': attempt.extra_fields,
                    'reflexion': attempt.reflexion
                } for attempt_id, attempt in env_data.get('bootstrap_attempts', {}).items()
            },
            'round_attempts': {
                round_id: {
                    attempt_id: {
                        'rewards': attempt.rewards,
                        'attempt_prompts': attempt.attempt_prompts,
                        'extra_fields': attempt.extra_fields,
                        'reflexion': attempt.reflexion
                    } for attempt_id, attempt in round_data.items()
                } for round_id, round_data in env_data.get('round_attempts', {}).items()
            }
        }
        
        # Store raw_prompts separately
        raw_prompts_data[env_id] = {
            'bootstrap_attempts': {
                attempt_id: attempt.raw_prompts for attempt_id, attempt in env_data.get('bootstrap_attempts', {}).items()
            },
            'round_attempts': {
                round_id: {
                    attempt_id: attempt.raw_prompts for attempt_id, attempt in round_data.items()
                } for round_id, round_data in env_data.get('round_attempts', {}).items()
            }
        }
    
    output_path = Path(config.sw_output_path)
    
    # Save main data to file
    with open(output_path / filename, "w") as f:
        json.dump(serializable_data, f, indent=2)
    
    # Save raw_prompts to a separate file
    raw_prompts_filename = f"raw_prompts_{filename}"
    with open(output_path / raw_prompts_filename, "w") as f:
        json.dump(raw_prompts_data, f, indent=2)
        
    if delete:
        try:
            os.remove(output_path / delete)
        except FileNotFoundError:
            pass
        try:
            os.remove(output_path / f"raw_prompts_{delete}")
        except FileNotFoundError:
            pass

async def run_evaluation(config: SciWorldConfig, data: dict = None):
    """
    Evaluate the model using batch inference.
    
    Args:
        config: Configuration object
    """
    # Initialize global data structure
    if data is None:
        data = {}
        for env_id in range(config.num_envs):
            data[env_id] = {
                'bootstrap_attempts': {},
                'round_attempts': {}
            }
    
    # Initialize AsyncOpenAI client
    if config.is_openrouter:
        # use openrouter
        client = AsyncOpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    else:
        client = AsyncOpenAI(api_key=config.api_key)
    
    from sciworld_armap.envs.base import raw_icl
    golden_attempts = raw_icl
    for attempt in golden_attempts[0]:
        logger.info(f"{attempt['role']} {'*'*100}")
        logger.info(attempt['content'])

    # bootstrap shot
    envs = load_envs(config.num_envs, config, gold_path=False, micro_repeat=config.num_initial_attempts)
    
    def wrapper(i):
        env = envs[i]
        env_id = i // config.num_initial_attempts
        attempt_id = i % config.num_initial_attempts
        
        # Initialize attempt object if it doesn't exist
        # if attempt_id not in data[env_id]['bootstrap_attempts']:
        data[env_id]['bootstrap_attempts'][attempt_id] = Attempt()
        
        # Get current attempt object
        current_attempt = data[env_id]['bootstrap_attempts'][attempt_id]
        
        first_round = True
        context_prompt = None
        
        def initial_interaction(messages):
            nonlocal first_round
            nonlocal context_prompt
            nonlocal current_attempt
            
            if first_round:
                messages = messages[0]
                messages[0]["content"] = f"""{config.task_prompt_cot}\n<Attempts>\nHere's an example (without the Thought part):\n{messages[0]["content"]}"""
                messages[-1]["content"] = f"""{messages[-1]["content"]}\n</Attempts>\n<Instructions>\n{config.neutral_round_instruction}\n</Instructions>\n{env.env.taskdescription()}"""                
                current_attempt.raw_prompts.append(copy.deepcopy(messages))
                current_attempt.attempt_prompts.append({"role": "user", "content": env.env.taskdescription()})
                first_round = False
                logger.info(f"{colorama.Fore.RED + messages[-1]['role']} {'*'*100}")
                # logger.debug(messages[-1]["content"])
                logger.info(colorama.Fore.RED + env.env.taskdescription())
                return messages, False
            else:
                assert messages[-1]["role"] == "assistant", "It's assistant's turn"
                current_attempt.attempt_prompts.append({"role": "assistant", "content": messages[-1]["content"]})
                logger.info(f"{colorama.Fore.GREEN + messages[-1]['role']} {'*'*100}")
                logger.info(colorama.Fore.GREEN + messages[-1]["content"])
                if context_prompt is not None:
                    messages[-2]["content"] = context_prompt

                prompt, state = env.step(messages[-1]["content"])
                context_prompt = prompt + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())

                # async debug
                if 'Task Failed. You have done something wrong' in prompt and not state.finished:
                    pdb.set_trace(header="task failed but not finished")
                
                # Record the reward
                current_attempt.rewards.append(state.reward)
                
                if not state.finished:
                    attempt_prompt = prompt + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                    current_attempt.attempt_prompts.append({"role": "user", "content": attempt_prompt})
                    augmented_prompt = prompt + "\n" + config.available_actions + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                    messages.append({"role": "user", "content": augmented_prompt})
                    logger.info(f"{colorama.Fore.RED + messages[-1]['role']} {'*'*100}")
                    # logger.debug(messages[-1]["content"])
                    logger.info(colorama.Fore.RED + attempt_prompt)
                    # save_data_snapshot(data, config, "bootstrap_attempts.json")
                    current_attempt.raw_prompts.append(copy.deepcopy(messages))
                    return messages, False
                else:
                    attempt_prompt = prompt
                    current_attempt.attempt_prompts.append({"role": "user", "content": attempt_prompt})
                    messages.append({"role": "user", "content": prompt})
                    logger.info(f"{colorama.Fore.RED + messages[-1]['role']} {'*'*100}")
                    # logger.debug(messages[-1]["content"])
                    logger.info(colorama.Fore.RED + attempt_prompt)
                    # save_data_snapshot(data, config, "bootstrap_attempts.json")
                    current_attempt.raw_prompts.append(copy.deepcopy(messages))
                    return None, True
            
        return initial_interaction
    
    # Replace ThreadPoolExecutor with anyio.create_task_group
    if data[0]['bootstrap_attempts'] == {}:
        async with anyio.create_task_group() as tg:
            async def process_env(i):
                while True:
                    try:
                        await converse(client, wrapper(i), copy.deepcopy(golden_attempts), config)
                        break
                    except Exception as e:
                        if not isinstance(e, KeyboardInterrupt):
                            logger.error(f"Error in {i}, {envs[i].env.taskdescription()}: {e}")
                            base_env = ScienceWorldEnvBase()
                            sciworld_env = SciWorldEnv(
                                task=envs[i].task,
                                env=base_env,
                                max_env_steps=config.max_env_steps,
                                api_key=config.api_key if config.use_openai_embedding else None
                            )
                            sciworld_env.reset()
                            envs[i] = sciworld_env
                            continue
                        else:
                            raise e
        
            for i in range(len(envs)):
                tg.start_soon(process_env, i)
    
    # Save data after bootstrap phase is complete
    save_data_snapshot(data, config, "bootstrap_attempts_final.json")

    # Print reward average for each sample after bootstrap
    for i in range(config.num_envs):
        rewards = []
        for attempt in data[i]['bootstrap_attempts'].values():
            rewards.extend(attempt.rewards)
        logger.info(f"Sample {i+1} bootstrap reward sum: {sum(rewards):.2f}")
    
    # main loop
    logger.info(f"Processing {config.num_envs} environments in {config.rounds} rounds...")
    
    for start_round in range(config.rounds):
        if not start_round in data[0]['round_attempts']:
            break
    for round_idx in range(start_round, config.rounds):
        logger.info(f"Round {round_idx+1}/{config.rounds}...")
        envs = load_envs(config.num_envs, config, gold_path=False)
        
        def wrapper2(i):
            env = envs[i]
            env_id = i
            
            # Initialize round attempt dictionary if it doesn't exist
            if round_idx not in data[env_id]['round_attempts']:
                data[env_id]['round_attempts'][round_idx] = {}
            
            # Initialize current attempt 
            data[env_id]['round_attempts'][round_idx][0] = Attempt()
            
            # Get current attempt object
            current_attempt = data[env_id]['round_attempts'][round_idx][0]
            
            first_round = True
            context_prompt = None
            
            def build_prompt(messages):
                nonlocal first_round
                nonlocal context_prompt
                nonlocal current_attempt
                
                if first_round:
                    prompt = f"{config.task_prompt_cot}"

                    if not config.max_attempts_in_context == 0:
                        prompt += f"\n<Attempts>\nYou can see several attempts below:"
                        messages.append({"role": "user", "content": prompt})
                        
                        # Use a single attempt counter for all attempts
                        attempt_buffer = []
                        attempt_counter = 1
                        
                        # Add all bootstrap attempts
                        for _, attempt_obj in data[env_id]['bootstrap_attempts'].items():
                            single_attempt = []
                            single_attempt.append({"role": "user", "content": f"Attempt {attempt_counter}:"})
                            single_attempt.extend(attempt_obj.get_processed_attempt_prompts(config))
                            attempt_buffer.append(single_attempt)
                            attempt_counter += 1
                        
                        # Add all previous round attempts
                        for prev_round_idx in range(round_idx):
                            for _, attempt_obj in data[env_id]['round_attempts'][prev_round_idx].items():
                                single_attempt = []
                                single_attempt.append({"role": "user", "content": f"Attempt {attempt_counter}:"})
                                single_attempt.extend(attempt_obj.get_processed_attempt_prompts(config))
                                attempt_buffer.append(single_attempt)
                                attempt_counter += 1

                        if config.max_attempts_in_context is not None:
                            # Take the last config.max_attempts_in_context attempts
                            attempt_buffer = attempt_buffer[-config.max_attempts_in_context:]
                        messages.extend(sum(attempt_buffer, []))

                        messages.append({"role": "user", "content": "</Attempts>"})

                    # Add instruction based on round type and task description
                    if config.icrl_mode == Methods.ICRL:
                        if config.explore_only:
                            instruction = config.exploration_instruction
                        elif config.explore_and_exploit:
                            instruction = config.explore_and_exploit_instruction
                        elif config.neutral_prompt:
                            instruction = config.neutral_round_instruction
                        elif config.exploit_only:
                            instruction = config.exploitation_instruction
                        else:
                            instruction = config.exploration_instruction if round_idx%2==0 else config.exploitation_instruction
                    elif config.icrl_mode == Methods.RANDOM_SAMPLING:
                        instruction = config.neutral_round_instruction
                    else:
                        raise ValueError(f"Invalid ablation mode: {config.icrl_mode}")
                    messages.append({"role": "user", "content": f"""<Instructions>{instruction}</Instructions>\n{env.env.taskdescription()}"""})
                    messages = merge_same_role_messages(messages)
                    
                    # Record raw prompt and attempt prompt
                    current_attempt.raw_prompts.append(copy.deepcopy(messages))
                    current_attempt.attempt_prompts.append({"role": "user", "content": env.env.taskdescription()})
                    
                    if config.react:
                        messages.append({"role": "assistant", "content": "Think: "})
                    
                    first_round = False
                    return messages, False
                else:
                    assert messages[-1]["role"] == "assistant", "It's assistant's turn"
                    if config.react:
                        messages = merge_same_role_messages(messages)

                    # Record raw prompt and attempt prompt
                    current_attempt.attempt_prompts.append({"role": "assistant", "content": messages[-1]["content"]})
                    
                    if context_prompt is not None:
                        messages[-2]["content"] = context_prompt
                    
                    prompt, state = env.step(messages[-1]["content"])
                    context_prompt = prompt + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                    
                    # Record the reward
                    current_attempt.rewards.append(state.reward)
                    
                    if not state.finished:
                        attempt_prompt = prompt + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                        current_attempt.attempt_prompts.append({"role": "user", "content": attempt_prompt})
                        augmented_prompt = prompt + "\n" + config.available_actions + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                        messages.append({"role": "user", "content": augmented_prompt})
                        current_attempt.raw_prompts.append(copy.deepcopy(messages))
                        if config.react:
                            messages.append({"role": "assistant", "content": "Think: "})
                        return messages, False
                    else:
                        attempt_prompt = prompt
                        current_attempt.attempt_prompts.append({"role": "user", "content": attempt_prompt})
                        messages.append({"role": "user", "content": prompt})
                        current_attempt.raw_prompts.append(copy.deepcopy(messages))
                        # Save snapshot after each completed round
                        # save_data_snapshot(data, config, f"sciworld_data_{round_idx}.json")
                        return None, True
            
            return build_prompt
        
        def wrapper_reflexion(i):
            env = envs[i]
            env_id = i
            
            # Initialize round attempt dictionary if it doesn't exist
            if round_idx not in data[env_id]['round_attempts']:
                data[env_id]['round_attempts'][round_idx] = {}
            
            # Initialize current attempt
            data[env_id]['round_attempts'][round_idx][0] = Attempt(reflexion=True)
            
            # Get current attempt object
            current_attempt = data[env_id]['round_attempts'][round_idx][0]
            
            round = 0
            context_prompt = None   

            def build_prompt(messages):
                nonlocal round
                nonlocal context_prompt
                nonlocal current_attempt
                
                if round == 0:
                    messages.append({"role": "user", "content": f"{config.task_prompt_cot}"})

                    # add reflections
                    added_preamble = False
                    for prev_round_idx in range(round_idx):
                        for _, attempt_obj in data[env_id]['round_attempts'][prev_round_idx].items():
                            if not added_preamble:
                                messages.append({"role": "assistant", "content": f"<Reflections>\nPrevious reflections:"})
                                added_preamble = True
                            messages.append({"role": "assistant", "content": "<Reflection>"})
                            messages.extend(attempt_obj.get_processed_attempt_prompts(config))
                            messages.append({"role": "assistant", "content": "</Reflection>"})
                    if added_preamble:
                        messages.append({"role": "assistant", "content": "</Reflections>"})
                    
                    messages.append({"role": "user", "content": f"<Instructions>{config.use_reflexion_instruction}</Instructions>\n{env.env.taskdescription()}"})
                    
                    messages = merge_same_role_messages(messages)
                    current_attempt.raw_prompts.append(copy.deepcopy(messages))

                    if env_id == 0:
                        print(f"{colorama.Fore.WHITE}{messages[-1]['role']}:\n{messages[-1]['content']}\n{'='*100}")
                    
                    round += 1
                    return messages, False
                elif round > 0:
                    assert messages[-1]["role"] == "assistant", "It's assistant's turn"
                    if env_id == 0:
                        print(f"{colorama.Fore.GREEN}{messages[-1]['role']}:\n{messages[-1]['content']}\n{'='*100}")
                    
                    if context_prompt is not None:
                        messages[-2]["content"] = context_prompt

                    prompt, state = env.step(messages[-1]["content"])

                    context_prompt = prompt + f" (Reward: {state.reward})" + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                    
                    # Record the reward
                    current_attempt.rewards.append(state.reward)
                    
                    if not state.finished:
                        augmented_prompt = prompt + "\n" + config.available_actions + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                        messages.append({"role": "user", "content": augmented_prompt})
                        current_attempt.raw_prompts.append(copy.deepcopy(messages))
                        if env_id == 0:
                            print(f"{colorama.Fore.WHITE}{messages[-1]['role']}:\n{messages[-1]['content']}\n{'='*100}")
                        return messages, False
                    else:
                        prompt_reflexion = f"{prompt}\n<Instructions>{config.do_reflexion_instruction}</Instructions>"

                        messages.append({"role": "user", "content": prompt_reflexion})
                        current_attempt.raw_prompts.append(copy.deepcopy(messages))
                        round = -1
                        if env_id == 0:
                            print(f"{colorama.Fore.WHITE}{messages[-1]['role']}:\n{messages[-1]['content']}\n{'='*100}")
                        return messages, False
                elif round == -1:
                    assert messages[-1]["role"] == "assistant", "It's assistant's turn"

                    current_attempt.raw_prompts.append(copy.deepcopy(messages))
                    current_attempt.attempt_prompts.append(messages[-1])
                    if env_id == 0:
                        print(f"{colorama.Fore.GREEN}{messages[-1]['role']}:\n{messages[-1]['content']}\n{'='*100}")
                    return None, True
                
                    
            return build_prompt
                
        # Process environments in parallel
        async with anyio.create_task_group() as tg:
            async def process_env_idx(i):
                assert config.exploration_temperature == config.exploitation_temperature == 1.0 or config.icrl_mode == Methods.ICRL, "Exploration and exploitation temp only supported for ICRL"
                temperature = config.exploration_temperature if round_idx % 2 == 0 else config.exploitation_temperature
                if config.icrl_mode == Methods.REFLEXION:
                    wrapper = wrapper_reflexion
                else:
                    wrapper = wrapper2
                while True:
                    try:
                        await converse(client, wrapper(i), [], config, temperature=temperature)
                        break
                    except Exception as e:
                        if not isinstance(e, KeyboardInterrupt):
                            logger.error(f"Error in {i}, {envs[i].env.taskdescription()}: {e}")
                            base_env = ScienceWorldEnvBase()
                            sciworld_env = SciWorldEnv(
                                task=envs[i].task,
                                env=base_env,
                                max_env_steps=config.max_env_steps,
                                api_key=config.api_key if config.use_openai_embedding else None
                            )
                            sciworld_env.reset()
                            envs[i] = sciworld_env
                            continue
                        else:
                            raise e
            
            for i in range(config.num_envs):
                tg.start_soon(process_env_idx, i)
        
        # Save data snapshot after each round is complete
        save_data_snapshot(data, config, f"sciworld_data_round_{round_idx}_final.json", delete=f"sciworld_data_round_{round_idx-1}_final.json")

        # Print reward average for each sample in the current round
        for i in range(config.num_envs):
            rewards = data[i]['round_attempts'][round_idx][0].rewards
            logger.info(f"Sample {i+1} round {round_idx+1} reward sum: {sum(rewards):.2f}")
        
def convert_keys_to_int(obj):
    """Convert string keys to integers if possible."""
    if isinstance(obj, dict):
        return {int(k) if isinstance(k, str) and k.isdigit() else k: convert_keys_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_int(item) for item in obj]
    return obj

def find_sciworld_file(folder_path, raw_prompts=False):
    """Find the sciworld data file in a given folder."""
    if raw_prompts:
        pattern = os.path.join(folder_path, "raw_prompts_sciworld_data_round_*_final.json")
    else:
        pattern = os.path.join(folder_path, "sciworld_data_round_*_final.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No sciworld data file found in {folder_path}")
    return files[0]  # Return the first matching file


def load_data(folder_path):
    data = json.load(open(find_sciworld_file(folder_path)), object_hook=convert_keys_to_int)
    raw_data = json.load(open(find_sciworld_file(folder_path, raw_prompts=True)), object_hook=convert_keys_to_int)
    for env_id, env_data in data.items():
        # convert bootstrap data to Attempt objects
        bootstrap_attempts = {}
        for attempt_id, attempt_data in env_data.get('bootstrap_attempts', {}).items():
            attempt = Attempt(
                raw_prompts=raw_data[env_id]['round_attempts'][attempt_id],
                rewards=attempt_data.get('rewards', []),
                attempt_prompts=attempt_data.get('attempt_prompts', []),
                extra_fields=attempt_data.get('extra_fields', {}),
                reflexion=attempt_data.get('reflexion', False)
            )
            bootstrap_attempts[attempt_id] = attempt
        env_data['bootstrap_attempts'] = bootstrap_attempts
        
        # convert round data to Attempt objects
        round_attempts = {}
        for round_id, round_data in env_data.get('round_attempts', {}).items():
            round_attempts[round_id] = {}
            for attempt_id, attempt_data in round_data.items():
                attempt = Attempt(
                    raw_prompts=raw_data[env_id]['round_attempts'][round_id][attempt_id],
                    rewards=attempt_data.get('rewards', []),
                    attempt_prompts=attempt_data.get('attempt_prompts', []),
                    extra_fields=attempt_data.get('extra_fields', {}),
                    reflexion=attempt_data.get('reflexion', False)
                )
                round_attempts[round_id][attempt_id] = attempt
        env_data['round_attempts'] = round_attempts
    
    return data

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
        data = load_data(config.checkpoint_path)
    await run_evaluation(config, data)

if __name__ == "__main__":
    anyio.run(main)