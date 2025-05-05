import pickle
import os
import re
import json
import sys
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
import random
from pathlib import Path
import pdb
from functools import partial

# Add the parent directory to the Python path to find eval_agent
script_path = Path(__file__).resolve()
parent_dir = str(script_path.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from omegaconf import OmegaConf, DictConfig

# Define base path for all directory references
base_path = os.getcwd()

# Keep only imports that are actually used
from openai import OpenAI
from sympy import sympify
from datasets import load_dataset
from typing import Literal
from scienceworld import ScienceWorldEnv as ScienceWorldEnvBase

# Import eval_agent modules
from sciworld_armap.utils.replace_sciworld_score import sciworld_monkey_patch
from sciworld_armap.envs.sciworld_env import SciWorldEnv
from sciworld_armap.tasks.sciworld import SciWorldTask

# Apply the monkey patch for ScienceWorld
sciworld_monkey_patch()

# Default configuration
DEFAULT_CONFIG = {
    # Experiment modes
    "rejection_sampling": False,
    "icrl_mode": "icrl",  # Literal["icrl", "exploration_only", "exploitation_only", "no_reward_exploration"],
    "no_reward": True,
    "zero_reward": False,
    "debug_run": True,
    
    # Mode parameter (special handling)
    "algorithm": "ICRL", # Literal["demo", "ICRL"] 
    
    # Parameters
    "num_char": 200,
    "num_weak_demo": 3000,
    "api_eval": True,
    
    # Model configuration
    "model_name": "gpt-4.1-mini",
    "judge_model_name": "gpt-4.1-mini",
    # "checkpoint_path": "google/gemma-7b-it",  # Only for reference
    # "base_model_id": "google/gemma-7b-it",    # For reference and tokenizer loading
    
    # Dataset configuration
    # "dataset_name": "game24",
    # "split": "test",
    "max_eval_samples": 45,
    "num_samples": 100,
    
    # Evaluation parameters
    "rounds": 10,
    "max_new_tokens": 1000,
    "env_step_limit": 100,
    
    # Path configuration
    "path_output": "ICL",
    
    # OpenAI API key
    "api_key": "sk-C8z62BDhmo4EW1bqOn2TTmdFR29ocUeZXLExkdmGS1T3BlbkFJQcA3zOug-aNTm98KC0Wjsv549b3OgxEGn9TKJknXMA",

    # Prompt templates
    "task_prompt_cot": """
You are a helpful assistant to do some scientific experiment in an environment.

<Environment description>
In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway  
You should explore the environment and find the items you need to complete the experiment.  
You can teleport to any room in one step.  
{available_actions}
</Environment description>
""",
    
    "exploration_instruction": """
Now, it's your turn and here is the task.
Look at the previous attempts and try to construct a plan that is different from every single one of the previous attempts, while making sure it is feasible as well. 
Keep the same "Thought: ... Action: single_action" format for your response.
""",

    "exploitation_instruction": """
Now, it's your turn and here is the task.
Based on the previous high reward attempts, try to construct a higher scoring plan while making sure it is feasible as well. 
Keep the same "Thought: ... Action: single_action" format for your response.
""",

    "neutral_round_instruction": """
Now, it's your turn and here is the task.
Take your time to think and then take the next action.
Make sure to write your final action in the "Action: single_action" format. It is parsed by a script.
""",

    "available_actions": """
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
teleport to LOC: teleport to a specific room  
focus on OBJ: signal intent on a task object  
eat OBJ: eat a food
go to OBJ: move to a new location
dunk OBJ into OBJ: dunk a container into a liquid
inventory: list agent's inventory
wait: task no action for 10 steps  
wait1: task no action for a step
""",

    # "no_reward_exploration_instruction": "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response. Provide a response that is different from every single one of the previous attempts demonstrated in the context, while making sure it correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format.",

#     "evaluation_prompt_template": """Please evaluate whether the proposed solution to the 24‐game is correct for this input:
# Question: {sample_question}
# Solution: {full_answer}

# – If the solution is correct, return **Answer**: 10.  
# – If it is incorrect, look at the numbers in each "left: ..." step and judge how likely they are to reach 24:
#    • sure → 3  
#    • likely → 2  
#    • impossible → 1  
# - If the step is invalid, such as using numbers outside of the 4 input numbers or not using all numbers at the last step, the score is 0. 

# Add up those step‐scores and return **Answer**: <sum of likeliness scores>."""
}

def parse_args():
    """
    Parse command line arguments and create a configuration object using OmegaConf
    
    Returns:
        tuple: (config, run_mode) where config is the OmegaConf object and run_mode indicates what to run
    """
    # Parse command line arguments
    config = OmegaConf.create(DEFAULT_CONFIG)  # Start with code defaults

    cli_conf = OmegaConf.from_cli()

    # CLI arguments always have highest priority
    config = OmegaConf.merge(config, cli_conf)
    
    config.task_prompt_cot = config.task_prompt_cot.format(available_actions=config.available_actions)
    
    # Apply debug mode settings if enabled
    if config.debug_run:
        config.num_samples = 1
        config.rounds = 10
        config.model_name = "gpt-4.1-nano-2025-04-14"
        config.judge_model_name = "gpt-4.1-nano-2025-04-14"
        print("*"*100)
        print("Debug run")
        print("*"*100)
    
    return config


def load_envs(num_envs, env_step_limit, gold_path=False):
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
        task_num = random.randint(0, len(task_names) - 1)
        task_name = task_names[task_num]
        var_num = random.randint(0, 9)  # Most tasks have 10 variations
        
        # Create SciWorldTask instance
        task = SciWorldTask(
            task_id=f"{task_name}-{var_num}",
            sub_task_name=task_name,
            variation_idx=var_num
        )

        # Initialize SciWorldEnv from eval_agent
        max_steps = env_step_limit
        
        sciworld_env = SciWorldEnv(
            instruction_path="",  # doesn't matter, prompting is disabled
            icl_path="",          # doesn't matter, prompting is disabled
            task=task,
            env=base_env,
            max_steps=max_steps,
            gold_path=gold_path
        )
        
        sciworld_env.reset()
        
        # Add to list of environments
        envs.append(sciworld_env)
    
    print(f"Created {len(envs)} ScienceWorld environments with random tasks")
    return envs

def evaluate_model_answer(model_answer, sample_question):
    try:
        lhs = sympify(model_answer.split("=")[0])
        
        # Extract numbers from the model answer
        pattern = r"[-+]?\d*\.\d+|\d+"
        matches = re.findall(pattern, model_answer.split("=")[0])
        
        operand = []
        for m in matches:
            operand.append(int(m))
        
        true_operand = [int(i) for i in sample_question.split(" ")]
        operand.sort()
        true_operand.sort()
        
        format_reward = 0
        
        if len(operand) != len(true_operand):
            format_reward = -30.0
        else:
            for i in range(len(operand)):
                if operand[i] != true_operand[i]:
                    format_reward = -30.0
                    break
                
        return format_reward, lhs
    except Exception as e:
        print(f"Error with input '{model_answer}': {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return -30.0, 0

def extract_evaluation_score(eval_response):
    """
    Extract the evaluation score from the model's response.
    
    Args:
        eval_response: The evaluation response
    
    Returns:
        float: Evaluation score
    """
    pattern = re.compile(r'\*\*Answer\*\*\s*:\s*([+-]?\d+)')
    
    m = pattern.search(eval_response)
    if m:
        gpt_eval_reward = float(m.group(1))
    else:
        gpt_eval_reward = 0
    
    return gpt_eval_reward

def save_results(envs, config, round_idx):
    """
    Save the evaluation results.
    
    Args:
        envs: List of environment instances with results
        config: Configuration object
        round_idx: Current round index
    """
    # Compute aggregated results
    avg_reward_list = []
    last_reward_list = []
    gen_list = []  # final generated text from each sample
    output_list = []  # detailed output per sample
    
    reward_design = "gpt_eval_reward"
    for env in envs:
        # Get rewards from each round
        round_rewards = [float(entry[reward_design]) for entry in env["output"]]
        avg_reward_list.append(np.mean(round_rewards))
        last_reward_list.append(round_rewards[-1] if round_rewards else 0)
        gen_list.append(env["output"][-1]["generated_text"] if env["output"] else "")
        output_list.append(env["output"])
    
    # Create directory and save files
    task = 'refactored'
    base_model_id = config.base_model_id
    max_new_tokens = config.max_new_tokens
    
    this_time_change = "ICRL_zero_reward_"
    if config.rejection_sampling:
        this_time_change += "rejection_simple_nano_48"
    else:
        this_time_change += "100_mini"
    
    max_eval_samples = config.max_eval_samples
    this_time_change += f"_evalnum_{max_eval_samples}"
    run = f"{this_time_change}_n_{config.rounds}"
    path = f"{base_path}/{config.path_output}/{task}/{base_model_id}/{run}"
    
    os.makedirs(path, exist_ok=True)
    
    with open(f'{path}/gen_list_n={config.rounds}_mt={max_new_tokens}.pkl', "wb") as f:
        pickle.dump(gen_list, f)
    with open(f'{path}/avg_reward_list_n={config.rounds}_mt={max_new_tokens}.pkl', "wb") as f:
        pickle.dump(avg_reward_list, f)
    with open(f'{path}/last_reward_list_n={config.rounds}_mt={max_new_tokens}.pkl', "wb") as f:
        pickle.dump(last_reward_list, f)
    with open(f'{path}/output_list.json', 'w') as f:
        json.dump(output_list, f)
    
    # Print results
    num_envs = len(envs)
    print(f"Evaluated on {num_envs} environments.")
    print(f"All Reward Average: {np.mean(avg_reward_list):.2%}")
    print(f"Last Reward Average: {np.mean(last_reward_list):.2%}")
    
    # Save summary
    with open(f"{path}/all_reward_avg_n={config.rounds}_mt={max_new_tokens}.txt", "w") as f:
        f.write(f"All Reward Average: {np.mean(avg_reward_list):.2%}")
    with open(f"{path}/last_reward_avg_n={config.rounds}_mt={max_new_tokens}.txt", "w") as f:
        f.write(f"Last Reward Average: {np.mean(last_reward_list):.2%}")

def converse(client: OpenAI, f, messages, config):
    while True:
        messages, done = f(messages)
        if not done:
            response = client.responses.create(model=config.model_name, input=messages)
            messages.append({"role": "assistant", "content": response.output_text})
        else:
            break
    return messages

def run_evaluation(config):
    """
    Evaluate the model using batch inference.
    
    Args:
        config: Configuration object
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=config.api_key)
    
    # gold shot
    envs = load_envs(1, config.env_step_limit, gold_path=True) 
    attempts = []
    for env in envs:
        messages = []
        messages.append({"role": "user", "content": env.env.taskdescription()})
        for action in env.env.get_gold_action_sequence():
            response = f"Thought: ...\nAction: {action}"
            messages.append({"role": "assistant", "content": response})
            observation, state = env.legacy_step(action)
            messages.append({"role": "user", "content": observation})
            attempts.append(messages)

    # bootstrap shot
    envs = load_envs(1, config.env_step_limit, gold_path=False)
    
    def wrapper(env):
        first_round = True
        def initial_interaction(messages):
            nonlocal first_round
            if first_round:
                messages = messages[0]
                messages[0]["content"] = f"""{config.task_prompt_cot}\n<Attempt>\nHere's an example (without the Thought part):\n{messages[0]["content"]}"""
                messages[-1]["content"] = f"""{messages[-1]["content"]}\n</Attempt>\n<Instructions>\n{config.neutral_round_instruction}\n</Instructions>\n{env.env.taskdescription()}"""                
                # prompt = f"{config.task_prompt_cot}\n"
                # prompt += "<Instructions>" + config.first_round_instruction + "</instructions>\n\n"
                # prompt += env.env.taskdescription()
                first_round = False
                for i in range(len(messages)):
                    print(messages[i]["role"], '*'*100)
                    print(messages[i]["content"])
                return messages, False
            else:
                assert messages[-1]["role"] == "assistant", "It's assistant's turn"
                print('assistant', '*'*100)
                print(messages[-1]["content"])
                messages[-2]["content"] = re.sub(f"\n{re.escape(config.available_actions)}", "", messages[-2]["content"])

                prompt, state = env.step(messages[-1]["content"])
                if not state.finished:
                    prompt += "\n" + config.available_actions + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                messages.append({"role": "user", "content": prompt})
                print('user', '*'*100)
                print(prompt)

                return messages, state.finished
        return initial_interaction
    
    with ThreadPoolExecutor(max_workers=10) as pool:
        api_outputs = list(pool.map(
            lambda env: converse(client, wrapper(env), attempts.copy(), config),
            envs
        ))
    
    num_envs = len(envs)
    print(f"Processing {num_envs} environments in {config.rounds} rounds...")
    
    # Run evaluation rounds
    for round_idx in range(config.rounds):
        print(f"Round {round_idx+1}/{config.rounds}...")
        
        # Build prompts for each sample
        batch_prompts = []
        for env in envs:
            prompt = build_prompt(env, round_idx, config)
            batch_prompts.append(prompt)
        
        # Generate model outputs
        with ThreadPoolExecutor(max_workers=12) as pool:
            api_outputs = list(pool.map(
                lambda p: client.responses.create(model=config.model_name, input=p).output_text,
                batch_prompts
            ))
        
        # Process outputs and prepare evaluation prompts
        eval_prompt_list = []
        eval_prompt_full_answer_list = []
        
        for i, output_obj in enumerate(api_outputs):
            generated_text = output_obj
            extracted_text, model_answer = extract_action(generated_text)
            
            eval_prompt_full_answer_list.append(extracted_text)
            eval_prompt_list.append(model_answer)
        
        # Evaluate model answers
        eval_result_list = []
        format_reward_list = []
        gpt_eval_prompt_list = []
        
        for index, model_answer in enumerate(eval_prompt_list):
            # Evaluate format and correctness
            format_reward, eval_result = evaluate_model_answer(model_answer, envs[index]['question'])
            format_reward_list.append(format_reward)
            eval_result_list.append(eval_result)
            
            # Create evaluation prompt using the template from config
            eval_prompt = config.evaluation_prompt_template.format(
                sample_question=envs[index]['question'],
                full_answer=eval_prompt_full_answer_list[index]
            )
            gpt_eval_prompt_list.append(eval_prompt)
        
        # Get evaluation scores from judge model
        with ThreadPoolExecutor(max_workers=12) as pool:
            eval_api_outputs = list(pool.map(
                lambda p: client.responses.create(model=config.judge_model_name, input=p).output_text,
                gpt_eval_prompt_list
            ))
        
        # Process results and update samples
        for i, output_obj in enumerate(api_outputs):
            generated_text = output_obj
            extracted_text, model_answer = extract_action(generated_text)
            
            # Get accuracy reward
            try:
                accuracy_reward_value = float(-np.abs(eval_result_list[i] - 24))
            except:
                accuracy_reward_value = -24.00
            if accuracy_reward_value == 0:
                accuracy_reward_value = 0.00
            accuracy_reward_str = f"{accuracy_reward_value:.2f}"
            
            # Get format reward
            format_reward = format_reward_list[i]
            format_reward_str = f"{format_reward:.2f}"
            
            # Get evaluation reward
            eval_response = eval_api_outputs[i]
            gpt_eval_reward = extract_evaluation_score(eval_response)
            gpt_eval_reward_str = f"{gpt_eval_reward:.2f}"
            
            # Update sample with new weak demo
            weak_demo = {
                "prompt": envs[i]["question"],
                "answer": extracted_text,
                "accuracy_reward": accuracy_reward_str,
                "format_reward": format_reward_str,
                "gpt_eval_reward": gpt_eval_reward_str
            }
            envs[i]["weak_demos"].append(weak_demo)
            
            # Record round output
            envs[i]["output"].append({
                "round": round_idx,
                "prompt": batch_prompts[i],
                "answer": model_answer,
                "generated_text": extracted_text,
                "accuracy_reward": accuracy_reward_str,
                "format_reward": format_reward_str,
                "gpt_eval_prompt": eval_prompt_full_answer_list[i],
                "gpt_eval_reward": gpt_eval_reward_str,
                "gpt_eval_response": eval_response
            })
        
        # Save intermediate results
        # with open(f"{base_path}/intermediate_round.pkl", "wb") as f:
        #     pickle.dump(envs, f)
            
        # Save results periodically
        if round_idx % 1 == 0:
            save_results(envs, config, round_idx)

if __name__ == "__main__":
    # Parse command line arguments and get a config object and run mode
    config = parse_args()
    run_evaluation(config)
    
    # Run the appropriate mode
    # if config.mode == "demo":
    #     print("Running demo mode with a single Game24 puzzle...")
    #     run_demo(config)
    # else:
    #     print("Running evaluation mode...")
    #     run_evaluation(config)


