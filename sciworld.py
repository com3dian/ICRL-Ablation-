import pickle
import os
import re
import json
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import anyio
from openai import AsyncOpenAI
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from typing import Literal
from scienceworld import ScienceWorldEnv as ScienceWorldEnvBase
from sciworld_armap.utils.replace_sciworld_score import sciworld_monkey_patch
from sciworld_armap.envs.sciworld_env import SciWorldEnv
from sciworld_armap.tasks.sciworld import SciWorldTask
from omegaconf import OmegaConf

# Add the parent directory to the Python path to find eval_agent
script_path = Path(__file__).resolve()
parent_dir = str(script_path.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

base_path = os.getcwd()

sciworld_monkey_patch()

@dataclass
class SciWorldConfig:
    sw_output_path: str = "ICL/sw/"  # ScienceWorld output path
    
    # Experiment modes
    icrl_mode: Literal["icrl", "exploration_only", "exploitation_only", "no_reward_exploration", "rejection_sampling"] = "icrl"
    # no_reward: bool = False
    # zero_reward: bool = False
    debug_run: bool = False
    
    # Mode parameter (special handling)
    # algorithm: Literal["demo", "ICRL"] = "ICRL"
    
    # Parameters
    # num_char: int = 200
    # num_weak_demo: int = 3000
    num_initial_attempts: int = 2
    # api_eval: bool = True
    max_env_steps: int = 15
    
    # Model configuration
    model_name: str = "gpt-4.1-mini"
    # model_name: str = "gpt-4.1-nano-2025-04-14"
    # judge_model_name: str = "gpt-4.1-mini"
    # checkpoint_path: str = "google/gemma-7b-it"  # Only for reference
    # base_model_id: str = "google/gemma-7b-it"    # For reference and tokenizer loading
    
    # Dataset configuration
    # dataset_name: str = "game24"
    # split: str = "test"
    max_eval_samples: int = 45
    num_envs: int = 4
    
    # Evaluation parameters
    rounds: int = 10
    # max_new_tokens: int = 1000
    # env_step_limit: int = 100
    
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
    
    exploration_instruction: str = """
Your location and the environment is reset now. It's your turn.
Look at the previous attempts and try to construct a plan that is different from every single one of the previous attempts, while making sure it is feasible as well. 
After thinking, make sure to write your action in the "Action: single_action" format. It is parsed by a script.
"""

    exploitation_instruction: str = """
Your location and the environment is reset now. It's your turn.
Based on the previous high reward attempts, try to construct a higher scoring plan while making sure it is feasible as well. 
After thinking, make sure to write your action in the "Action: single_action" format. It is parsed by a script.
"""

    neutral_round_instruction: str = """
Your location and the environment is reset now. It's your turn.
Take your time to think and then take the next action.
After thinking, make sure to write your action in the "Action: single_action" format. It is parsed by a script.
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

    # CLI arguments always have highest priority
    config = OmegaConf.merge(default_config, cli_conf)
    
    config.task_prompt_cot = config.task_prompt_cot.format(available_actions=config.available_actions)

    # Create a timestamped output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config.sw_output_path = Path(config.sw_output_path) / config.icrl_mode / timestamp
    
    # Apply debug mode settings if enabled
    if config.debug_run:
        config.num_envs = 1
        config.rounds = 10
        config.num_initial_attempts = 1
        config.model_name = "gpt-4.1-nano-2025-04-14"
        config.judge_model_name = "gpt-4.1-nano-2025-04-14"
        print("*"*100)
        print("Debug run")
        print("*"*100)
    
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
        task_num = i  #! weird
        task_name = task_names[task_num]
        # var_num = random.randint(0, 9)  # Most tasks have 10 variations
        var_num = 0
        
        for _ in range(micro_repeat):
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
                max_env_steps=config.max_env_steps
            )
            
            sciworld_env.reset()
            
            # Add to list of environments
            envs.append(sciworld_env)
    
    # print(f"Created {len(envs)} ScienceWorld environments with random tasks")
    return envs

# def evaluate_model_answer(model_answer, sample_question):
#     try:
#         lhs = sympify(model_answer.split("=")[0])
        
#         # Extract numbers from the model answer
#         pattern = r"[-+]?\d*\.\d+|\d+"
#         matches = re.findall(pattern, model_answer.split("=")[0])
        
#         operand = []
#         for m in matches:
#             operand.append(int(m))
        
#         true_operand = [int(i) for i in sample_question.split(" ")]
#         operand.sort()
#         true_operand.sort()
        
#         format_reward = 0
        
#         if len(operand) != len(true_operand):
#             format_reward = -30.0
#         else:
#             for i in range(len(operand)):
#                 if operand[i] != true_operand[i]:
#                     format_reward = -30.0
#                     break
                
#         return format_reward, lhs
#     except Exception as e:
#         print(f"Error with input '{model_answer}': {type(e).__name__}: {e}")
#         import traceback
#         traceback.print_exc()
#         return -30.0, 0

# def extract_evaluation_score(eval_response):
#     """
#     Extract the evaluation score from the model's response.
    
#     Args:
#         eval_response: The evaluation response
    
#     Returns:
#         float: Evaluation score
#     """
#     pattern = re.compile(r'\*\*Answer\*\*\s*:\s*([+-]?\d+)')
    
#     m = pattern.search(eval_response)
#     if m:
#         gpt_eval_reward = float(m.group(1))
#     else:
#         gpt_eval_reward = 0
    
#     return gpt_eval_reward

# def save_results(envs, config, round_idx):
#     """
#     Save the evaluation results.
    
#     Args:
#         envs: List of environment instances with results
#         config: Configuration object
#         round_idx: Current round index
#     """
#     # Compute aggregated results
#     avg_reward_list = []
#     last_reward_list = []
#     gen_list = []  # final generated text from each sample
#     output_list = []  # detailed output per sample
    
#     reward_design = "gpt_eval_reward"
#     for env in envs:
#         # Get rewards from each round
#         round_rewards = [float(entry[reward_design]) for entry in env["output"]]
#         avg_reward_list.append(np.mean(round_rewards))
#         last_reward_list.append(round_rewards[-1] if round_rewards else 0)
#         gen_list.append(env["output"][-1]["generated_text"] if env["output"] else "")
#         output_list.append(env["output"])
    
#     # Create directory and save files
#     task = 'refactored'
#     base_model_id = config.base_model_id
#     max_new_tokens = config.max_new_tokens
    
#     this_time_change = "ICRL_zero_reward_"
#     if config.rejection_sampling:
#         this_time_change += "rejection_simple_nano_48"
#     else:
#         this_time_change += "100_mini"
    
#     max_eval_samples = config.max_eval_samples
#     this_time_change += f"_evalnum_{max_eval_samples}"
#     run = f"{this_time_change}_n_{config.rounds}"
#     path = f"{base_path}/{config.path_output}/{task}/{base_model_id}/{run}"
    
#     os.makedirs(path, exist_ok=True)
    
#     with open(f'{path}/gen_list_n={config.rounds}_mt={max_new_tokens}.pkl', "wb") as f:
#         pickle.dump(gen_list, f)
#     with open(f'{path}/avg_reward_list_n={config.rounds}_mt={max_new_tokens}.pkl', "wb") as f:
#         pickle.dump(avg_reward_list, f)
#     with open(f'{path}/last_reward_list_n={config.rounds}_mt={max_new_tokens}.pkl', "wb") as f:
#         pickle.dump(last_reward_list, f)
#     with open(f'{path}/output_list.json', 'w') as f:
#         json.dump(output_list, f)
    
#     # Print results
#     num_envs = len(envs)
#     print(f"Evaluated on {num_envs} environments.")
#     print(f"All Reward Average: {np.mean(avg_reward_list):.2%}")
#     print(f"Last Reward Average: {np.mean(last_reward_list):.2%}")
    
#     # Save summary
#     with open(f"{path}/all_reward_avg_n={config.rounds}_mt={max_new_tokens}.txt", "w") as f:
#         f.write(f"All Reward Average: {np.mean(avg_reward_list):.2%}")
#     with open(f"{path}/last_reward_avg_n={config.rounds}_mt={max_new_tokens}.txt", "w") as f:
#         f.write(f"Last Reward Average: {np.mean(last_reward_list):.2%}")

async def converse(client: AsyncOpenAI, f, messages, config):
    """
    Converse with the model.
    
    Args:
        client: AsyncOpenAI client
        f: function that
            takes messages on each call (initial argument on first call)
            outputs messages and done flag on each call (return value and done flag on last call)
        messages: initial input to f
        config: configuration object
    """
    while True:
        messages, done = f(messages)
        if not done:
            response = await client.chat.completions.create(
                model=config.model_name, 
                messages=[{"role": m["role"], "content": m["content"]} for m in messages]
            )
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

async def run_evaluation(config):
    """
    Evaluate the model using batch inference.
    
    Args:
        config: Configuration object
    """
    # Initialize AsyncOpenAI client
    client = AsyncOpenAI(api_key=config.api_key)
    
    # gold shot
    # envs = load_envs(1, gold_path=True) 
    # golden_attempts = []
    # for env in envs:
    #     messages = []
    #     messages.append({"role": "user", "content": env.env.taskdescription()})
    #     print('user', '*'*100)
    #     print(messages[-1]["content"])
    #     for action in env.env.get_gold_action_sequence():
    #         response = f"Thought: ...\nAction: {action}"
    #         messages.append({"role": "assistant", "content": response})
    #         print('assistant', '*'*100)
    #         print(response)
    #         observation, state = env.legacy_step(action)
    #         messages.append({"role": "user", "content": observation + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())})
    #         print('user', '*'*100)
    #         print(messages[-1]["content"])
    #         if state.finished:
    #             break
    #     golden_attempts.append(messages)
    from sciworld_armap.envs.base import raw_icl
    golden_attempts = raw_icl
    for attempt in golden_attempts[0]:
        print(attempt['role'], '*'*100)
        print(attempt['content'])

    # bootstrap shot
    envs = load_envs(config.num_envs, config, gold_path=False, micro_repeat=config.num_initial_attempts)
    
    def wrapper(env):
        first_round = True
        attempt = []
        context_prompt = None
        def initial_interaction(messages):
            nonlocal first_round
            nonlocal context_prompt
            if first_round:
                messages = messages[0]
                messages[0]["content"] = f"""{config.task_prompt_cot}\n<Attempt>\nHere's an example (without the Thought part):\n{messages[0]["content"]}"""
                messages[-1]["content"] = f"""{messages[-1]["content"]}\n</Attempt>\n<Instructions>\n{config.neutral_round_instruction}\n</Instructions>\n{env.env.taskdescription()}"""                
                attempt.append({"role": "user", "content": env.env.taskdescription()})
                first_round = False
                print(messages[-1]["role"], '*'*100)
                print(messages[-1]["content"])
                return messages, False
            else:
                assert messages[-1]["role"] == "assistant", "It's assistant's turn"
                attempt.append({"role": "assistant", "content": messages[-1]["content"]})
                print(messages[-1]["role"], '*'*100)
                print(messages[-1]["content"])
                if context_prompt is not None:
                    messages[-2]["content"] = context_prompt

                prompt, state = env.step(messages[-1]["content"])
                context_prompt = prompt + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                if not state.finished:
                    attempt_prompt = f" (reward: {state.reward})" + prompt + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                    attempt.append({"role": "user", "content": attempt_prompt})
                    augmented_prompt = prompt + "\n" + config.available_actions + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                    messages.append({"role": "user", "content": augmented_prompt})
                    print(messages[-1]["role"], '*'*100)
                    print(messages[-1]["content"])
                    return messages, False
                else:
                    attempt_prompt = f" (reward: {state.reward})" + prompt
                    attempt.append({"role": "user", "content": attempt_prompt})
                    messages.append({"role": "user", "content": prompt})
                    print(messages[-1]["role"], '*'*100)
                    print(messages[-1]["content"])
                    return attempt, True
        return initial_interaction
    
    # Replace ThreadPoolExecutor with anyio.create_task_group
    api_outputs = []
    async with anyio.create_task_group() as tg:
        async def process_env(env, i):
            result = await converse(client, wrapper(env), golden_attempts.copy(), config)
            api_outputs.append((i, result))
        
        for i, env in enumerate(envs):
            tg.start_soon(process_env, env, i)
    
    api_outputs.sort(key=lambda x: x[0])
    api_outputs = [x[1] for x in api_outputs]

    data = defaultdict(lambda: {"attempts": [], "rewards": []})
    assert len(api_outputs) == config.num_envs * config.num_initial_attempts
    for i in range(config.num_envs):
        for j in range(config.num_initial_attempts):
            idx = i * config.num_initial_attempts + j
            data[i]["attempts"].append(api_outputs[idx])
            data[i]["rewards"].append(envs[idx].state.reward_history)
    
    Path(f"{base_path}/{config.sw_output_path}").mkdir(parents=True, exist_ok=True)
    with open(f"{base_path}/{config.sw_output_path}/bootstrap_attempts.json", "w") as f:
        json.dump(data, f)

    # print reward average for each sample
    for i in range(config.num_envs):
        print(f"Sample {i+1} reward average: {np.mean([a for b in data[i]['rewards'] for a in b]):.2%}")
    
    # main loop
    print(f"Processing {config.num_envs} environments in {config.rounds} rounds...")
    
    for round_idx in range(config.rounds):
        print(f"Round {round_idx+1}/{config.rounds}...")
        envs = load_envs(config.num_envs, config, gold_path=False)
        
        def wrapper2(i):
            env = envs[i]
            attempt = []
            first_round = True
            context_prompt = None
            def build_prompt(messages):
                nonlocal attempt
                nonlocal first_round
                nonlocal context_prompt
                if first_round:
                    prompt = f"{config.task_prompt_cot}\n<Attempt>\nYou can see several attempts below:"
                    messages.append({"role": "user", "content": prompt})
                    attempts = data[i]["attempts"]
                    for attempt_idx, attempt in enumerate(attempts):
                        messages.append({"role": "user", "content": f"Attempt {attempt_idx+1}:"})
                        messages.extend(attempt)
                    if config.icrl_mode == "icrl":
                        messages.append({"role": "user", "content": f"""<Instructions>\n{config.exploration_instruction if round_idx%2==0 else config.exploitation_instruction}\n</Instructions>\n{env.env.taskdescription()}"""})
                    elif config.icrl_mode == "rejection_sampling":
                        messages.append({"role": "user", "content": f"""<Instructions>\n{config.neutral_round_instruction}\n</Instructions>\n{env.env.taskdescription()}"""})
                    messages = merge_same_role_messages(messages)
                    attempt.append({"role": "user", "content": env.env.taskdescription()})
                    first_round = False
                    return messages, False
                else:
                    assert messages[-1]["role"] == "assistant", "It's assistant's turn"
                    attempt.append({"role": "assistant", "content": messages[-1]["content"]})
                    if context_prompt is not None:
                        messages[-2]["content"] = context_prompt
                    
                    prompt, state = env.step(messages[-1]["content"])
                    context_prompt = prompt + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                    if not state.finished:
                        attempt_prompt = f" (reward: {state.reward})" + prompt + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                        attempt.append({"role": "user", "content": attempt_prompt})
                        augmented_prompt = prompt + "\n" + config.available_actions + "\nAvailable objects: " + ', '.join(env.env.get_possible_objects())
                        messages.append({"role": "user", "content": augmented_prompt})
                        return messages, False
                    else:
                        attempt_prompt = f" (reward: {state.reward})" + prompt
                        attempt.append({"role": "user", "content": attempt_prompt})
                        messages.append({"role": "user", "content": prompt})
                        return attempt, True
            
            return build_prompt
        
        # Replace ThreadPoolExecutor with anyio
        api_outputs = []
        async with anyio.create_task_group() as tg:
            async def process_env_idx(i):
                result = await converse(client, wrapper2(i), [], config)
                api_outputs.append((i, result))
            
            for i in range(config.num_envs):
                tg.start_soon(process_env_idx, i)
        
        api_outputs.sort(key=lambda x: x[0])
        api_outputs = [x[1] for x in api_outputs]
        
        for i in range(config.num_envs):
            data[i]['attempts'].append(api_outputs[i])
            data[i]['rewards'].append(envs[i].state.reward_history)
            
        with open(f"{base_path}/{config.sw_output_path}/sciworld_data_{round_idx}.json", "w") as f:
            json.dump(data, f)

        # print reward average for each sample
        for i in range(config.num_envs):
            print(f"Sample {i+1} reward average: {np.mean(data[i]['rewards'][-1]):.2%}")
        
        
        # Build prompts for each sample
        # batch_prompts = []
        # for env in envs:
        #     prompt = build_prompt(env, round_idx, config)
        #     batch_prompts.append(prompt)
        
        # Generate model outputs
        # with ThreadPoolExecutor(max_workers=12) as pool:
            # api_outputs = list(pool.map(
            #     lambda p: client.responses.create(model=config.model_name, input=p).output_text,
            #     batch_prompts
            # ))
        
        # Process outputs and prepare evaluation prompts
        # eval_prompt_list = []
        # eval_prompt_full_answer_list = []
        
        # for i, output_obj in enumerate(api_outputs):
        #     generated_text = output_obj
        #     extracted_text, model_answer = extract_action(generated_text)
            
        #     eval_prompt_full_answer_list.append(extracted_text)
        #     eval_prompt_list.append(model_answer)
        
        # # Evaluate model answers
        # eval_result_list = []
        # format_reward_list = []
        # gpt_eval_prompt_list = []
        
        # for index, model_answer in enumerate(eval_prompt_list):
        #     # Evaluate format and correctness
        #     format_reward, eval_result = evaluate_model_answer(model_answer, envs[index]['question'])
        #     format_reward_list.append(format_reward)
        #     eval_result_list.append(eval_result)
            
        #     # Create evaluation prompt using the template from config
        #     eval_prompt = config.evaluation_prompt_template.format(
        #         sample_question=envs[index]['question'],
        #         full_answer=eval_prompt_full_answer_list[index]
        #     )
        #     gpt_eval_prompt_list.append(eval_prompt)
        
        # # Get evaluation scores from judge model
        # with ThreadPoolExecutor(max_workers=12) as pool:
        #     eval_api_outputs = list(pool.map(
        #         lambda p: client.responses.create(model=config.judge_model_name, input=p).output_text,
        #         gpt_eval_prompt_list
        #     ))
        
        continue
        
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
            # eval_response = eval_api_outputs[i]
            # gpt_eval_reward = extract_evaluation_score(eval_response)
            # gpt_eval_reward_str = f"{gpt_eval_reward:.2f}"
            
            # Update sample with new weak demo
            weak_demo = {
                "prompt": envs[i]["question"],
                "answer": extracted_text,
                "accuracy_reward": accuracy_reward_str,
                "format_reward": format_reward_str,
                "gpt_eval_reward": gpt_eval_reward_str
            }
            envs[i]["weak_demos"].append(weak_demo)
            
        for i in range(config.num_envs):
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

async def main():
    # Parse command line arguments and get a config object
    config = parse_args()
    await run_evaluation(config)

if __name__ == "__main__":
    anyio.run(main)
    
    # Run the appropriate mode
    # if config.mode == "demo":
    #     print("Running demo mode with a single Game24 puzzle...")
    #     run_demo(config)
    # else:
    #     print("Running evaluation mode...")
    #     run_evaluation(config)


