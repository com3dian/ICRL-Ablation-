import pickle
import os
import re
import tqdm
import json
import time
import sys
import numpy as np
# import torch
import argparse
from utils import *
from prm_eval import *

from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
# Number of characters used to compute the reward.
num_char = 200
num_weak_demo = 3000
rejection_sampling = True
api_eval = True


client = OpenAI(api_key="Your_API_Key")


sys.path.append('tree-of-thought-llm/src')

from sympy import sympify
from tot.tasks import get_task


from tot.prompts.game24 import (
    standard_prompt,
    cot_prompt,
    propose_prompt,
    value_prompt,
    value_last_step_prompt
)



def evaluate_checkpoint(
    checkpoint_path="game_of_24",
    base_model_id="GPT-4.1",
    max_eval_samples=45,
    n=51,
    max_new_tokens=1000
):
    num_samples = 48
    task = get_task("game24")


    with open('game24_rejection.txt', 'r') as f:
        one_shot_prompt = f.read()
    
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    

    
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_new_tokens)
    
    # Prepare a list for all samples that meet the criteria.
    # Each entry stores the question, a history of weak demos, and outputs per round.
    samples = []

    for idx in range(num_samples):
        question = task.get_input(900+idx)
        samples.append({
            "question": question,
            "weak_demos": [],  # will hold dictionaries with keys: prompt, answer, reward
            "output": []       # will record output details per round
        })
    
    num_samples = len(samples)
    print(f"Processing {num_samples} samples in {n} rounds...")

    


    task_prompt = "Use all 4 numbers provided in the Input without repetition and basic arithmetic operations (+ - * /) to obtain 24. Only three steps are required and allowed."  

    task_prompt_cot = task_prompt
    # Run n rounds.
    for round_idx in range(n):
        print(f"Round {round_idx+1}/{n}...")
        # Build a prompt for each sample.
        batch_prompts = []
        for sample in samples:
            
            
            prompt = one_shot_prompt
            # prompt = ""
            # prompt += f"**Task**: {task_prompt_cot}\n"
            if round_idx == -1:
                # prompt += cot_prompt.format(input=sample['question']) + "\nSteps:"
                prompt += "<attempt>\n"
                # prompt += f"**Task**: {task_prompt_cot}"
                prompt += f"**Prompt**: Input: {sample['question']}\n"
                # prompt += f"**Format Reward**: 0.00\n"
                prompt += f"**Reward**: 0.00\n"
                
            else:
                if not rejection_sampling: 
                    # Add previous weak demonstrations if any.
                    for weak_demo in sample["weak_demos"][-num_weak_demo:]:
                        prompt += "<attempt>\n"
                        # prompt += f"**Task**: {task_prompt_cot}"
                        prompt += f"**Prompt**: Input: {weak_demo['prompt']}\n"
                        # prompt += f"**Format Reward**: {weak_demo['format_reward']}\n"
                        # prompt += f"**Reward**: {weak_demo['gpt_eval_reward']}\n"
                        # Steps: ??
                        prompt += "" + weak_demo['answer'][:] + "\n"
                        # prompt += f"**Format Reward**: {weak_demo['format_reward']}\n"
                        prompt += f"**Reward**: {weak_demo['gpt_eval_reward']}\n"
                        prompt += "</attempt>"
                    
                    prompt += "<instructions>\n"
                    # may need to adjust the one_shot_prompt demonstrations once we start explore and exploit.
                    if round_idx % 2 == 0:
                        prompt += exploration_instruction
                    else:
                        prompt += exploitation_instruction

                    prompt += "</instructions>\n"

                prompt += "<instructions>\n"
                prompt += "Only make one attempt, and put your answer in `<answer>**Response** Step1: ... (left: ...) Step2: ... (left: ...) Step3: ... (left: ...) **Answer**: <math operations of the 4 input numbers.></answer>` format. Whether the Answer is correct or incorrect, do not try again. \n"
                prompt += "</instructions>\n"
                prompt += f"**Input**: {sample['question']}\n"

                

            batch_prompts.append(prompt)
        

        model_name = "gpt-4.1-mini"

            
        with ThreadPoolExecutor(max_workers=12) as pool:
            api_outputs = list(pool.map(
                lambda p: client.responses.create(model=model_name, input=p).output_text,
                batch_prompts
            ))
        
        eval_prompt_list = []
        
        # if round_idx != 0:
        
        for i, output_obj in enumerate(api_outputs):
            # Retrieve generated text.
            generated_text = output_obj


            pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)


            try:
                generated_text = pattern.findall(generated_text)[0]

            except:
                generated_text = ""


            m = re.search(r'^(?:.*\*\*Answer\*\*):\s*(.*)$', generated_text, re.DOTALL)

            if m:
                model_answer = m.groups()[-1].strip()
            else:
                model_answer = ""


            print("model_answer", model_answer)


            if "\\\\(" in model_answer:
                print("found \\\\("*20)
                model_answer = model_answer.replace("\\\\(", "")
   
            eval_prompt_list.append(model_answer)
        
        eval_result_list = []
        format_reward_list = []

        gpt_eval_prompt_list = []

        for index, model_answer in enumerate(eval_prompt_list):

            # create eval prompt
            eval_prompt = f"Evaluate if the given solution for the 24 game is correct for this input: {samples[index]['question']}. Solution: {model_answer}. Return -10 if correct. If incorrect, if the solution is invalid such as using the same number twice, using numbers other than the 4 numbers in the input or claiming there is no solution for the problem, return 10; if the solution is valid, but the answer is not 24, count the number of edits of the operations required to make it into a correct solution of 24. The higher the number of edits, the further away the solution is from the correct solution. Return the number of edits. Put your response in the following format: **Answer**: <integer number of edits>."
            gpt_eval_prompt_list.append(eval_prompt)

            print()
            print("eval_prompt", eval_prompt)
            print()
            try:
                lhs = sympify(model_answer.split("=")[0])
                expr = sympify(model_answer.split("=")[0], evaluate=False)
                # operand = [int(i) for i in expr.args]

                pattern = r"[-+]?\d*\.\d+|\d+"

                # 2. Find all substrings that look like numbers
                matches = re.findall(pattern, model_answer.split("=")[0])

                # 3. (Optional) Convert each to int or float
                operand = []
                for m in matches:
                    operand.append(int(m))

                true_operand = [int(i) for i in samples[index]['question'].split(" ")]
                operand.sort()
                true_operand.sort()

                print("model_answer", model_answer)
                print("operand", operand)
                print("true_operand", true_operand)

                format_reward = 0

                if len(operand) != len(true_operand):
                    format_reward = -30.0
                else:
                    for i in range(len(operand)):
                        if operand[i] != true_operand[i]:
                            format_reward = -30.0
                            break
                print("format_reward", format_reward)
                format_reward_list.append(format_reward)
                # expr = sympify(model_answer.split("=")[0], evaluate=False)
                # print(lhs)
                # print(lhs == 24)
                eval_result_list.append(lhs)
            except Exception as e:
                # Print the error details
                print(f"Error with input '{model_answer}': {type(e).__name__}: {e}")
                # Optionally print full traceback:
                import traceback
                traceback.print_exc()
                # Still append a default value
                eval_result_list.append(0)
                format_reward_list.append(-30.0)

        #     sampling_params_eval = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

        eval_model_name = "gpt-4.1-mini"
            
        with ThreadPoolExecutor(max_workers=12) as pool:
            eval_api_outputs = list(pool.map(
                lambda p: client.responses.create(model=eval_model_name, input=p).output_text,
                gpt_eval_prompt_list
            ))

        # Process batch responses.
        for i, output_obj in enumerate(api_outputs):
            # Retrieve generated text.
            generated_text = output_obj
            
            # answer template
            pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

            # first or last, could be an important difference

            try:
                generated_text = pattern.findall(generated_text)[0]
                # generated_text = generated_text[8:-9]
                # generated_text = generated_text.split("<answer>")[1]
            except:
                generated_text = ""

            

            m = re.search(r'^(?:.*\*\*Answer\*\*):\s*(.*)$', generated_text, re.DOTALL)
            if m:
                model_answer = m.groups()[-1].strip()
            else:
                model_answer = ""

            

            
                
            try:
                accuracy_reward_value = float(-np.abs(eval_result_list[i] - 24))
            except:
                accuracy_reward_value = -24.00
            if accuracy_reward_value == 0:
                accuracy_reward_value = 0.00
            accuracy_reward_str = f"{accuracy_reward_value:.2f}"
            
            print("accuracy_reward_str", accuracy_reward_str)
            print('accuracy_reward_value', accuracy_reward_value)


            format_reward = format_reward_list[i]
            format_reward_str = f"{format_reward:.2f}"


            eval_response = eval_api_outputs[i]

            

            pattern = re.compile(r'\*\*Answer\*\*\s*:\s*([+-]?\d+)')

            m = pattern.search(eval_response)
            if m:
                gpt_eval_reward = m.group(1)
                gpt_eval_reward = -float(gpt_eval_reward)
                print("gpt_eval_reward", gpt_eval_reward)  # -> '1'
            else:
                print("No match")
                gpt_eval_reward = 0

            

            gpt_eval_reward_str = f"{gpt_eval_reward:.2f}"



            
            weak_demo = {
                "prompt": samples[i]["question"],
                "answer": generated_text,
                "accuracy_reward": accuracy_reward_str,
                "format_reward": format_reward_str,
                "gpt_eval_reward": gpt_eval_reward_str
            }
            # Append to the sample's weak demo history.
            samples[i]["weak_demos"].append(weak_demo)
            # Record the round output.
            samples[i]["output"].append({
                "round": round_idx,
                "prompt": batch_prompts[i],
                "answer": model_answer,
                "generated_text": generated_text,
                "accuracy_reward": accuracy_reward_str,
                "format_reward": format_reward_str,
                "gpt_eval_reward": gpt_eval_reward_str,
                "gpt_eval_response": eval_response
            })
        
        # Optionally, save intermediate results after each round.
        # For example, you could pickle the samples list:
        with open("intermediate_round.pkl", "wb") as f:
            pickle.dump(samples, f)
            
        if round_idx % 1 == 0:

            # After all rounds, compute aggregated results.
            avg_reward_list = []
            last_reward_list = []
            gen_list = []  # final generated text from each sample.
            output_list = []  # detailed output per sample (each is a list of round outputs).

            reward_design = "gpt_eval_reward"
            for sample in samples:
                # Get rewards from each round.
                round_rewards = [float(entry[reward_design]) for entry in sample["output"]]
                avg_reward_list.append(np.mean(round_rewards))
                last_reward_list.append(round_rewards[-1] if round_rewards else 0)
                gen_list.append(sample["output"][-1]["generated_text"] if sample["output"] else "")
                output_list.append(sample["output"])

            # Save the results to files.
            task = 'alpaca_game24_api'
            if rejection_sampling:
                this_time_change = "rejection_simple_prompt_48"
            else:
                this_time_change = "simple_nano_48"

            this_time_change += f"_evalnum_{max_eval_samples}"
            run = f"{this_time_change}_n_{n}"
            path = f"/sfs/weka/scratch/ks8vf/ICL/{task}/{base_model_id}/{run}"
            
            path = f"/sfs/weka/scratch/ks8vf/ICL/{task}/{base_model_id}/{run}"
            os.makedirs(path, exist_ok=True)

            with open(f'{path}/gen_list_n={n}_mt={max_new_tokens}.pkl', "wb") as f:
                pickle.dump(gen_list, f)
            with open(f'{path}/avg_reward_list_n={n}_mt={max_new_tokens}.pkl', "wb") as f:
                pickle.dump(avg_reward_list, f)
            with open(f'{path}/last_reward_list_n={n}_mt={max_new_tokens}.pkl', "wb") as f:
                pickle.dump(last_reward_list, f)
            with open(f'{path}/output_list.json', 'w') as f:
                json.dump(output_list, f)

            # Print final aggregated results.
            print(f"Evaluated on {num_samples} samples.")
            print(f"All Reward Average: {np.mean(avg_reward_list):.2%}")
            print(f"Last Reward Average: {np.mean(last_reward_list):.2%}")

            # Save summary to text files.
            with open(f"{path}/all_reward_avg_n={n}_mt={max_new_tokens}.txt", "w") as f:
                f.write(f"All Reward Average: {np.mean(avg_reward_list):.2%}")
            with open(f"{path}/last_reward_avg_n={n}_mt={max_new_tokens}.txt", "w") as f:
                f.write(f"Last Reward Average: {np.mean(last_reward_list):.2%}")

if __name__ == "__main__":
    print("Evaluating checkpoint in batch mode...")
    evaluate_checkpoint()
    # main()


