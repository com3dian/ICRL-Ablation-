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

from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from itertools import chain
# Number of characters used to compute the reward.
num_char = 200


rejection_sampling = 0
ICRL = 1
exploration_only = 0
exploitation_only = 0
no_reward_exploration = 0
exploration_or_exploitation = 0
no_reward = 0
zero_reward = 0


load_samples = 0

num_weak_demo = 300000


api_eval = True


client = OpenAI(api_key="Your_API_Key")




no_reward_exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response. Provide a response that is different from every single one of the previous attempts demonstrated in the context, while making sure it correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

# simple instruction
exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> examples, each showing a candidate Response, and the Rewards for each step of the Response. Your task: Based on the previous attempts, try your best to produce a response that can achieve higher rewards, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response, the Reward for each step of the Response, and a total Return of the steps.  Provide a response that is completely different for any steps from every single one of the previous attempts demonstrated in the context, while making sure it correctly follows the task instruction. Make sure the response correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

explore_or_exploit_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. You have two options, exploration or exploitation. For exploration, provide a response that is completely different for any steps from every single one of the previous attempts demonstrated in the context, while making sure it correctly follows the task instruction. Make sure the response correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format.\n For exploitation, based on the previous attempts, try your best to produce a response that can achieve higher rewards, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format, while making sure it correctly follows the task instruction, and put it in `<answer>…</answer>` format. Pick one option to follow."


text = ""


# import sys
# import os

# Add the directory to Python's path
sys.path.append('/tree-of-thought-llm/src')

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
    num_samples = 100
    task = get_task("game24")


    max_eval_samples = num_samples

    with open('game24_3steps.txt', 'r') as f:
        one_shot_prompt = f.read()
    
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    

    
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_new_tokens)
    

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
    
    
    if load_samples:
        with open("intermediate_round_game24_llm_optimizers.pkl", "rb") as f:
            samples = pickle.load(f)
        
    
    task_prompt = "You are a comedy writer competing for the highest laughs. Your goal is to write a short passage that would score a perfect 10 on the humor scale of a seasoned comedy critic. A score of 10 means the critic laughed out loud and found the content wildly entertaining, creative, and clever. You are allowed numerous attempts, each attempt are followed by a reward for your response, and will be placed in the context. Learn from these experience of attempts and provide a response that can achieve a higher reward from this comedy critic. \n"
    
    task_prompt = "Write a short humorous passage for the following prompt that matches with the following reward."
    task_prompt0 = "Write a short humorous passage for the following prompt."
    task_prompt = "Write a short passage for the following prompt that is as humorous as possible."
    
    task_prompt_cot = "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number."

    task_prompt = "Use all 4 numbers provided in the Input without repetition and basic arithmetic operations (+ - * /) to obtain 24. Only three steps are required and allowed."  
    # For the first step, you can only use numbers from the input. For the second and third steps, you can use the numbers from what is left after the previous step."
    task_prompt_cot = task_prompt
    # Run n rounds.
    for round_idx in range(n):
        print(f"Round {round_idx+1}/{n}...")
        # Build a prompt for each sample.
        batch_prompts = []
        for sample in samples:

            
            prompt = one_shot_prompt

            if round_idx == -1:

                prompt += "<attempt>\n"

                prompt += f"**Prompt**: Input: {sample['question']}\n"
                if not no_reward:
                    prompt += f"**Reward**: 0.00\n"
                
            else:
                if not rejection_sampling: 
                    # Add previous weak demonstrations if any.
                    for weak_demo in sample["weak_demos"][-num_weak_demo:]:
                        prompt += "<attempt>\n"

                        prompt += f"**Prompt**: Input: {weak_demo['prompt']}\n"

                        
                        ### steps reward here
                        
                        if not no_reward:
                        
                            try:
                                prompt +="**Response** Step" + weak_demo['answer'].split("Step")[1] 
                                prompt += f"<Reward: {weak_demo['gpt_eval_reward_list'][0]}> Step" + weak_demo['answer'].split("Step")[2] 
                                prompt += f"<Reward: {weak_demo['gpt_eval_reward_list'][1]}> Step" + weak_demo['answer'].split("Step")[3].split("**Answer**")[0] 
                                prompt += f"<Reward: {weak_demo['gpt_eval_reward_list'][2]}> **Answer**" + weak_demo['answer'].split("**Answer**")[1]
                            except:
                                prompt += "" + weak_demo['answer'][:] + "\n"
                        else:
                            prompt += "" + weak_demo['answer'][:] + "\n"
                        # prompt += f"**Format Reward**: {weak_demo['format_reward']}\n"
                        
                        if not no_reward: 
                            if zero_reward: 
                                prompt += f"**Reward**: {0.00}\n"
                            else:
                                # prompt += f"**Reward**: {weak_demo['gpt_eval_reward']}\n"
                                
                                # prompt += f"""**Reward**: Step 1 Reward: {weak_demo['gpt_eval_reward_list'][0]}, \n
                                # Step 2 Reward: {weak_demo['gpt_eval_reward_list'][1]}, \n
                                # Step 3 Reward: {weak_demo['gpt_eval_reward_list'][2]}, \n
                                # Final Answer Reward: {weak_demo['gpt_eval_reward_list'][3]}. \n
                                # \n"""
                                # prompt += f"""**Return**: {weak_demo['gpt_eval_return']}. \n"""
                                prompt += f"""**Reward**: {weak_demo['gpt_eval_return']}. \n"""
                            
                        prompt += "</attempt>"
                    
                    prompt += "<instructions>\n"
                    # may need to adjust the one_shot_prompt demonstrations once we start explore and exploit.
                    
                    if ICRL: 
                        if round_idx % 2 == 1:
                            prompt += exploitation_instruction
                        else:
                            prompt += exploration_instruction  
                        # if round_idx % 2 == 0:
                        #     prompt += exploration_instruction
                        # else:
                        #     prompt += exploitation_instruction
                    if exploration_only:
                        prompt += exploration_instruction
                    if exploitation_only:
                        prompt += exploitation_instruction
                    if no_reward_exploration: 
                        prompt += no_reward_exploration_instruction
                    if exploration_or_exploitation: 
                        prompt += explore_or_exploit_instruction


                prompt += " Only make one attempt, and put your answer in `<answer>**Response** Step1: ... (left: ...) Step2: ... (left: ...) Step3: ... (left: ...) **Answer**: <math operations of the 4 input numbers, even if it does not equal 24></answer>` format. Whether the Answer is correct or incorrect, do not try again. \n"

                prompt += "\n</instructions>\n"
                prompt += "<attempt>\n"

                prompt += f"**Prompt**: Input: {sample['question']}\n"

            batch_prompts.append(prompt)
        

        model_name = "gpt-4.1"

            
        with ThreadPoolExecutor(max_workers=12) as pool:
            api_outputs = list(pool.map(
                lambda p: client.responses.create(model=model_name, input=p).output_text,
                batch_prompts
            ))
        
        eval_prompt_list = []
        eval_prompt_full_answer_list = []
        
        
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
        
                
            eval_prompt_full_answer_list.append(generated_text)
            eval_prompt_list.append(model_answer)
        
        eval_result_list = []
        format_reward_list = []

        gpt_eval_prompt_list = []

        for index, model_answer in enumerate(eval_prompt_list):
            
        
            
            
            ### to-do for step wise reward eval. 
            all_steps_answer = eval_prompt_full_answer_list[index]
            try: 
                if len(all_steps_answer.split("**Response**")[1].split("Step 2:")) == 1:
                    first_step = all_steps_answer.split("**Response**")[1].split("Step2:")[0]
                else:
                    first_step = all_steps_answer.split("**Response**")[1].split("Step 2:")[0]
            except:
                first_step = ""
                    
                    
            try: 
                second_step = "(left from previous step: " + all_steps_answer.split("**Response**")[1].split("(left: ")[1]
                second_step += "(left:" + all_steps_answer.split("**Response**")[1].split("(left: ")[2].split("Step 3:")[0]
                if len(all_steps_answer.split("**Response**")[1].split("(left: ")[2].split("Step 3:")) == 1:
                    second_step = "(left from previous step: " + all_steps_answer.split("**Response**")[1].split("(left: ")[1]
                    second_step += "(left:" + all_steps_answer.split("**Response**")[1].split("(left: ")[2].split("Step3:")[0]
            except:
                second_step = ""
                
            try: 
                third_step = "(left from previous step: " + all_steps_answer.split("**Response**")[1].split("(left: ")[2]
                third_step +=  "(left:" + all_steps_answer.split("**Response**")[1].split("(left: ")[3].split("**Answer**:")[0]
            except:
                third_step = ""
                
                
            try:
                answer_step = all_steps_answer.split("**Answer**:")[1]
            except: 
                answer_step = ""
                

            
            eval_prompt_1 = f"""Rule of Game of 24: use all 4 numbers provided in the Input without repetition and basic arithmetic operations (+ - * /) to obtain 24. Only three steps are required and allowed.
            Given the following four numbers, {samples[index]['question']}, the first step to game of 24 is {first_step}. Evaluate this step:
            Look at the numbers in each “left: …” step and reason whether it is possible to reach 24:
            • sure → 3  
            • likely → 1  
            • impossible → 0 
            Return the score in the following `**Answer**: <integer score>.` format"""
            eval_prompt_2 = f"""Rule of Game of 24: use all 4 numbers provided in the Input without repetition and basic arithmetic operations (+ - * /) to obtain 24. Only three steps are required and allowed.
            Given the following three numbers from what is left from the game of 24, the second step is {second_step}. Evaluate this step: 
            Look at the numbers in each “left: …” step and reason whether it is possible to reach 24:
            • sure → 3  
            • likely → 1  
            • impossible → 0 
            Return the score in the following `**Answer**: <integer score>.` format"""
            eval_prompt_3 = f"""Rule of Game of 24: use all 4 numbers provided in the Input without repetition and basic arithmetic operations (+ - * /) to obtain 24. Only three steps are required and allowed.
            Given the following two numbers from what is left from the game of 24, the third step is {third_step}. Evaluate this step: 
            Look at the numbers in each “left: …” step and reason whether it is possible to reach 24:
            • sure → 3  
            • likely → 1  
            • impossible → 0 
            Return the score in the following `**Answer**: <integer score>.` format"""
            
            eval_prompt_4 = f"""Rule of Game of 24: use all 4 numbers provided in the Input without repetition and basic arithmetic operations (+ - * /) to obtain 24. Only three steps are required and allowed.
            Given the following four numbers, {samples[index]['question']}, the solution to game of 24 is {answer_step}. Evaluate this solution:
            If the solution is invalid, such as using numbers outside of the 4 input numbers or wrong calculations, the score is 0.
            Otherwise, the score is 10.
            Return the score in the following `**Answer**: <integer score>.` format"""
            
            eval_prompt = [eval_prompt_1, eval_prompt_2, eval_prompt_3, eval_prompt_4]
            
            
            
            
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
                eval_result_list.append(lhs)
            except Exception as e:
                # Print the error details
                print(f"Error with input '{model_answer}': {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                # Still append a default value
                eval_result_list.append(0)
                format_reward_list.append(-30.0)

        
        eval_model_name = "gpt-4.1-mini"
        eval_model_name = "gpt-4.1"
        

        flat_prompts = list(chain.from_iterable(gpt_eval_prompt_list))

        # 2. fire them off in one pool
        with ThreadPoolExecutor(max_workers=20) as pool:
            flat_results = list(pool.map(
                lambda prompt: client.responses.create(model=eval_model_name, input=prompt).output_text,
                flat_prompts
            ))

        # 3. reshape back into groups of 4
        group_size = 4
        eval_api_outputs = [
            flat_results[i*group_size:(i+1)*group_size]
            for i in range(len(gpt_eval_prompt_list))
        ]

        # Process batch responses.
        for i, output_obj in enumerate(eval_api_outputs):
            # Retrieve generated text.
            generated_text = output_obj

            pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

            # first or last, could be an important difference

            try:
                generated_text = pattern.findall(generated_text)[0]
            except:
                generated_text = ""

            m = re.search(r'^(?:.*\*\*Answer\*\*):\s*(.*)$', generated_text, re.DOTALL)
            if m:
                model_answer = m.groups()[-1].strip()
            else:
                model_answer = ""

            

            
        
            print('+='*20)
            print('eval_result_list[i]:', eval_result_list[i])
            
            
            try:
                accuracy_reward_value = float(-np.abs(eval_result_list[i] - 24))
            except:
                accuracy_reward_value = -24.00
            # accuracy_reward_value = float(-np.abs(eval_result_list[i] - 24))
            if accuracy_reward_value == 0:
                accuracy_reward_value = 0.00
            accuracy_reward_str = f"{accuracy_reward_value:.2f}"
            
            print("accuracy_reward_str", accuracy_reward_str)
            print('accuracy_reward_value', accuracy_reward_value)


            format_reward = format_reward_list[i]
            format_reward_str = f"{format_reward:.2f}"


            eval_responses = eval_api_outputs[i]
            
            
            gpt_eval_reward_list = []
            gpt_eval_reward_str_list = []
            for eval_response in eval_responses:
                pattern = re.compile(r'\*\*Answer\*\*\s*:\s*([+-]?\d+)')

                m = pattern.search(eval_response)
                if m:
                    gpt_eval_reward = m.group(1)
                    gpt_eval_reward = float(gpt_eval_reward)
                    print("gpt_eval_reward", gpt_eval_reward)  # -> '1'
                else:
                    print("No match")
                    gpt_eval_reward = 0

                gpt_eval_reward_str = f"{gpt_eval_reward:.2f}"
                
                
                gpt_eval_reward_list.append(gpt_eval_reward)
                gpt_eval_reward_str_list.append(gpt_eval_reward_str)
                
                
            gpt_eval_return = np.sum(gpt_eval_reward_list)
            gpt_eval_return_str = f"{gpt_eval_return:.2f}"



            weak_demo = {
                "prompt": samples[i]["question"],
                "answer": eval_prompt_full_answer_list[i],
                "accuracy_reward": accuracy_reward_str,
                "format_reward": format_reward_str,
                "gpt_eval_reward_list": gpt_eval_reward_str_list,
                "gpt_eval_return":gpt_eval_return_str
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
                "model_answer": eval_prompt_full_answer_list[i],
                "gpt_eval_prompt": gpt_eval_prompt_list[i],
                "gpt_eval_reward_list": gpt_eval_reward_str_list,
                "gpt_eval_return": gpt_eval_return_str,
                "gpt_eval_response": eval_responses
            })
        
        with open("intermediate_round_game24_llm_optimizers.pkl", "wb") as f:
            pickle.dump(samples, f)
            
        if round_idx % 1 == 0:

            # After all rounds, compute aggregated results.
            avg_reward_list = []
            last_reward_list = []
            gen_list = []  # final generated text from each sample.
            output_list = []  # detailed output per sample (each is a list of round outputs).

            reward_design = "gpt_eval_return"
            for sample in samples:
                # Get rewards from each round.
                round_rewards = [float(entry[reward_design]) for entry in sample["output"]]
                avg_reward_list.append(np.mean(round_rewards))
                last_reward_list.append(round_rewards[-1] if round_rewards else 0)
                gen_list.append(sample["output"][-1]["generated_text"] if sample["output"] else "")
                output_list.append(sample["output"])

            # Save the results to files.
            task = 'alpaca_game24_api'
            
            this_time_change = ""
            
            
            
            if rejection_sampling:
                this_time_change += "rejection_100"
            else:
                this_time_change += "4.1_eval_llm_as_optimizer_reward_only_ICRL"
                
            this_time_change += "_4.1"
            

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


