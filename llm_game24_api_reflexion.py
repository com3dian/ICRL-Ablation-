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

from itertools import chain
# Number of characters used to compute the reward.
num_char = 200


rejection_sampling = 0
ICRL = 0
exploration_only = 0
exploitation_only = 0
no_reward_exploration = 0
exploration_or_exploitation = 0
no_reward = 0
zero_reward = 0

num_weak_demo = 3000


api_eval = True

eval_prompt = "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24."
eval_prompt = "Evaluate if given numbers can reach 24 (sure/likely/impossible)"
# eval_prompt = f"Evaluate if the given solution for the 24 game is correct for this input: {}. Solution: {}. Return -10 if correct. If incorrect, if the solution is invalid such as using the same number twice or using numbers other than the 4 numbers in the input, return 10; if the solution is valid, but the answer is not 24, count the number of edits of the operations required to make it into a correct solution of 24. The higher the number of edits, the further away the solution is from the correct solution. Return the number of edits. Put your response in the following format: Answer: <integer number of edits>."

client = OpenAI(api_key="sk-C8z62BDhmo4EW1bqOn2TTmdFR29ocUeZXLExkdmGS1T3BlbkFJQcA3zOug-aNTm98KC0Wjsv549b3OgxEGn9TKJknXMA")


exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. Provide a response that is completely different from any previous attempts demonstrated in the context, and wrap it in `<answer>…</answer>`."

exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> entries. Each entry contains: •A candidate Response •Its Format Reward and Accuracy Reward.  Your task: 1. Parse all <attempt> blocks and identify the attempts with the top reward scores. 2. Among those top‐scoring attempts, select the ones that are most distinct from each other in style or angle. 3. Create a single new mega response that fuses the strongest elements from each distinct, high‐scoring attempt both in Format Reward and Accuracy Reward. 4. Return only this new, combined mega response wrapped in an <answer>…</answer> tag."

exploration_instruction = ""
exploitation_instruction = ""

exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. Provide a response that is completely different from any previous attempts demonstrated in the context, and put it in `**Response** Steps: ... **Answer**: ...` format."
exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> entries. Each entry contains: •A candidate Response •Its Format Reward and Accuracy Reward  Your task: 1. Parse all <attempt> blocks and identify the attempts with the top reward scores. 2. Among those top‐scoring attempts, select the ones that are most distinct from each other in style or angle. 3. Create a single new mega response that fuses the strongest elements from each distinct, high‐scoring attempt both in Format Reward and Accuracy Reward. 4. Return only this new, combined mega response, and put it in `**Response** Steps: ... **Answer**: <math operations of the 4 input numbers> format."


exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. Provide a response that is different from every single one of the previous attempts demonstrated in the context, while making sure it correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."
# exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> entries. Each entry contains: •A candidate Response •Its Format Reward and Accuracy Reward  Your task: make the best educated guess based on the high reward attempts to produce a more correct response that can achieve a higher reward, while making sure it correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... (Correct/Incorrect.) **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> entries. Each entry contains: •A candidate Response •Its Format Reward and Accuracy Reward  Your task: make the best educated guess based on the high reward attempts to produce a more correct response that can achieve a higher reward, while making sure it correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> entries. Each entry contains: •A candidate Response •Its Reward. From all the high reward responses, find the most promising response and modify it to recieve the full 10 reward, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

no_reward_exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response. Provide a response that is different from every single one of the previous attempts demonstrated in the context, while making sure it correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."
# use the least common high reward attempt as your starting point to make small edits that can achieve a higher reward, while making sure it correctly follows the task instruction, and put it in `**Response** Steps: ... **Answer**: ...` format.

### prompts for step wise reward and return.

# exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response, the Reward for each step of the Response, and a total Return of the steps. Provide a response that contains at least one step that is different from every single one of the previous attempts demonstrated in the context. Start your exploration at the earliest step that recieves a score of 0. Make sure the response correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> examples, each showing a candidate Response, the Reward for each step of the Response, and a total Return of the steps. Your task: From all the high reward responses, find the most promising response and modify it to recieve the highest return, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response, the Reward for each step of the Response, and a total Return of the steps.  Provide a response that is completely different for any steps from every single one of the previous attempts demonstrated in the context, while making sure it correctly follows the task instruction. Make sure the response correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

# exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response, the Reward for each step of the Response, and a total Return of the steps. Explore a different response than provided in the attempt. Give priority for exploring a different steps to the early steps, such that step 1 > step 2 > step 3. Avoid repeating the same step that result in a reward of 0, starting from Step 1."

# exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> examples, each showing a candidate Response, the Reward for each step of the Response, and a total Return of the steps. Your task: make the best educated guess based on the high reward steps and high return attempts to produce a more correct response that can achieve a higher reward and return, while making sure it correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

# exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> examples, each showing a candidate Response, the Reward for each step of the Response, and a total Return of the steps. Your task: Strictly follow steps that has shown to recieve the highest reward, starting from Step 1, Step 2, etc. Then only improve on subsequent steps that doesn't achieve a high reward. Make sure the response correctly follows the task instruction."

# exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> entries. Each entry contains: •A candidate Response •Its Format Reward and Accuracy Reward  Your task: make the best educated guess based on the high reward attempts to produce a more correct response that can achieve a higher reward, while making sure it correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."


explore_or_exploit_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. You have two options, exploration or exploitation. For exploration, provide a response that is completely different for any steps from every single one of the previous attempts demonstrated in the context, while making sure it correctly follows the task instruction. Make sure the response correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format.\n For exploitation, from all the high reward responses, find the most promising response and modify it to receive the highest return, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format, while making sure it correctly follows the task instruction, and put it in `<answer>…</answer>` format. Pick one option to follow."

text = ""


# import sys
# import os

# Add the directory to Python's path
sys.path.append('/sfs/weka/scratch/ks8vf/ICL/tree-of-thought-llm/src')

from sympy import sympify
from tot.tasks import get_task


from tot.prompts.game24 import (
    standard_prompt,
    cot_prompt,
    propose_prompt,
    value_prompt,
    value_last_step_prompt
)

def main():
    # 1. Initialize the Game24 task; this will load all of the "24" puzzles
    task = get_task("game24")
    print(f"Loaded {len(task)} puzzles.\n")

    # 2. Pick one puzzle (here, the first)
    idx = 0
    puzzle = task.get_input(idx)  # e.g. "4 4 6 8"
    print(f"Puzzle #{idx}: {puzzle}\n")

    # 3. Fill each of the built-in templates
    std = standard_prompt.format(input=puzzle)
    cot = cot_prompt.format(input=puzzle)
    propose = propose_prompt.format(input=puzzle)
    val = value_prompt.format(input=puzzle)
    # for the last-step judge we need both input and a sample answer:
    sample_answer = "(4 + 8) * (6 - 4) = 24"
    val_last = value_last_step_prompt.format(input=puzzle, answer=sample_answer)



    # 4. Print them out
    print("=== Standard Prompt ===\n")
    print(std)
    print("\n=== Chain-of-Thought Prompt ===\n")
    print(cot)
    print("\n=== Propose Prompt ===\n")
    print(propose)
    print("\n=== Value Prompt ===\n")
    print(val)
    print("\n=== Value-Last-Step Prompt ===\n")
    print(val_last)
    # read the 24.csv data the same way as tot. 
    batch_prompts = [cot+"\nSteps:"]

    model_name = "gpt-4.1-mini"
    # model_name = "gpt-4o-mini"
    # model_name = "gpt-4.1-nano"
    with ThreadPoolExecutor(max_workers=20) as pool:
        api_outputs = list(pool.map(
            lambda p: client.responses.create(model=model_name, input=p).output_text,
            batch_prompts
        ))
    print("api_outputs", api_outputs)
    for generated_text in api_outputs:
        m = re.search(r'^(?:.*Answer:)\s*(.*)$', generated_text, re.DOTALL)
        if m:
            model_answer = m.groups()[-1].strip()
        print(model_answer)

    # model_answer = "((4 / 1) * 6) * 1 = 24"
    lhs = sympify(model_answer.split("=")[0])
    expr = sympify(model_answer.split("=")[0], evaluate=False)
    operand = [int(i) for i in expr.args]
    true_operand = [int(i) for i in puzzle.split(" ")]

    operand.sort()
    true_operand.sort()

    print("operand", operand)
    print("true_operand", true_operand)

    # sort the operand and true_operand

    print("sorted_operand", operand)
    print("sorted_true_operand", true_operand)
    print("check", operand == true_operand)

    # entire_expression = sympify(model_answer)
    print("lhs", lhs)
    print(lhs == 24)
    # print("entire_expression", entire_expression)



def evaluate_checkpoint(
    # checkpoint_path='meta-llama/Meta-Llama-3-8B-Instruct',
    # base_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    checkpoint_path="google/gemma-7b-it",
    base_model_id="google/gemma-7b-it",
    # judge_model_id="google/gemma-3-12b-it",
    # judge_model_id="google/gemma-7b-it",
    judge_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_name='tatsu-lab/alpaca',
    split="test",
    max_eval_samples=45,
    n=50,
    max_new_tokens=1000
):
    num_samples = 100
    task = get_task("game24")


    max_eval_samples = num_samples
    """
    Evaluate the model using batch inference. For each round, we send all questions
    together to vLLM, then update each prompt with the generated response as a weak demo.
    The process repeats for n rounds.
    """
    # Load dataset.
    if dataset_name == "openai/gsm8k":
        dataset = load_dataset(dataset_name, 'main', split=split)
    elif dataset_name == "tatsu-lab/alpaca":
        dataset = load_dataset(dataset_name, split='train')

    # Optionally reduce dataset size.
    if max_eval_samples < len(dataset):
        dataset = dataset.select(range(25, max_eval_samples))
    
    # Load one_shot prompt.
    # with open('length_prompt_attempt.txt', 'r') as f:
    #     one_shot_prompt = f.read()
    # with open('humor_prompt_attempt_task_two_reward.txt', 'r') as f:
    #     one_shot_prompt = f.read()

    with open('game24_3steps.txt', 'r') as f:
        one_shot_prompt = f.read()
    
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Initialize the vLLM instance.
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # judge_llm = LLM(model=judge_model_id)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # llm = LLM(model=base_model_id)
    # judge_llm = llm
    
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_new_tokens)
    
    # Prepare a list for all samples that meet the criteria.
    # Each entry stores the question, a history of weak demos, and outputs per round.
    samples = []
    # for sample in dataset:
    #     question = sample.get('instruction', "")
    #     question_input = sample.get('input', "")
    #     if len(question_input) != 0:
    #         continue
    #     samples.append({
    #         "question": question,
    #         "weak_demos": [],  # will hold dictionaries with keys: prompt, answer, reward
    #         "output": []       # will record output details per round
    #     })
    for idx in range(num_samples):
        question = task.get_input(900+idx)
        samples.append({
            "question": question,
            "weak_demos": [],  # will hold dictionaries with keys: prompt, answer, reward
            "output": []       # will record output details per round
        })
    
    num_samples = len(samples)
    print(f"Processing {num_samples} samples in {n} rounds...")
    
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
            # prompt = ""
            # prompt += f"**Task**: {task_prompt_cot}\n"
#             if round_idx == -1:
#                 # prompt += cot_prompt.format(input=sample['question']) + "\nSteps:"
#                 prompt += "<attempt>\n"
#                 # prompt += f"**Task**: {task_prompt_cot}"
#                 prompt += f"**Prompt**: Input: {sample['question']}\n"
#                 # prompt += f"**Format Reward**: 0.00\n"
#                 if not no_reward:
#                     prompt += f"**Reward**: 0.00\n"
                
#             else:
#                 if not rejection_sampling: 
#                     # Add previous weak demonstrations if any.
#                     for weak_demo in sample["weak_demos"][-num_weak_demo:]:
#                         prompt += "<attempt>\n"
#                         # prompt += f"**Task**: {task_prompt_cot}"
#                         prompt += f"**Prompt**: Input: {weak_demo['prompt']}\n"
#                         # prompt += f"**Format Reward**: {weak_demo['format_reward']}\n"
#                         # prompt += f"**Reward**: {weak_demo['gpt_eval_reward']}\n"
#                         # Steps: ??
                        
#                         ### steps reward here
                        
#                         if not no_reward:
                        
#                             try:
#                                 prompt +="**Response** Step" + weak_demo['answer'].split("Step")[1] 
#                                 prompt += f"<Reward: {weak_demo['gpt_eval_reward_list'][0]}> Step" + weak_demo['answer'].split("Step")[2] 
#                                 prompt += f"<Reward: {weak_demo['gpt_eval_reward_list'][1]}> Step" + weak_demo['answer'].split("Step")[3].split("**Answer**")[0] 
#                                 prompt += f"<Reward: {weak_demo['gpt_eval_reward_list'][2]}> **Answer**" + weak_demo['answer'].split("**Answer**")[1]
#                             except:
#                                 prompt += "" + weak_demo['answer'][:] + "\n"
#                         else:
#                             prompt += "" + weak_demo['answer'][:] + "\n"
#                         # prompt += f"**Format Reward**: {weak_demo['format_reward']}\n"
                        
#                         if not no_reward: 
#                             if zero_reward: 
#                                 prompt += f"**Reward**: {0.00}\n"
#                             else:
#                                 # prompt += f"**Reward**: {weak_demo['gpt_eval_reward']}\n"
                                
#                                 prompt += f"""**Reward**: Step 1 Reward: {weak_demo['gpt_eval_reward_list'][0]}, \n
#                                 Step 2 Reward: {weak_demo['gpt_eval_reward_list'][1]}, \n
#                                 Step 3 Reward: {weak_demo['gpt_eval_reward_list'][2]}, \n
#                                 Final Answer Reward: {weak_demo['gpt_eval_reward_list'][3]}. \n
#                                 \n"""
#                                 prompt += f"""**Return**: {weak_demo['gpt_eval_return']}. \n"""
                            
#                         prompt += "</attempt>"
                    
#                     prompt += "<instructions>\n"
#                     # may need to adjust the one_shot_prompt demonstrations once we start explore and exploit.
                    
#                     if ICRL: 
#                         if round_idx % 4 == 1:
#                             prompt += exploitation_instruction
#                         else:
#                             prompt += exploration_instruction        
#                         # if round_idx % 2 == 0:
#                         #     prompt += exploration_instruction
#                         # else:
#                         #     prompt += exploitation_instruction
#                     if exploration_only:
#                         prompt += exploration_instruction
#                     if exploitation_only:
#                         prompt += exploitation_instruction
#                     if no_reward_exploration: 
#                         prompt += no_reward_exploration_instruction
#                     if exploration_or_exploitation: 
#                         prompt += explore_or_exploit_instruction
                        
                        
#                     # prompt += " Only make one attempt, and put your answer in `<answer>**Response** Step1: ... (left: ...) Step2: ... (left: ...) Step3: ... (left: ...) **Answer**: <math operations of the 4 input numbers.></answer>` format. Whether the Answer is correct or incorrect, do not try again. \n"

            for weak_demo in sample["weak_demos"][-num_weak_demo:]:
                prompt += f"**Plan**: {weak_demo['plan']}\n"
            
            prompt += " Only make one attempt, and put your answer in `<answer>**Response** Step1: ... (left: ...) Step2: ... (left: ...) Step3: ... (left: ...) **Answer**: <math operations of the 4 input numbers, even if it does not equal 24></answer>` format. Whether the Answer is correct or incorrect, do not try again. \n"

            # prompt += "\n</instructions>\n"
            # else:
                # prompt += f"**Task**: {task_prompt_cot}"
            # prompt += f"**Prompt**: Input: {sample['question']}\nSteps:"
            # prompt += " Strictly following the Response format as demonstrated in the examples."
            # prompt += "<attempt>\n"
            # prompt += f"**Task**: {task_prompt_cot}"
            prompt += f"**Prompt**: Input: {sample['question']}\n"
            # prompt += f"**Format Reward**: 0.00\n"
            # prompt += f"**Accuracy Reward**: 0.00\n"
                
            # Append the new attempt with the current question.
            # prompt += "<attempt>\n"
            # prompt += f"**Task**: {task_prompt}"
            # prompt += f"**Prompt**: {sample['question']}\n"
            # prompt += "**Reward**: 10.00\n"
            batch_prompts.append(prompt)
        
        # Send all prompts together in one batch.
        # (Assuming vLLM accepts a list of prompts.)
        # vllm_outputs = llm.generate(batch_prompts, sampling_params)
        # model_name = "o4-mini"
        model_name = "gpt-4.1"
        # model_name = "gpt-4.1-mini"
        # model_name = "gpt-4o-mini"
        # model_name = "gpt-4.1-nano"
            
        with ThreadPoolExecutor(max_workers=20) as pool:
            api_outputs = list(pool.map(
                lambda p: client.responses.create(model=model_name, input=p).output_text,
                batch_prompts
            ))
        
        eval_prompt_list = []
        eval_prompt_full_answer_list = []
        
        # if round_idx != 0:
        
        for i, output_obj in enumerate(api_outputs):
            # Retrieve generated text.
            generated_text = output_obj
            ## Use regex to extract text up to </attempt>
            # pattern = r"(?s)^.*?</attempt>"
            # m = re.match(pattern, generated_text, flags=re.DOTALL)
            # if m:
            #     model_answer = m.group(0)
            # else:
            #     model_answer = ""

            # m = re.search(r'^(?:.*Answer:)\s*(.*)$', generated_text, re.DOTALL)

            pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

            # first or last, could be an important difference

            # print("before extraction", "-"*20)
            # print(generated_text)
            try:
                generated_text = pattern.findall(generated_text)[0]
                # generated_text = generated_text[8:-9]
                # generated_text = generated_text.split("<answer>")[1]
            except:
                generated_text = ""
            # print("after extraction", "+"*20)
            # print(generated_text)

            m = re.search(r'^(?:.*\*\*Answer\*\*):\s*(.*)$', generated_text, re.DOTALL)
            # pattern = r'^(?:.*\*\*Answer\*\*):\s*(.*)$'
            if m:
                model_answer = m.groups()[-1].strip()
            else:
                model_answer = ""


            print("model_answer", model_answer)


            if "\\\\(" in model_answer:
                print("found \\\\("*20)
                model_answer = model_answer.replace("\\\\(", "")
            
            ### full response for eval.
            # model_answer = generated_text
                
            eval_prompt_full_answer_list.append(generated_text)
            eval_prompt_list.append(model_answer)
        
        eval_result_list = []
        format_reward_list = []

        gpt_eval_prompt_list = []

        for index, model_answer in enumerate(eval_prompt_list):
            
            
            

            # create eval prompt
            eval_prompt = f"Evaluate if the given solution for the 24 game is correct for this input: {samples[index]['question']}. Solution: {model_answer}. Return -10 if correct. If incorrect, if the solution is invalid such as using the same number twice or using numbers other than the 4 numbers in the input, return 10; if the solution is valid, but the answer is not 24, count the number of edits of the operations required to make it into a correct solution of 24. The higher the number of edits, the further away the solution is from the correct solution. Return the number of edits. Put your response in the following format: **Answer**: <integer number of edits>."
            eval_prompt = f"Evaluate if the given solution for the 24 game is correct for this input: {samples[index]['question']}. Solution: {model_answer}. Return -10 if correct. If incorrect, count the number of edits of the operations required to make it into a correct solution of 24. The higher the number of edits, the further away the solution is from the correct solution. Return the number of edits required to turn this into a correct solution. Put your response in the following format: **Answer**: <integer number of edits>."
            eval_prompt = f"Evaluate if the given solution for the 24 game is correct for this input: {samples[index]['question']}. Solution: {model_answer}. Return -10 if correct. If incorrect, evaluate the likeliness of the numbers in (left:...) for the first step and the second step to achieve 24. Rate this likeliness from 1 to 5 for each step. Sum the two numbers and return the result in the following format: **Answer**: <integer number of the sum of likeliness>."
            eval_prompt = f"Evaluate if the given solution for the 24 game is correct for this input: {samples[index]['question']}. Solution: {model_answer}. Return -10 if correct. If incorrect, evaluate if the numbers in (left:...) can reach 24 (sure/likely/impossible). The likeliness score are sure: 2, likely: 1, impossible: 0. Sum the numbers for each step and return the result in the following format: **Answer**: <integer number of the sum of likeliness>."
            
            eval_prompt = f"""Please evaluate whether the proposed solution to the 24‐game is correct for this input:
Question: {samples[index]['question']}
Solution: {model_answer}

– If the solution is correct, return **Answer**: -10.  
– If it is incorrect, look at the numbers in each “left: …” step and judge how likely they are to reach 24:
   • sure → 3  
   • likely → 2  
   • impossible → 1  

Add up those step‐scores and return **Answer**: <sum of likeliness scores>."""
            
            eval_prompt = f"""Please evaluate whether the proposed solution to the 24‐game is correct for this input:
Question: {samples[index]['question']}
Solution: {eval_prompt_full_answer_list[index]}

– If the solution is correct, return **Answer**: 10.  
– If it is incorrect, look at the numbers in each “left: …” step and judge how likely they are to reach 24:
   • sure → 3  
   • likely → 2  
   • impossible → 1  
- If the step is invalid, such as using numbers outside of the 4 input numbers or not using all numbers at the last step, the score is 0. 

Add up those step‐scores and return **Answer**: <sum of likeliness scores>."""
            
            
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
                
            # and think about how likely they are to reach 24
            # If the step is invalid, such as using numbers outside of the 4 input numbers or wrong calculations, the score is 0.
            
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
        
        # eval_model_name = "gpt-4.1-mini"
        eval_model_name = "gpt-4.1-mini"
        
        # with ThreadPoolExecutor(max_workers=20) as pool:
        #     eval_api_outputs = list(pool.map(
        #         lambda p: client.responses.create(model=eval_model_name, input=p).output_text,
        #         gpt_eval_prompt_list
        #     ))

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
                # generated_text = generated_text[8:-9]
                # generated_text = generated_text.split("<answer>")[1]
            except:
                generated_text = ""

            
            # Use regex to extract text up to </attempt>
            # pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
            # try:
            #     model_answer = pattern.findall(generated_text)[0]
            # except:
            #     model_answer = ""
            # m = re.search(r'^(?:.*Answer:)\s*(.*)$', generated_text, re.DOTALL)
            m = re.search(r'^(?:.*\*\*Answer\*\*):\s*(.*)$', generated_text, re.DOTALL)
            if m:
                model_answer = m.groups()[-1].strip()
            else:
                model_answer = ""

            

            
                
            # if rejection_sampling:
            #     model_answer = generated_text



#             eval_prompt = f"You are a seasoned comedy critic. Read the TEXT below and rate its overall humor on a 1‑to‑10 scale, where 1 means not funny at all” and 10 means hilarious, laugh‑out‑loud. Return your answer in exactly this format: Humor rating: <integer 1‑10> TEXT:<<<{model_answer}>>>"
#             sampling_params_eval = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128)

#             eval_result = llm.generate([eval_prompt], sampling_params_eval)

            # if round_idx != 0:

                # eval_result = eval_result_list[i]

                # _RATING_RE = re.compile(r"Humor rating:\s*(10|[1-9])\b")


                # def get_humor_rating(text):
                #     """Return the humor rating or None if the pattern isn't present."""
                #     m = _RATING_RE.search(text)
                #     return int(m.group(1)) if m else None
                # if api_eval:
                #     reward_str = get_humor_rating(eval_result)
                # else:
                #     reward_str = get_humor_rating(eval_result.outputs[0].text)
                # try:
                #     reward_value = int(reward_str)
                # except:
                #     reward_value = 0
                # if api_eval: 
                #     print("[]"*20, "\n eval_result", eval_result)
                # else:
                #     print("[]"*20, "\n eval_result", eval_result.outputs[0].text)
                # print("--"*10) 
                
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
                
                
            # for j in range(len(gpt_eval_reward_list) - 2):
            #     if gpt_eval_reward_list[j] == 0:
            #         for k in range(j+1, len(gpt_eval_reward_list)):
            #             gpt_eval_reward_list[k] = 0
            #             gpt_eval_reward_str_list[k] = "0"
                
                
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
            
        ## Generating Reflexion    
        batch_prompts = []
        for sample in samples:
            # prompt = one_shot_prompt
            # Add previous weak demonstrations if any.
            # for weak_demo in sample["weak_demos"]:
            last_weak_demo = sample["weak_demos"][-1]
            prompt = ""
            prompt += "<attempt>\n"
            prompt += f"**Question**: {last_weak_demo['prompt']}\n"
            # 4399: at first, we don't have weak demo at all. 
            # prompt += f"**Plan**: {weak_demo['reflexion']}\n"
            prompt += f"**Return**: {last_weak_demo['gpt_eval_return']}\n"
            prompt += last_weak_demo['answer'] + "\n"
            # Append the new attempt with the current question.
            prompt += "<attempt>\n"
            
            # here, instead of asking a question, ask for a reflexion
            
            prompt += "Instruction: You will be given the history of a past experience in which you encountered a task that required you to provide a response to a prompt aiming to maximize a reward, and you attempted a response. You were unsuccessful in providing an answer that achieved the specified desirable numerical reward of 10.0. Instead of recounting the details of the task itself, focus on analyzing the approach you took and the specific actions or steps you attempted. Based on this reflection, devise a concise, revised plan of action that acknowledges your error and details the exact measures or methods you should have employed. For example, if you attempted steps A and B but overlooked step C, construct a plan that explicitly incorporates step C into your approach. This self-reflection and plan will be essential for when you reattempt the task. Present your plan immediately following the keyword “Plan:”.\n"
            prompt += "Plan:"
            
            # print('+ + '*20)
            # print(prompt)
            batch_prompts.append(prompt)
        
        # max_token_reflexion = 256
        # sampling_params_reflexion = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_token_reflexion)
            
        # vllm_outputs = llm.generate(batch_prompts, sampling_params_reflexion)
        with ThreadPoolExecutor(max_workers=20) as pool:
            api_outputs = list(pool.map(
                lambda p: client.responses.create(model=model_name, input=p).output_text,
                batch_prompts
            ))
        
        for i, output_obj in enumerate(api_outputs):
            # Retrieve generated text.
            generated_text = output_obj
            
            reflexion = generated_text
            # Use regex to extract text up to </attempt>
            # pattern = r"(?s)^.*?</attempt>"
            # m = re.match(pattern, generated_text, flags=re.DOTALL)
            # if m:
            #     model_answer = m.group(0)
            # else:
            #     model_answer = ""
            
            # Compute reward.
            
            # # reward_value = len(model_answer) / num_char
            # reward_value = -1
            # reward_str = f"{reward_value:.2f}"
            
            
            sample = samples[i]
            
            
            last_weak_demo = sample["weak_demos"][-1]
            
            last_weak_demo['plan'] = reflexion
        
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
                this_time_change += "100_reflexion_ICRL"
                
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


