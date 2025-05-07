import pickle
import os
import re
import tqdm
import json
import time
import sys
import numpy as np
import torch
import argparse
from utils import *
from prm_eval import *

from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

from concurrent.futures import ThreadPoolExecutor

# from vllm.utils import DeviceCUDAList
from openai import OpenAI
# Number of characters used to compute the reward.
num_char = 200


api_eval = True


rejection_sampling = 0
ICRL = 1
exploitation_only = 0
exploration_only_no_reward = 0
exploration_and_exploitation = 0

no_reward = 0


exploration_or_exploitation = 0
num_weak_demos = 3000



sort_by_reward = 0

# client = OpenAI(api_key="sk-YxXAlZKGBu7-E2o5Vb3ARpIkVThWK_vAXSLE6WVzVrT3BlbkFJkMjHoPexFKJBS_fwHmGSmwjjn-ZksUWq3njQV5u1oA")

client = OpenAI(api_key="sk-C8z62BDhmo4EW1bqOn2TTmdFR29ocUeZXLExkdmGS1T3BlbkFJQcA3zOug-aNTm98KC0Wjsv549b3OgxEGn9TKJknXMA")


exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. Provide a response that is completely different from any previous attempts demonstrated in the context, and wrap it in `<answer>…</answer>`."

exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> entries. Each entry contains: •A candidate Response •Its numerical Reward  Your task: 1. Parse all <attempt> blocks and identify the attempts with the top reward scores. 2. Among those top‐scoring attempts, select the ones that are most distinct from each other in style or angle. 3. Create a single new mega response that fuses the strongest elements from each distinct, high‐scoring attempt within 200 words. 4. Return only this new, combined mega response wrapped in an <answer>…</answer> tag."


exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. Provide a response that is different from any previous attempts demonstrated in the context, and wrap it in `<answer>…</answer>`."

exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. Provide a response that is different from previous attempts demonstrated in the context, and wrap it in `<answer>…</answer>`."

exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> entries. Each entry contains a candidate Response and its reward. Your task: make the best educated guess based on the high reward attempts to produce response that can achieve a higher reward, while making sure it correctly follows the task instruction, and put it in `<answer>…</answer>` format."

# exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> entries. Each entry contains: •A candidate Response •Its numerical Reward  Your task: 1. Parse all <attempt> blocks and identify the attempts with the top reward scores. 2. Among those top‐scoring attempts, select the ones that are most distinct from each other in style or angle. 3. Create a single new mega response that fuses the strongest elements from each distinct, high‐scoring attempt. 4. Return only this new, combined mega response wrapped in an <answer>…</answer> tag."

explore_or_exploit_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. You have two options, exploration or exploitation. For exploration, provide a response that is different from previous attempts demonstrated in the context, and wrap it in `<answer>…</answer>`. For exploitation, make the best educated guess based on the high reward attempts to produce response that can achieve a higher reward, while making sure it correctly follows the task instruction, and put it in `<answer>…</answer>` format. Pick one option to follow."

explore_and_exploit_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. Write your new text that is different from the old ones and has a reward as high as possible, and put it in `<answer>…</answer>` format."
sys.path.append('/sfs/weka/scratch/ks8vf/ICL/tree-of-thought-llm/src')


from tot.tasks import get_task
from tot.prompts.text import (
    standard_prompt,
    cot_prompt,
    vote_prompt,
    compare_prompt,
)
def main():
    # ---------------------------------------------------------------------
    # 1.  Initialise the Text task (100 random creative-writing prompts)
    # ---------------------------------------------------------------------
    task = get_task("text")
    print(f"Loaded {len(task)} creative-writing instructions.\n")

    # ---------------------------------------------------------------------
    # 2.  Pick one instruction
    # ---------------------------------------------------------------------
    idx = 1
    instruction = task.get_input(idx).strip()
    print(f"Instruction #{idx}:\n{instruction}\n")

    # ---------------------------------------------------------------------
    # 3.  Fill the prompt templates
    # ---------------------------------------------------------------------
    std_prompt = standard_prompt.format(input=instruction)
    cot_prompt_filled = cot_prompt.format(input=instruction)

    # For a first generation we usually need the `Passage:` token that the
    # scoring code looks for.  Appending it here nudges the model to continue
    # with the actual passage.
    generation_starter = cot_prompt_filled

    print("=== Standard Prompt ===\n")
    print(std_prompt)
    print("\n=== CoT Prompt (+Passage header) ===\n")
    print(generation_starter)

    # ---------------------------------------------------------------------
    # 4.  Ask a model to continue the prompt
    # ---------------------------------------------------------------------
    # client = OpenAI()
    model_name = "gpt-4.1-mini"     # any chat-completion-style model works
        
    with ThreadPoolExecutor(max_workers=12) as pool:
        api_outputs = list(pool.map(
            lambda p: client.responses.create(model=model_name, input=p).output_text,
            [generation_starter]
        ))

    model_answer = api_outputs[0].strip()
    print("\n=== Raw model output ===\n")
    print(model_answer)
    
    score_prompt = '''Analyze the following passage, then at the last line conclude "Thus the coherency score is {s}", where s is an integer from 1 to 10.
'''
    
    score_prompt += model_answer
    
    eval_model_name = "gpt-4.1-mini"
    with ThreadPoolExecutor(max_workers=12) as pool:
        api_outputs = list(pool.map(
            lambda p: client.responses.create(model=eval_model_name, input=p).output_text,
            [score_prompt]
        ))

    model_answer = api_outputs[0].strip()
    print("\n=== Raw eval model output ===\n")
    print(model_answer)

    # ---------------------------------------------------------------------
    # 5.  Score the passage with the built-in evaluator
    # ---------------------------------------------------------------------
    # info = task.test_output(idx, model_answer)
    # print("\n=== Coherence scores ===")
    # print(f"Five reviewer scores: {info['rs']}")
    # print(f"Average coherence   : {info['r']:.1f}/10\n")


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
    with open('humor_prompt_attempt_task_two_reward.txt', 'r') as f:
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
    
    task = get_task("text")
    
    samples = []
    for idx in range(num_samples):
        # question = sample.get('instruction', "")
        # question_input = sample.get('input', "")
        instruction = task.get_input(idx)
        
        cot_prompt_filled = cot_prompt.format(input=instruction)

        # For a first generation we usually need the `Passage:` token that the
        # scoring code looks for.  Appending it here nudges the model to continue
        # with the actual passage.
        question = cot_prompt_filled
        # if len(question_input) != 0:
        #     continue
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
    
    
    # task_prompt = ""
    
    # Run n rounds.
    for round_idx in range(n):
        print(f"Round {round_idx+1}/{n}...")
        # Build a prompt for each sample.
        batch_prompts = []
        for sample in samples:
            
            
            # prompt = one_shot_prompt
            prompt = ""
            
            # if not rejection_sampling: 
            #     # Add previous weak demonstrations if any.
            #     if sort_by_reward: 
            #         weak_demos = sorted(sample["weak_demos"], key=lambda d: float(d["reward"]))
            #     else:
            #         weak_demos = sample["weak_demos"]
            #     for weak_demo in weak_demos[-num_weak_demo:]:
            #         prompt += "<attempt>\n"
            #         # prompt += f"**Task**: {task_prompt}"
            #         prompt += f"**Prompt**: {weak_demo['prompt']}\n"
            #         if not no_reward: 
            #             prompt += f"**Reward**: {weak_demo['reward']}\n"
            #         prompt += weak_demo['answer'][:-28] + "\n"
            #         if not no_reward: 
            #             prompt += f"**Reward**: {weak_demo['reward']}\n"
            #         prompt += "</attempt>"
            #     if ICRL: 
            #         if round_idx % 2 == 0:
            #             prompt += exploration_instruction
            #         else:
            #             prompt += exploitation_instruction
            #     if exploration_only_no_reward:
            #         prompt += exploration_instruction
            #     if exploration_and_exploitation:
            #         prompt += explore_and_exploit_instruction
            #     if exploitation_only:
            #         prompt += exploitation_instruction
            #     if exploration_or_exploitation:
            #         prompt += explore_or_exploit_instruction
            #     # prompt += explore_or_exploit_instruction
            # # else:
            # #     if round_idx == 0:
            # #         prompt += f"**Task**: {task_prompt0}"
            # #     else:
            # #         prompt += f"**Task**: {task_prompt}"
            
            # prompt += task_prompt
            for weak_demo in sample["weak_demos"][-num_weak_demos:]:
                prompt += f"**Plan**: {weak_demo['plan']}\n"
                
            prompt += "Provide the response in `<answer>…</answer>` format."
                
            prompt += f"**Prompt**: {sample['question']}\n"
            # Append the new attempt with the current question.
            # prompt += "<attempt>\n"
            # prompt += f"**Task**: {task_prompt}"
            # prompt += f"**Prompt**: {sample['question']}\n"
            # prompt += "**Reward**: 10.00\n"
            batch_prompts.append(prompt)
        
        # Send all prompts together in one batch.
        # (Assuming vLLM accepts a list of prompts.)
        # vllm_outputs = llm.generate(batch_prompts, sampling_params)
        # model_name = "gpt-4.1-mini"
        model_name = "gpt-4.1"
            
        with ThreadPoolExecutor(max_workers=12) as pool:
            api_outputs = list(pool.map(
                lambda p: client.responses.create(model=model_name, input=p).output_text,
                batch_prompts
            ))
        
        eval_prompt_list = []
        
        if round_idx != 0:
        
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
                    
                pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
                try:
                    model_answer = pattern.findall(generated_text)[0]
                except:
                    model_answer = ""
                    
                if rejection_sampling:
                    model_answer = generated_text
                    
                    

                    
                ### only keep the passage for eval
                try: 
                    model_answer = model_answer.split("Passage:")[1]
                except:
                    model_answer = model_answer

                eval_prompt = f"Instruction: You are a seasoned comedy critic. Read the Base Answer and the TEXT below and rate the TEXT's overall humor on a 1‑to‑10 scale, where 1 means way less funny compared to the Base Answer, 5 means TEXT is as funny as the Base Answer and 10 means way more hilarious, laugh‑out‑loud than the Base Answer.  Base Answer: {samples[i]['weak_demos'][0]['answer']} TEXT:<<<{model_answer}>>> Return your answer in exactly this format: Humor rating: <integer 1‑10>.\n Response:"
                
                eval_prompt = f'''Analyze the following passage, then rate the passsage's overall coherency on a 1-to-10 scale. passage:<<<{model_answer}>>> Return your answer in exactly this format: Coherency score: <integer 1‑10>.\n 
'''
                
                eval_prompt = f"Instruction: You are a seasoned text coherence evaluator. Read the TEXT below and rate the TEXT’s overall coherency on a 1-to-10 scale, where 1 means much more disjointed and hard to follow than the Base Answer, 5 means as coherent as the Base Answer, and 10 means way more flawlessly structured and crystal-clear than the Base Answer. Base Answer: {samples[i]['weak_demos'][0]['answer']} TEXT:<<<{model_answer}>>> Return your answer in exactly this format: Coherency score: <integer 1-10>.\nResponse:"
                
                
                base_answer = samples[i]['weak_demos'][0]['answer']
                
                
                # print("0 = 0 "*10)
                # print("base_answer", base_answer)
                # print("0 = 0 "*10)
                # print("model_answer", model_answer)
                try:
                    base_answer = base_answer.split("Passage:")[1]
                except:
                    base_answer = base_answer
                    
                    
                base_answer = """At dawn, golden light slips past pale curtains, rousing the world in quiet celebration. A lone robin greets the morning with a clear, cheerful trill, its song drifting across dew-laden grass. A gentle breeze stirs the leaves, carrying the fresh, earthy scent of new growth. Nearby, rooftops and empty streets lie poised between night’s calm and the city’s stirring pulse, promising simple comforts like a warm cup of coffee. In this tranquil pause, one senses life’s renewal and the gentle invitation to greet the day with hope and gratitude."""
                
                eval_prompt = f"Instruction: You are a seasoned text coherence evaluator. Read the TEXT below and rate the TEXT’s overall coherency on a 1-to-10 scale, where 1 means less coherent than the Base Answer, 5 means as coherent as the Base Answer, and 10 means way more coherent than the Base Answer. Be a strict and conservative evaluator and only gave a high score when the TEXT is truly better than the Base Answer. Base Answer: <<<{base_answer}>>> TEXT:<<<{model_answer}>>> Return your answer in exactly this format: Coherency score: <integer 1-10>. \nResponse:"
                
                # eval_prompt = f"Instruction: You are a seasoned text originality evaluator. Read the TEXT below and rate the TEXT’s overall originality on a 1-to-10 scale, where 1 means less original than the Base Answer, 5 means as original as the Base Answer, and 10 means way more original than the Base Answer. Base Answer: <<<{base_answer}>>> TEXT:<<<{model_answer}>>> Return your answer in exactly this format: Originality score: <integer 1-10>. \nResponse:"
                

                # eval_prompt = f"Instruction: You are a seasoned text coherence evaluator. Read the TEXT below and rate the TEXT’s overall coherency on a 1-to-10 scale, where 1 means much more disjointed and hard to follow, 5 means as coherent as any response generated by ChatGPT, and 10 means extremely flawlessly structured and crystal-clear. Be a strict and conservative evaluator and make sure to provide a score lower than 5 for all TEXT that has a below average coherence and only gave a high score when the TEXT is extremely coherent. TEXT:<<<{model_answer}>>> Return your answer in exactly this format: Coherency score: <integer 1-10>.\nResponse:"
     
                

                # eval_prompt = (
                #     f"Instruction: You are a seasoned comedy critic. Read the Base Answer and the TEXT below and "
                #     "rate the TEXT's overall humor on a 1‑to‑10 scale, where 1 means about as funny as the Base Answer, "
                #     "2 means roughly twice as funny, …, and 10 means about ten times funnier—truly laugh‑out‑loud compared to the Base Answer. "
                #     f"Base Answer: {samples[i]['weak_demos'][0]['answer']} "
                #     f"TEXT:<<<{model_answer}>>> "
                #     "Return your answer in exactly this format: Humor rating: <integer 1‑10>.\n Response:"
                # )
                
                  


                # eval_prompt = (
                #     f"Instruction: Please act as an impartial judge and evaluate the quality of the responses provided by an AI assistant. Read the **Base Answer** and the **TEXT** below, "
                #     "Assess the humor of the TEXT compared to the Base Answer on a 1‑to‑10 scale."
                #     "Begin with a brief, objective explanation of your reasoning."
                #     "After the explanation, you must rate the response outputting in this format (no extra text): Humor rating: <integer 1‑10>.\n"
                #     "Response:"
                # )
                # eval_prompt = (
                #     f"Instruction: Please act as an impartial judge and evaluate the quality of the responses provided by an AI assistant. Read the **Base Answer** and the **TEXT** below, "
                #     "Assess the humor of the TEXT compared to the Base Answer on a 1‑to‑10 scale."
                #     # "Begin with a brief, objective explanation of your reasoning."
                #     "You must rate the response outputting in this format (no extra text): Humor rating: <integer 1‑10>.\n"
                #     "Response:"
                # )
                
                # eval_prompt = (
                #     f"Instruction: Please act as an impartial judge and evaluate the quality of the responses provided by an AI assistant. Read the **Base Answer** and the **TEXT** below, "
                #     "then rate how much funnier the TEXT is than the Base Answer on a 1‑to‑10 scale.\n\n"
                #     "Imagine readig this paragraph in front of an audiance. Rubric (relative to the Base Answer):\n"
                #     "  1 — About as funny; virtually no extra laughs.\n"
                #     "  2 — Roughly twice as funny; mild smirk.\n"
                #     "  3 — Three × funnier; a few soft chuckles.\n"
                #     "  4 — Four ×; steady chuckling.\n"
                #     "  5 — Five ×; consistent laughs across the room.\n"
                #     "  6 — Six ×; strong humor, several standout lines.\n"
                #     "  7 — Seven ×; loud laugh‑out‑loud moments.\n"
                #     "  8 — Eight ×; sustained laughter, tear‑wipes.\n"
                #     "  9 — Nine ×; roaring laughter, crowd struggles to stop.\n"
                #     "  10 — Ten ×; uncontrollable, prolonged laughter; comedy gold.\n\n"
                #     f"Base Answer: {samples[i]['weak_demos'][0]['answer']} "
                #     f"TEXT:<<<{model_answer}>>> "
                #     "Return your answer in exactly this format: Humor rating: <integer 1‑10>.\n"
                #     "Response:"
                # )

                eval_prompt_list.append(eval_prompt)


            sampling_params_eval = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

           
            
            if api_eval:
            
                # model_name = "gpt-4o-mini"
                # model_name = "gpt-4.1-mini"
                model_name = "gpt-4.1"
                with ThreadPoolExecutor(max_workers=12) as pool:
                    eval_result_list = list(pool.map(
                        lambda p: client.responses.create(model=model_name, input=p).output_text,
                        eval_prompt_list
                    ))
            else:
                eval_result_list = judge_llm.generate(eval_prompt_list, sampling_params_eval)

            # _RATING_RE = re.compile(r"Humor rating:\s*(10|[1-9])\b")
            _RATING_RE = re.compile(r"Coherency score:\s*(10|[1-9])\b")
            # _RATING_RE = re.compile(r"Originality score:\s*(10|[1-9])\b")

        def get_humor_rating(text):
            """Return the humor rating or None if the pattern isn't present."""
            m = _RATING_RE.search(text)
            return int(m.group(1)) if m else None



        # reward_str = get_humor_rating(eval_result[0].outputs[0].text)
        # try:
        #     reward_value = int(reward_str)
        # except:
        #     reward_value = 0

        # Process batch responses.
        for i, output_obj in enumerate(api_outputs):
            # Retrieve generated text.
            generated_text = output_obj
            # Use regex to extract text up to </attempt>
            pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
            try:
                model_answer = pattern.findall(generated_text)[0]
            except:
                model_answer = ""
                
                
            if rejection_sampling:
                model_answer = generated_text
                
            # model_answer = model_answer.split("Passage:")[1]



#             eval_prompt = f"You are a seasoned comedy critic. Read the TEXT below and rate its overall humor on a 1‑to‑10 scale, where 1 means not funny at all” and 10 means hilarious, laugh‑out‑loud. Return your answer in exactly this format: Humor rating: <integer 1‑10> TEXT:<<<{model_answer}>>>"
#             sampling_params_eval = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128)

#             eval_result = llm.generate([eval_prompt], sampling_params_eval)

            if round_idx != 0:

                eval_result = eval_result_list[i]

                # _RATING_RE = re.compile(r"Humor rating:\s*(10|[1-9])\b")
                RATING_RE = re.compile(r"Coherency score:\s*(10|[1-9])\b")
                # RATING_RE = re.compile(r"Originality score:\s*(10|[1-9])\b")


                def get_humor_rating(text):
                    """Return the humor rating or None if the pattern isn't present."""
                    m = _RATING_RE.search(text)
                    return int(m.group(1)) if m else None
                if api_eval:
                    reward_str = get_humor_rating(eval_result)
                else:
                    reward_str = get_humor_rating(eval_result.outputs[0].text)
                try:
                    reward_value = int(reward_str)
                except:
                    reward_value = 0
                if api_eval: 
                    print("[]"*20, "\n eval_result", eval_result)
                else:
                    print("[]"*20, "\n eval_result", eval_result.outputs[0].text)
                print("--"*10)
                print("reward_str", reward_str)
                print('reward_value', reward_value)
            else:

                reward_str = str(1.00)
                reward_value = 1.00




            # # Compute reward.
            # reward_value = len(model_answer) / num_char
            # reward_str = f"{reward_value:.2f}"

            if round_idx == 0:
                eval_result = ""
                eval_prompt_i = ""
            else:
                eval_prompt_i = eval_prompt_list[i]
            # Create a weak demo dictionary.
            weak_demo = {
                "prompt": samples[i]["question"],
                "answer": model_answer,
                "reward": reward_str
            }
            # Append to the sample's weak demo history.
            samples[i]["weak_demos"].append(weak_demo)
            # Record the round output.
            samples[i]["output"].append({
                "round": round_idx,
                "prompt": batch_prompts[i],
                "answer": model_answer,
                "generated_text": generated_text,
                "reward": reward_value,
                "eval_generated_text": eval_result,
                "eval_prompt": eval_prompt_i
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
            prompt += f"**Reward**: {last_weak_demo['reward']}\n"
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
        with ThreadPoolExecutor(max_workers=12) as pool:
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

            for sample in samples:
                # Get rewards from each round.
                round_rewards = [entry["reward"] for entry in sample["output"]]
                avg_reward_list.append(np.mean(round_rewards))
                last_reward_list.append(round_rewards[-1] if round_rewards else 0)
                gen_list.append(sample["output"][-1]["generated_text"] if sample["output"] else "")
                output_list.append(sample["output"])

            # Save the results to files.
            task = 'creative_writing_api'

            
            # this_time_change = "ICRL_"
            this_time_change = ""
            if rejection_sampling:
                this_time_change += "best_of_n_"
            else:
                this_time_change += "reflexion_"
            this_time_change += "100_gpt_4.1_same_base"

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