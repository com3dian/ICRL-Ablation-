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

from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

num_char = 200


api_eval = True


rejection_sampling = 0
ICRL = 1








num_weak_demos = 3000

zero_shot = 0
CoT = 1


sort_by_reward = 0



client = OpenAI(api_key="Your_API_Key")



sys.path.append('/tree-of-thought-llm/src')


from tot.tasks import get_task
from tot.prompts.text import (
    standard_prompt,
    cot_prompt,
    vote_prompt,
    compare_prompt,
)


def evaluate_checkpoint(
    checkpoint_path="creative_writing",
    base_model_id="GPT-4.1",
    max_eval_samples=45,
    n=1,
    max_new_tokens=1000
):
    num_samples = 100
    
    max_eval_samples = num_samples
    
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_new_tokens)
    
    task = get_task("text")
    
    samples = []
    for idx in range(num_samples):
        instruction = task.get_input(idx)
        
        cot_prompt_filled = cot_prompt.format(input=instruction)

        question = cot_prompt_filled
        samples.append({
            "question": question,
            "weak_demos": [],  # will hold dictionaries with keys: prompt, answer, reward
            "output": []       # will record output details per round
        })
    
    num_samples = len(samples)
    print(f"Processing {num_samples} samples in {n} rounds...")

    
    # Run n rounds.
    for round_idx in range(n):
        print(f"Round {round_idx+1}/{n}...")
        # Build a prompt for each sample.
        batch_prompts = []
        for sample in samples:
            prompt = ""
            prompt += f"**Prompt**: {sample['question']}"
            if CoT:
                prompt += "Let's first think step by step, and place your chain of thoughts between <think> and </think> tags. Then put your answer containing the Plan and the Passage in `<answer>...</answer>` tags."
            if zero_shot:
                prompt += "Put your answer containing the Plan and the Passage in `<answer>...</answer>` tags. Make sure the Passage is as coherent as possible."
            batch_prompts.append(prompt)
        

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
                
                
                base_answer = samples[i]['weak_demos'][0]['answer']
                
                try:
                    base_answer = base_answer.split("Passage:")[1]
                except:
                    base_answer = base_answer
                    
                    
                base_answer = """At dawn, golden light slips past pale curtains, rousing the world in quiet celebration. A lone robin greets the morning with a clear, cheerful trill, its song drifting across dew-laden grass. A gentle breeze stirs the leaves, carrying the fresh, earthy scent of new growth. Nearby, rooftops and empty streets lie poised between night’s calm and the city’s stirring pulse, promising simple comforts like a warm cup of coffee. In this tranquil pause, one senses life’s renewal and the gentle invitation to greet the day with hope and gratitude."""
                
                eval_prompt = f"Instruction: You are a seasoned text coherence evaluator. Read the TEXT below and rate the TEXT’s overall coherency on a 1-to-10 scale, where 1 means less coherent than the Base Answer, 5 means as coherent as the Base Answer, and 10 means way more coherent than the Base Answer. Be a strict and conservative evaluator and only gave a high score when the TEXT is truly better than the Base Answer. Base Answer: <<<{base_answer}>>> TEXT:<<<{model_answer}>>> Return your answer in exactly this format: Coherency score: <integer 1-10>. \nResponse:"
                

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
            
            batch_prompts.append(prompt)
        

            
        # vllm_outputs = llm.generate(batch_prompts, sampling_params_reflexion)
        model_name = "gpt-4.1-nano"
        with ThreadPoolExecutor(max_workers=12) as pool:
            api_outputs = list(pool.map(
                lambda p: client.responses.create(model=model_name, input=p).output_text,
                batch_prompts
            ))
        
        for i, output_obj in enumerate(api_outputs):
            # Retrieve generated text.
            generated_text = output_obj
            
            reflexion = generated_text
            
            
            sample = samples[i]
            
            
            last_weak_demo = sample["weak_demos"][-1]
            
            last_weak_demo['plan'] = reflexion
        
        # Optionally, save intermediate results after each round.
        # For example, you could pickle the samples list:
        with open("intermediate_round_zero_shot.pkl", "wb") as f:
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

            this_time_change = ""
            if rejection_sampling:
                this_time_change += "best_of_n_"
            else:
                this_time_change += "zero_shot"
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