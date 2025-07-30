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


api_eval = False  # Set to False to use vLLM instead of OpenAI
openrouter_eval = True  # Set to True to use OpenRouter API


rejection_sampling = 0


exploration_or_exploitation = 0
num_weak_demos = 3000



sort_by_reward = 0



# Initialize models based on selected backend
if not api_eval and not openrouter_eval:
    # vLLM initialization
    llm = LLM(model="Qwen/Qwen3-32B", 
              tensor_parallel_size=2,  # Adjust based on GPU count
              gpu_memory_utilization=0.95)
elif api_eval:
    # OpenAI API initialization
    client = OpenAI(api_key="Your_API_Key")
elif openrouter_eval:
    # OpenRouter configuration (exactly following math_bench.py)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
    OPENROUTER_MODEL = "meta-llama/llama-4-maverick"
    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


# Helper function for OpenRouter API calls (exactly following math_bench.py pattern)
def openrouter_generate(prompt, temperature=0.6, max_tokens=1000):
    """Make a request to OpenRouter API - following math_bench.py"""
    try:
        output = openrouter_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,  # math_bench.py uses max_completion_tokens
        )
        return output.choices[0].message.content
    except Exception as e:
        print(f"OpenRouter API error: {e}")
        return ""


# Load prompts directly to avoid import issues
cot_prompt = '''
Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}

Make a plan then write. Your output should be of the following format:

Plan:
Your plan here.

Passage:
Your passage here.
'''

# Simple task implementation that loads the actual data
class TextTask:
    def __init__(self):
        # Load the actual data from tot
        data_path = '/sfs/weka/scratch/ks8vf/tree-of-thought-llm/src/tot/data/text/data_100_random_text.txt'
        self.data = open(data_path).readlines()
    
    def __len__(self):
        return len(self.data)
    
    def get_input(self, idx):
        return self.data[idx % len(self.data)].strip()

def get_task(name):
    if name == "text":
        return TextTask()
    raise ValueError(f"Unknown task: {name}")


def evaluate_checkpoint(
    checkpoint_path="creative_writing",
    base_model_id=OPENROUTER_MODEL,
    max_eval_samples=45,
    n=51,
    max_new_tokens=1000
):
    num_samples = 100
    
    max_eval_samples = num_samples
    
    # Load tokenizer only if not using OpenRouter
    if not openrouter_eval and not api_eval:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_new_tokens)

    
    task = get_task("text")
    
    samples = []
    for idx in range(num_samples):
        instruction = task.get_input(idx)
        
        cot_prompt_filled = cot_prompt.format(input=instruction)

        # For a first generation we usually need the `Passage:` token that the
        # scoring code looks for.  Appending it here nudges the model to continue
        # with the actual passage.
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

            for weak_demo in sample["weak_demos"][-num_weak_demos:]:
                prompt += f"**Plan**: {weak_demo['plan']}\n"
                
            # prompt += "Provide the response in `<answer>…</answer>` format."
            prompt += "Instruction: provide your Plan and Passage in `<answer>Plan:...Passage:...</answer>` format."    
            prompt += f"**Prompt**: {sample['question']}\n"
            

            batch_prompts.append(prompt)
        
        if api_eval:
            # Use OpenAI API for generation
            model_name = "gpt-4.1"
            with ThreadPoolExecutor(max_workers=20) as pool:
                api_outputs = list(pool.map(
                    lambda p: client.responses.create(model=model_name, input=p).output_text,
                    batch_prompts
                ))
        elif openrouter_eval:
            # Use OpenRouter API for generation
            with ThreadPoolExecutor(max_workers=20) as pool:
                api_outputs = list(pool.map(
                    lambda p: openrouter_generate(p, temperature=0.6, max_tokens=max_new_tokens),
                    batch_prompts
                ))
        else:
            # Use vLLM for generation
            vllm_outputs = llm.generate(batch_prompts, sampling_params)
            api_outputs = [output.outputs[0].text for output in vllm_outputs]
        
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
                
                eval_prompt = f"Instruction: You are a seasoned text coherence evaluator. Read the TEXT below and rate the TEXT's overall coherency on a 1-to-10 scale, where 1 means less coherent than the Base Answer, 5 means as coherent as the Base Answer, and 10 means way more coherent than the Base Answer. Be a strict and conservative evaluator and only gave a high score when the TEXT is truly better than the Base Answer. Base Answer: <<<{base_answer}>>> TEXT:<<<{model_answer}>>> Return your answer in exactly this format: Coherency score: <integer 1-10>. One-shot example: Base Answer: <<<A>>>  TEXT:<<<B>>> Assistant: Coherency score: <integer 1-10>. Reasoning: <≤ 2 concise sentences explaining why you chose that score>\nResponse:"

                eval_prompt_list.append(eval_prompt)


            if not openrouter_eval and not api_eval:
                sampling_params_eval = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

           
            
            if api_eval:
                # Use OpenAI API for evaluation
                model_name = "gpt-4.1"
                with ThreadPoolExecutor(max_workers=20) as pool:
                    eval_result_list = list(pool.map(
                        lambda p: client.responses.create(model=model_name, input=p).output_text,
                        eval_prompt_list
                    ))
            elif openrouter_eval:
                # Use OpenRouter API for evaluation
                with ThreadPoolExecutor(max_workers=20) as pool:
                    eval_result_list = list(pool.map(
                        lambda p: openrouter_generate(p, temperature=0.0, max_tokens=64),
                        eval_prompt_list
                    ))
            else:
                # Use vLLM for evaluation
                eval_vllm_outputs = llm.generate(eval_prompt_list, sampling_params_eval)
                eval_result_list = [output.outputs[0].text for output in eval_vllm_outputs]

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
                reward_str = get_humor_rating(eval_result)
                try:
                    reward_value = int(reward_str)
                except:
                    reward_value = 0
                print("[]"*20, "\n eval_result", eval_result)
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
        
        if api_eval:
            # Use OpenAI API for reflexion generation
            model_name = "gpt-4.1"
            with ThreadPoolExecutor(max_workers=20) as pool:
                api_outputs = list(pool.map(
                    lambda p: client.responses.create(model=model_name, input=p).output_text,
                    batch_prompts
                ))
        elif openrouter_eval:
            # Use OpenRouter API for reflexion generation
            max_token_reflexion = 256
            with ThreadPoolExecutor(max_workers=20) as pool:
                api_outputs = list(pool.map(
                    lambda p: openrouter_generate(p, temperature=0.6, max_tokens=max_token_reflexion),
                    batch_prompts
                ))
        else:
            # Use vLLM for reflexion generation
            max_token_reflexion = 256
            sampling_params_reflexion = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_token_reflexion)
            vllm_outputs = llm.generate(batch_prompts, sampling_params_reflexion)
            api_outputs = [output.outputs[0].text for output in vllm_outputs]
        
        for i, output_obj in enumerate(api_outputs):
            # Retrieve generated text.
            generated_text = output_obj
            
            reflexion = generated_text
            
            
            sample = samples[i]
            
            
            last_weak_demo = sample["weak_demos"][-1]
            
            last_weak_demo['plan'] = reflexion
        
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

            

            this_time_change = ""
            if rejection_sampling:
                this_time_change += "best_of_n_"
            else:
                this_time_change += "reflexion_"
            this_time_change += "new_eval_prompt"

            this_time_change += f"_evalnum_{max_eval_samples}"
            run = f"{this_time_change}_n_{n}"
            path = f"/sfs/weka/scratch/ks8vf/code_submission/ICL/{task}/{base_model_id}/{run}"
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
    # Test with fewer rounds to verify implementation
    evaluate_checkpoint(n=40)  # 100 rounds for full experiment
    # main()