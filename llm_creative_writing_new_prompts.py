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
from vllm import LLM, SamplingParams

from concurrent.futures import ThreadPoolExecutor


from openai import OpenAI
num_char = 200


api_eval = False  # Set to False to use vLLM instead of OpenAI


rejection_sampling = 0
ICRL = 1
exploitation_only = 0
exploration_only_no_reward = 0
exploration_and_exploitation = 0

no_ICRL = 0

no_reward = 0
zero_reward = 0

exploration_or_exploitation = 0

meta_instruction_change = 1


num_weak_demo = 3000

load_samples = 0  # Don't load previous samples for testing

sort_by_reward = 0

# Initialize vLLM model when not using API
if not api_eval:
    llm = LLM(model="Qwen/Qwen3-32B", 
              tensor_parallel_size=2,  # Adjust based on GPU count
              gpu_memory_utilization=0.95)
else:
    client = OpenAI(api_key="Your_API_Key")

if meta_instruction_change: 
    exploitation_instruction = "Meta Prompt: surpass the highest reward observed so far. Review the trials, then output your candidate inside `<response>…</response>`."
    exploration_instruction = "Meta Prompt: Review every <trial> (response + reward). Reply with a new, non‑repeating answer wrapped in `<response>…</response>` tags"
else: 
    exploration_instruction = "Instruction: Examine all the `<trial>…</trial>` examples, each showing a candidate Response and its Reward. Provide a response that is different from previous trials demonstrated in the context, and wrap it in `<response>…</response>`."
    exploitation_instruction = "Instruction: You will be given multiple <trial>…</trial> entries. Each entry contains a candidate Response and its Reward. Your task: Based on the previous trials, try your best to produce a response that can achieve a higher reward, while making sure it correctly follows the task instruction, and put it in `<response>…</response>` format."

explore_or_exploit_instruction = "Instruction: Examine all the `<trial>…</trial>` examples, each showing a candidate Response and its Reward. You have two options, exploration or exploitation. For exploration, provide a response that is different from previous trials demonstrated in the context, and wrap it in `<response>…</response>`. For exploitation, make the best educated guess based on the high reward trials to produce response that can achieve a higher reward, while making sure it correctly follows the task instruction, and put it in `<response>…</response>` format. Pick one option to follow."


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
    base_model_id="Qwen/Qwen3-32B",
    split="test",
    max_eval_samples=45,
    n=51,
    max_new_tokens=1000
):
    num_samples = 100  # Test with 5 samples
    
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
    
    if load_samples:
        with open("intermediate_round_creative_writing_simple_prompt.pkl", "rb") as f:
            samples = pickle.load(f)

    
    # Run n rounds.
    for round_idx in range(n):
        print(f"Round {round_idx+1}/{n}...")
        # Build a prompt for each sample.
        batch_prompts = []
        for sample in samples:
            
            
            prompt = ""
            
            if not rejection_sampling: 
                # Add previous weak demonstrations if any.
                if sort_by_reward: 
                    weak_demos = sorted(sample["weak_demos"], key=lambda d: float(d["reward"]))
                else:
                    weak_demos = sample["weak_demos"]
                for weak_demo in weak_demos[-num_weak_demo:]:
                    prompt += "<trial>\n"
                    prompt += f"**Task Query**: {weak_demo['prompt']}\n"
                    if not no_reward: 
                        if zero_reward:
                            prompt += f"**Reward**: {0.00}\n"
                        else:
                            prompt += f"**Reward**: {weak_demo['reward']}\n"
                    prompt += weak_demo['answer'][:-28] + "\n"
                    if not no_reward: 
                        if zero_reward:
                            prompt += f"**Reward**: {0.00}\n"
                        else:
                            prompt += f"**Reward**: {weak_demo['reward']}\n"
                    prompt += "</trial>"
                if ICRL: 
                    if round_idx % 2 == 0:
                        prompt += exploration_instruction
                    else:
                        prompt += exploitation_instruction
                if exploration_only_no_reward:
                    prompt += exploration_instruction
                if exploration_and_exploitation:
                    prompt += explore_and_exploit_instruction
                if exploitation_only:
                    prompt += exploitation_instruction
                if exploration_or_exploitation:
                    prompt += explore_or_exploit_instruction
                if no_ICRL:
                    prompt += "Instruction: put your response to the following prompt in `<response>…</response>` format."

            prompt += f"**Task Query**: {sample['question']}\n"

            batch_prompts.append(prompt)
        
        if api_eval:
            # Use OpenAI API for generation
            model_name = "gpt-4.1"
            with ThreadPoolExecutor(max_workers=12) as pool:
                api_outputs = list(pool.map(
                    lambda p: client.responses.create(model=model_name, input=p).output_text,
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
                    
                pattern = re.compile(r'<response>(.*?)</response>', re.DOTALL)
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
                
                eval_prompt = f"Instruction: You are a seasoned text coherence evaluator. Read the TEXT below and rate the TEXT’s overall coherency on a 1-to-10 scale, where 1 means less coherent than the Base Answer, 5 means as coherent as the Base Answer, and 10 means way more coherent than the Base Answer. Be a strict and conservative evaluator and only gave a high score when the TEXT is truly better than the Base Answer. Base Answer: <<<{base_answer}>>> TEXT:<<<{model_answer}>>> Return your answer in exactly this format: Coherency score: <integer 1-10>. One-shot example: Base Answer: <<<A>>>  TEXT:<<<B>>> Assistant: Coherency score: <integer 1-10>. Reasoning: < 2 concise sentences explaining why you chose that score>\nResponse:"

                eval_prompt_list.append(eval_prompt)


            sampling_params_eval = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

           
            
            if api_eval:
                # Use OpenAI API for evaluation
                model_name = "gpt-4.1"
                with ThreadPoolExecutor(max_workers=12) as pool:
                    eval_result_list = list(pool.map(
                        lambda p: client.responses.create(model=model_name, input=p).output_text,
                        eval_prompt_list
                    ))
            else:
                # Use vLLM for evaluation
                eval_vllm_outputs = llm.generate(eval_prompt_list, sampling_params_eval)
                eval_result_list = [output.outputs[0].text for output in eval_vllm_outputs]


            _RATING_RE = re.compile(r"Coherency score:\s*(10|[1-9])\b")

        def get_humor_rating(text):
            """Return the humor rating or None if the pattern isn't present."""
            m = _RATING_RE.search(text)
            return int(m.group(1)) if m else None

        # Process batch responses.
        for i, output_obj in enumerate(api_outputs):
            # Retrieve generated text.
            generated_text = output_obj
            # Use regex to extract text between <response> tags
            pattern = re.compile(r'<response>(.*?)</response>', re.DOTALL)
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
        
        # Optionally, save intermediate results after each round.
        # For example, you could pickle the samples list:
        with open("intermediate_round_creative_writing_simple_prompt.pkl", "wb") as f:
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
                this_time_change += "ICRL_"
            this_time_change += "new_eval_prompt"

            this_time_change += f"_evalnum_{max_eval_samples}"
            run = f"{this_time_change}_n_{n}"
            path = f"/sfs/weka/scratch/ks8vf/code_submission/ICL/{task}/{base_model_id}/{run}"
            
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
    evaluate_checkpoint(n=100)  # 100 rounds for full experiment