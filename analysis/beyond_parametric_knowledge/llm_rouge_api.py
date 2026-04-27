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

from distinct_n.metrics import distinct_n_corpus_level

from distinct_n.metrics import distinct_n_sentence_level

from openai import OpenAI


from concurrent.futures import ThreadPoolExecutor


client = OpenAI(api_key="sk-C8z62BDhmo4EW1bqOn2TTmdFR29ocUeZXLExkdmGS1T3BlbkFJQcA3zOug-aNTm98KC0Wjsv549b3OgxEGn9TKJknXMA")

from rouge_score import rouge_scorer
# from tqdm import tqdm

def compute_rouge_recall(hypotheses, references):
    """
    hypotheses, references: lists of strings of equal length.
    Returns a dict with averaged 'rouge1', 'rouge2', 'rougeL' F1 scores.
    """
    # rouge = evaluate.load('rouge')
    
    # results = rouge.compute(predictions=hypotheses, references=references, rouge_types=["rouge1"],use_stemmer=True)
    # results already contains 'rouge1', 'rouge2', 'rougeL' (F1 scores by default)
    
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    results = scorer.score(hypotheses, references)['rouge1'].recall
    return results





# Number of characters used to compute the reward.
num_char = 200
num_weak_demos = 1000

rejection_sampling = False
# To install Sentence-Transformers, run:
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer

import numpy as np

import evaluate

def compute_rouge(hypotheses, references):
    """
    hypotheses, references: lists of strings of equal length.
    Returns a dict with averaged 'rouge1', 'rouge2', 'rougeL' F1 scores.
    """
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=hypotheses, references=references)
    # results already contains 'rouge1', 'rouge2', 'rougeL' (F1 scores by default)
    return results
    
    
    


def distinct_1_paragraph(paragraph: str) -> float:
    """
    Compute distinct‑1 for a paragraph of text.

    :param paragraph: The input text (one or more sentences).
    :return: The distinct‑1 score: (# unique tokens) / (total tokens).
    """
    # 1) Normalize & tokenize: lowercase and grab word tokens
    tokens = re.findall(r"\w+", paragraph.lower())

    # 2) Delegate to the existing sentence‑level metric
    #    which handles the zero‑division guard.
    return distinct_n_sentence_level(tokens, 1)


def distinct_1_sentence(paragraph: str) -> float:
    """
    Compute the distinct‑1 score for a paragraph by:
      1. Splitting the paragraph into sentences.
      2. Tokenizing each sentence into words.
      3. Averaging distinct‑1 across sentences (corpus level).

    :param paragraph: The input text (one or more sentences).
    :return: The distinct‑1 score (average over sentences).
    """
    # 1) Split into sentences on ., ! or ? followed by whitespace
    sentences = re.split(r'(?<=[\.!?])\s+', paragraph.strip())
    
    # 2) Tokenize each sentence and lowercase
    token_lists = []
    for sent in sentences:
        tokens = re.findall(r"\w+", sent.lower())
        if tokens:
            token_lists.append(tokens)
    
    # 3) If no valid sentences, return 0.0
    if not token_lists:
        return 0.0
    
    # 4) Compute average distinct‑1 across sentences
    return distinct_n_corpus_level(token_lists, 1)

def cosine_similarity(vec1, vec2):
    # Calculate the cosine similarity between two vectors
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_texts(sentence_model, texts):
    # Load the pre-trained model
    # Encode the list of texts
    embeddings = sentence_model.encode(texts)
    return embeddings
        

def evaluate_checkpoint(
    checkpoint_path='meta-llama/Meta-Llama-3-8B-Instruct',
    base_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_name='tatsu-lab/alpaca',
    split="test",
    max_eval_samples=35,
    n=200,
    max_new_tokens=1000
):
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
    # with open('diversity_prompt_attempt.txt', 'r') as f:
    #     one_shot_prompt = f.read()
    
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Initialize the vLLM instance.
    # llm = LLM(model=base_model_id)
    # sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_new_tokens)
    
    # Prepare a list for all samples that meet the criteria.
    # Each entry stores the question, a history of weak demos, and outputs per round.
    samples = []
    for sample in dataset:
        question = sample.get('instruction', "")
        question_input = sample.get('input', "")
        target_output = sample.get('output', "")
        if len(question_input) != 0:
            continue
        samples.append({
            "question": question,
            "weak_demos": [],  # will hold dictionaries with keys: prompt, answer, reward
            "output": [],       # will record output details per round
            "target_output": target_output
        })
    
    num_samples = len(samples)
    print(f"Processing {num_samples} samples in {n} rounds...")
    
    
    # motivating_instruction = "<instruction>Instruction: Review all the example <attempt> </attempt> blocks, each pairing a Response with its Reward score, and identify the patterns that distinguish higher‑scoring answers. Infer the criteria for a good response based on these patterns. Then, for the final question presented, generate a completely new and improved response that can achieve the high reward score 10.00. Critically, avoid repeating or slightly modifying the Response or style of the previous Responses; instead, explore different angles or approaches you believe will better satisfy the inferred criteria and achieve a higher reward. Your output should only be this novel response to the following prompt.</instruction>\n"
    
    motivating_instruction = "<instruction>Instruction: Review all the example <attempt> </attempt> blocks, each pairing a Response with its Reward score, and identify the patterns that distinguish higher‑scoring answers. Infer the criteria for a good response based on these patterns. Then, for the final question presented, make an educated guess based on the past experiences and generate a response that has a better chance to achieve the high reward score 10.00. </instruction>\n"

    motivating_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. First, randomly choose from the two options: exploration or exploitation. If your choice is exploration, provide a response that is entirely different from any previous attempts demonstrated in the context, and wrap it in `<answer>…</answer>`. If your choice is expliotation, identify what distinguishes top‑scoring answers from low‑scoring ones. In one `<pattern>…</pattern>` block, list under “High‑reward texts:” the texts shared by the best responses, and under “Low‑reward texts:” the texts shared by the poor responses. Then, using those insights, craft a new answer to the final question that would likely earn a perfect 10.00, and wrap it in `<answer>…</answer>`."
    
    exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. Provide a response that is completely different from any previous attempts demonstrated in the context, and wrap it in `<answer>…</answer>`."
#     exploitation_instruction = "Instruction: Your overall goal is through trial and error, provide a response that matches closely to a target response. For this round, examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. Identify what distinguishes top‑scoring answers from low‑scoring ones. In one `<pattern>…</pattern>` block, list under “High‑reward texts:” the texts shared by the best responses, and under “Low‑reward texts:” the texts shared by the poor responses. Provide as much details as possible from previous attempts. List under “Less frequent but high-potential texts:” that less frequently appear from the demonstrations but may have a potential to increase the reward. Then, combining all those insights, especially from the less frequent but high-potential texts, craft a new answer to the final question that would likely earn a perfect 10.00, and wrap it in `<answer>…</answer>`."
    
    
    exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> entries. Each entry contains: •A candidate Response •Its numerical Reward  Your task: 1. Parse all <attempt> blocks and identify the attempts with the top reward scores. 2. Among those top‐scoring attempts, select the ones that are most distinct from each other in style or angle. 3. Create a single new mega response that fuses the strongest elements from each distinct, high‐scoring attempt. 4. Return only this new, combined mega response wrapped in an <answer>…</answer> tag."
    
    # pure_exploration_instruction = "Instruction: Your overall goal is through trial and error, provide a response that matches closely to a target response. For this round, examine all the `<attempt>…</attempt>` examples, each showing a candidate Response and its Reward. Identify what distinguishes top‑scoring answers from low‑scoring ones. Identify what distinguishes top‑scoring answers from low‑scoring ones. In one `<pattern>…</pattern>` block, list under “High‑reward texts:” the texts shared by the best responses, and under “Low‑reward texts:” the texts shared by the poor responses. Provide as much details as possible from previous attempts. Then, combining all the texts and patterns from the top-scoring answers to craft a mega answer that could potentially achieve the highest reward so far, and wrap it in `<answer>…</answer>`"
    # motivating_instruction = ""
    
    
    task_prompt = "**Task**: Provide a short piece of response for the following prompt to match as closely as a ground truth text, which is hidden, and only the degree of match is specified in reward. \n"
    
    # Run n rounds.
    for round_idx in range(n):
        print(f"Round {round_idx+1}/{n}...")
        # Build a prompt for each sample.
        batch_prompts = []
        for sample in samples:
            # prompt = one_shot_prompt
            prompt = ""
            if not rejection_sampling:
                # Add previous weak demonstrations if any.
                for weak_demo in sample["weak_demos"][-num_weak_demos:]:
                    prompt += "<attempt>\n"
                    # prompt += task_prompt
                    prompt += f"**Prompt**: {weak_demo['prompt']}\n"
                    prompt += f"**Reward**: {weak_demo['reward']}\n"
                    prompt += weak_demo['answer'] + "\n"

                if round_idx % 2 == 0:
                    prompt += exploration_instruction
                else:
                    prompt += exploitation_instruction
            else:
                prompt += "**Task**: Provide a short piece of response for the following prompt to match as closely as a ground truth text, which is hidden"
                # prompt += "**Task**: Write the longest, most comprehensive response for the following prompt potentially covering all possible key words that could be mentioned."
                prompt += f"**Prompt**: {sample['question']}\n"
            # # Append the new attempt with the current question.
            # # Adding an instruction
            # prompt += "<attempt>\n"
            # prompt += task_prompt
            # prompt += f"**Prompt**: {sample['question']}\n"
            # prompt += "**Reward**: 10.00\n"
            batch_prompts.append(prompt)
        
        # Send all prompts together in one batch.
        # (Assuming vLLM accepts a list of prompts.)
        
        # api_outputs = []
        # for prompt in tqdm(batch_prompts):
        #     response = client.responses.create(
        #         model="o3-mini",
        #         input=prompt
        #     )
        #     api_outputs.append(response.output_text)
        # model_name = "o3-mini"
        model_name = "gpt-4.1-mini"
            
        with ThreadPoolExecutor(max_workers=12) as pool:
            api_outputs = list(pool.map(
                lambda p: client.responses.create(model=model_name, input=p).output_text,
                batch_prompts
            ))
            
        # asyncio.run(main())
        
        
        
        # vllm_outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process batch responses.
        for i, generated_text in enumerate(api_outputs):
            # Retrieve generated text.
            # generated_text = output_obj.outputs[0].text
            # Use regex to extract text up to </attempt>
            # pattern = r"(?s)^.*?</attempt>"
            # m = re.match(pattern, generated_text, flags=re.DOTALL)
            # if m:
            #     model_answer = m.group(0)
            # else:
            #     model_answer = ""
            
            # pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
            # try:
            #     model_answer = pattern.findall(generated_text)[0]
            # except:
            #     model_answer = ""
            
            
            model_answer = generated_text
            
            # Compute reward.
            
            # reward_value = len(model_answer) / num_char
            reward_value = -1
            reward_str = f"{reward_value:.2f}"
            
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
                "reward": reward_value
            })
            
        # calculate reward again
        for i in range(len(samples)):
            samples[i]['weak_demos'][0]['reward'] = "0.00"

            original_answer = samples[i]['weak_demos'][0]['answer']
            last_answer = samples[i]['weak_demos'][-1]['answer']
            
            target_answer = samples[i]['target_output']

            # original_embedding = encode_texts(sentence_model, original_answer)
            # last_embedding = encode_texts(sentence_model, last_answer)
            # reward = 1 - cosine_similarity(original_embedding, last_embedding)
            
            # reward = compute_rouge([target_answer],[last_answer])['rouge1']
            
            reward = compute_rouge_recall(target_answer,last_answer)
            reward *= 10


            # reward = distinct_1_paragraph(last_answer)
            
            # reward -= 0.8
            # reward *- 10
            
            reward_str = f"{reward:.3f}"

            samples[i]['weak_demos'][-1]['reward'] = reward_str

            samples[i]["output"][-1]['reward'] = reward_str
                
                
        
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
                round_rewards = [float(entry["reward"]) for entry in sample["output"]]
                avg_reward_list.append(np.mean(round_rewards))
                last_reward_list.append(round_rewards[-1] if round_rewards else 0)
                gen_list.append(sample["output"][-1]["generated_text"] if sample["output"] else "")
                output_list.append(sample["output"])

            # Save the results to files.


            task = 'alpaca_rouge_api'
            if rejection_sampling: 
                run = f"rejection_sampling_recall_{n}"
            else: 
                run = f"ICRL_recall_{n}"
            path = f"/sfs/weka/scratch/ks8vf/ICL/{task}/{model_name}/{run}"
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

            print("path: \n", path)

if __name__ == "__main__":
    
    print("Evaluating checkpoint in batch mode...")
    evaluate_checkpoint()