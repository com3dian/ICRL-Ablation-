"""
pseudocode:
input: target context length, raw prompt data
for each question 
    go through the timeline till the first instance where the input context length is greater than target context length
    return the running max till that point


"""

from dataclasses import dataclass
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'math'))
from math_bench import DataStore
from transformers import AutoTokenizer

@dataclass
class Config:
    data_path: tuple[str] = (
        "/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/icrl/20250729_0118_aime_local_notee",
        "/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/reflexion/20250728_2240_aime_reflexion",
        "/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/selfrefine/20250728_2308_aime_selfrefine"
    )
    target_context_length: int = 4096
    model_name: str = "Qwen/Qwen3-32B"

def get_raw_prompt_data(data_path):
    data_store = DataStore.load_data_snapshot(data_path)
    return data_store

def get_average_running_max(data_store, encoder, target_context_length):
    for problem in data_store.problem_histories:
        problem.running_max = 0
        for attempt in problem.attempts:
            messages = attempt.raw_prompt
            input_txt = encoder.apply_chat_template(messages, tokenize=False)
            prompt_tokens = encoder.encode(input_txt)
            if len(prompt_tokens) < target_context_length:
                reward = attempt.extra_fields["real_reward"] if "real_reward" in attempt.extra_fields else attempt.reward
                problem.running_max = max(problem.running_max, 1 if reward > .9 else 0)
    
    print([problem.running_max for problem in data_store.problem_histories])
    return sum(problem.running_max for problem in data_store.problem_histories) / len(data_store.problem_histories)

def main():
    encoder = AutoTokenizer.from_pretrained(Config.model_name)
    for data_path in Config.data_path:
        data_store = get_raw_prompt_data(data_path)
        average_running_max = get_average_running_max(data_store, encoder, Config.target_context_length)
        print(f"Average running max for {data_path}: {average_running_max}")

if __name__ == "__main__":
    main()