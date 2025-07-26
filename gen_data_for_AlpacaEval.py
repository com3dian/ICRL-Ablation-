import sys
import json 
import numpy as np
from tqdm import tqdm

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

task = get_task("text")
print(f"Loaded {len(task)} creative-writing instructions.\n")

method = "self-refine"
method = "ICRL"

# load responses for self-refine
# path = "/sfs/weka/scratch/ks8vf/ICL/creative_writing_api/Qwen/Qwen3-32B/self_refine_seperate_run_evalnum_100_n_100/output_list.json"

# load responses for ICRL
if method == "ICRL":
    path = "/sfs/weka/scratch/ks8vf/code_submission/ICL/creative_writing_api/Qwen/Qwen3-32B/ICRL_new_eval_prompt_evalnum_100_n_100/output_list.json"
elif method == "self-refine":
    path = "/sfs/weka/scratch/ks8vf/ICL/creative_writing_api/Qwen/Qwen3-32B/self_refine_seperate_run_evalnum_100_n_100/output_list.json"

with open(path, 'r') as f:
    output = json.load(f)
    
best_responses = []
for question in output[:]:
    reward_list = []
    # Process all trials (up to 100)
    for trial in question[:20]:
        reward_list.append(float(trial['reward']))
    idx = np.argmax(reward_list)
    
    try:
        best_responses.append(question[idx]["generated_text"][9:-10].split('Passage:')[1])
    except:
        print("--"*20)
        print("--"*20)
        print(question[idx]["generated_text"])


# ---------------------------------------------------------------------
# 2.  Create AlpacaEval format entries
# ---------------------------------------------------------------------
entries = []

for idx in tqdm(range(100)):
    instruction = task.get_input(idx).strip()
    cot_instruction = cot_prompt.format(input=instruction)[:-125]

    entry = {
        "instruction": cot_instruction,
        "output": best_responses[idx],
        "generator": f"qwen3_32b_{method}",
        "dataset": "helpful_base",
        "datasplit": "eval"
    }

    entries.append(entry)


with open(f"/sfs/weka/scratch/ks8vf/code_submission/ICRL/qwen3_32b_{method}_responses.json", "w") as f:
    json.dump(entries, f, indent=2)

print(f"Successfully saved {len(entries)} entries to qwen3_32b_{method}_responses.json")