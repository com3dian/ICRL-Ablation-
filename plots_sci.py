# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import glob
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict

# %%
def get_sum_df(path):
    data = json.load(open(path))
    dict_data = defaultdict(dict)
    for env_id in data.keys():
        for round_idx in data[env_id]['round_attempts'].keys():
            dict_data[env_id][round_idx] = data[str(env_id)]['round_attempts'][str(round_idx)]['0']['rewards']
    df = pd.DataFrame(dict_data)
    df = df.applymap(lambda x: np.sum(x) if isinstance(x, list) else x)
    return df

df_icrl = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250505_2341/sciworld_data_round_49_final.json")
df_rejsample = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250505_2250/sciworld_data_round_39_final.json")


# %%
# Create a single figure for comparing both methods
fig, ax = plt.subplots(figsize=(16, 6))

# Plot for ICRL data
means = df_icrl.mean(axis=1)
std_devs = df_icrl.std(axis=1)
rounds = range(len(means))

ax.plot(rounds, means, 'b-', label='ICRL Method')
ax.fill_between(rounds, means - std_devs, means + std_devs, alpha=0.3, color='b')

# Plot for Rejection Sampling data
means_rej = df_rejsample.mean(axis=1)
std_devs_rej = df_rejsample.std(axis=1)
rounds_rej = range(len(means_rej))

ax.plot(rounds_rej, means_rej, 'r-', label='Rejection Sampling Method')
ax.fill_between(rounds_rej, means_rej - std_devs_rej, means_rej + std_devs_rej, alpha=0.3, color='r')

ax.set_xlabel('Round')
ax.set_ylabel('Cumulative Reward')
ax.set_title('Comparison of ICRL and Rejection Sampling Methods')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

# %%

with open("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250505_2250/sciworld_data_round_19_final.json", "r") as f:
    data = json.load(f)


# %%
# path = "/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250505_2341/sciworld_data_round_49_final.json"
path = "/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250506_0308/sciworld_data_round_7_final.json"
path = "/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250506_0308/raw_prompts_sciworld_data_round_7_final.json"
data = json.load(open(path))
# dict_data = defaultdict(dict)
# for env_id in data.keys():
#     for round_idx in data[env_id]['round_attempts'].keys():
#         dict_data[env_id][round_idx] = data[str(env_id)]['round_attempts'][str(round_idx)]['0']['rewards']
# df = pd.DataFrame(dict_data)
# df
#%%
print(data['2']['round_attempts']['6']['0'][0][0]['content'])
