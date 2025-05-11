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
from scipy.ndimage import gaussian_filter
import scienceplots

plt.style.use(['science', 'no-latex'])

def find_sciworld_file(folder_path, raw_prompts=False):
    """Find the sciworld data file in a given folder."""
    if raw_prompts:
        pattern = os.path.join(folder_path, "raw_prompts_sciworld_data_round_*_final.json")
    else:
        pattern = os.path.join(folder_path, "sciworld_data_round_*_final.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No sciworld data file found in {folder_path}")
    return files[0]  # Return the first matching file

def convert_keys_to_int(obj):
    """Convert string keys to integers if possible."""
    if isinstance(obj, dict):
        return {int(k) if isinstance(k, str) and k.isdigit() else k: convert_keys_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_int(item) for item in obj]
    return obj

def get_sum_df(path):
    path = find_sciworld_file(path)
    data = json.load(open(path))
    dict_data = defaultdict(dict)
    for env_id in data.keys():
        for round_idx in data[env_id]['round_attempts'].keys():
            dict_data[env_id][round_idx] = data[str(env_id)]['round_attempts'][str(round_idx)]['0']['rewards']
    df = pd.DataFrame(dict_data)
    df = df.applymap(lambda x: [0 if xx < 0 else xx for xx in x])
    df = df.applymap(lambda x: np.sum(x) if isinstance(x, list) else x)
    df = df[df.index.astype(int) < 40]
    df = df.drop(columns=df.columns[cols_to_drop])
    return df

def plot_per_step(*dfs, **kwargs):
    # Create a single figure for comparing all methods
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Add more colors if needed
    
    for idx, df in enumerate(dfs):
        means = df.mean(axis=1)
        std_devs = df.std(axis=1)
        rounds = df.index
        color = colors[idx % len(colors)]
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        
        ax.plot(rounds, means, f'{color}-', label=label)
        # ax.fill_between(rounds, means - std_devs, means + std_devs, alpha=0.3, color=color)

    ax.set_xlabel('Round')
    ax.set_ylabel('Reward')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_per_step_running_max(*dfs, **kwargs):
    # Create a single figure for comparing all methods
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Add more colors if needed
    
    for idx, df in enumerate(dfs):
        # create running max df
        df_running_max = df.cummax(axis=0)
        means = df_running_max.mean(axis=1)
        rounds = df.index
        color = colors[idx % len(colors)]
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        
        ax.plot(rounds, means, f'{color}-', label=label)

    ax.set_xlabel('Round')
    ax.set_ylabel('Running Max Reward')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_per_step_sliding_average(*dfs, window_size=10, **kwargs):
    # Create a single figure for comparing all methods
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Add more colors if needed
    
    for idx, df in enumerate(dfs):
        # Calculate sliding window average
        df_sliding = df.rolling(window=window_size).mean()
        means = df_sliding.mean(axis=1)
        # Only plot from window_size onwards
        rounds = df.index[window_size - 1:]
        means = means[window_size - 1:]
        color = colors[idx % len(colors)]
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        
        ax.plot(rounds, means, f'{color}-', label=label)

    ax.set_xlabel('Round')
    ax.set_ylabel(f'Sliding Average Reward (Window Size: {window_size})')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_per_step_gaussian_smoothed(*dfs, param, **kwargs):
    # Create a single figure for comparing all methods
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Add more colors if needed
    
    for idx, df in enumerate(dfs):
        # Apply Gaussian smoothing to each column
        df_smoothed = df.apply(lambda x: gaussian_filter(x, sigma=param))
        means = df_smoothed.mean(axis=1)
        rounds = df.index
        color = colors[idx % len(colors)]
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        std_devs = df_smoothed.std(axis=1)
        
        ax.plot(rounds, means, f'{color}-', label=label)
        ax.fill_between(rounds, means - .5*std_devs/np.sqrt(len(df)), means + .5*std_devs/np.sqrt(len(df)), alpha=0.3, color=color)
    ax.set_xlabel('Round')
    ax.set_ylabel(f'Gaussian Smoothed Reward (σ={param})')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

#%%
cols_to_drop = [1, 8, 10, 11, 24, 25]
cols_to_drop = []
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_0111_pos_reward/sciworld_data_round_49_final.json") # OG
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1933") # better prompt, shared on slack
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_2104") # 29 envs
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_2111_4.1-mini") # 4.1-mini
df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1114_e&e_but_not") # 29 envs rerun
# df_pos_reward = df_pos_reward[(df_pos_reward.index.astype(int) % 2 == 1) | (df_pos_reward.index.astype(int) < 1)]

# df_random_sampling = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250507_1517_random_sampling")
df_random_sampling = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250510_2108_29_4.1_mini")
# df_3_attempts = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1510_3_attempts")
# df_zero_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1636_zero_rewards")
# df_zero_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1751_zero_rewards")
# df_exploration_only = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1653_explore_only")
df_exploration_only = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1112_only_explore") # 29 envs
# df_e_and_e = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1730_e_and_e")
# df_e_and_e = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_2005_e_and_e")
df_e_and_e = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1446_e&e") # 29 envs
df_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250510_1609_reflexion_29_4.1mini")
# df_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250510_1746_reflexion_29")

#%%

plot_per_step_running_max(df_exploration_only, df_pos_reward, df_e_and_e, df_reflexion, df_random_sampling, label_0='Explore Only', label_1='Pos Reward', label_2='E&E', label_3='Reflexion', label_4='Random Sampling')
plot_per_step(df_exploration_only, df_pos_reward, df_e_and_e, df_reflexion, df_random_sampling, label_0='Explore Only', label_1='Pos Reward', label_2='E&E', label_3='Reflexion', label_4='Random Sampling')
plot_per_step_gaussian_smoothed(df_exploration_only, df_pos_reward, df_e_and_e, df_reflexion, df_random_sampling, param=1, label_0='Explore Only', label_1='Pos Reward', label_2='E&E', label_3='Reflexion', label_4='Random Sampling')
# %%
path = find_sciworld_file("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250510_1609_reflexion_29_4.1mini", raw_prompts=True)
data = json.load(open(path), object_hook=convert_keys_to_int)

#%%
round = 20
env_id = 8
for i in range(len(data[env_id]['round_attempts'][round][0][-1])):
    print(data[env_id]['round_attempts'][round][0][-1][i]['content'])
    print('-'*100)
#%%
h = 4
for i in range(2*h if h > 0 else 1):
    # print(data['4']['round_attempts'][str(h)]['0']['attempt_prompts'][i]['content'])
    print(data[env_id]['round_attempts'][round][0][h][i]['content'])
    print('-'*100)

# %%
h = 5
for i in range(h):
    print(data[env_id]['bootstrap_attempts'][0][h][i]['content'])
    print('-'*100)

# %%
df_reflexion.cummax(axis=0).mean() - df_pos_reward[:len(df_reflexion)].cummax(axis=0).mean()

# %%
df_reflexion.max(axis=0) - df_pos_reward[:len(df_reflexion)].max(axis=0)

# %%