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
from datetime import datetime
import uuid
plt.style.use(['science', 'no-latex'])

# Increase font size for all elements
plt.rcParams.update({
    'font.size': 10,  # Base font size
    # 'axes.titlesize': 36,  # Title font size
    'axes.labelsize': 30,  # Axis label font size
    'xtick.labelsize': 36,  # X-axis tick label size
    'ytick.labelsize': 36,  # Y-axis tick label size
    'legend.fontsize': 36,  # Legend font size
    # 'figure.titlesize': 54  # Figure title size
})

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

    offset = 0
    # extract bootstrap_attempts' rewards if exists
    if 'bootstrap_attempts' in data['0'].keys() and len(data['0']['bootstrap_attempts']) > 0:

        for env_id in data.keys():
            rewards_list_list = []
            for bootstrap_idx in data[env_id]['bootstrap_attempts'].keys():
                rewards_list_list.append(data[env_id]['bootstrap_attempts'][bootstrap_idx]['rewards'])
            for i, rewards_list in enumerate(rewards_list_list):
                rewards_list_list[i] = np.sum([0 if x < 0 else x for x in rewards_list])
            dict_data[env_id][0] = [np.mean(rewards_list_list)]
            offset = 1
    # extract round_attempts' rewards
    for env_id in data.keys():
        for round_idx in data[env_id]['round_attempts'].keys():
            dict_data[env_id][int(round_idx) + offset] = data[str(env_id)]['round_attempts'][str(round_idx)]['0']['rewards']
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
        std_devs = df.std(axis=1)/np.sqrt(len(df))
        rounds = df.index
        color = colors[idx % len(colors)]
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        
        ax.plot(rounds, means, f'{color}-', label=label)
        # ax.fill_between(rounds, means - std_devs, means + std_devs, alpha=0.3, color=color)

    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Reward')
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
        means = pd.Series(gaussian_filter(means, sigma=1), index=means.index)
        std_devs = df_running_max.std(axis=1)/np.sqrt(len(df))/4
        rounds = df.index
        color = colors[idx % len(colors)]
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        print(means)
        ax.plot(rounds, means, f'{color}-', label=label)
        ax.fill_between(rounds, means - std_devs, means + std_devs, alpha=0.3, color=color)

    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Running Max Episode Return')
    ax.legend()

    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'figures/{datetime.now().strftime("%Y%m%d_%H%M%S")}-{uuid.uuid4()}.pdf', format='pdf', bbox_inches='tight')
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

    ax.set_xlabel('Trial Number')
    ax.set_ylabel(f'Sliding Average Reward (Window Size: {window_size})')
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
        
        ax.plot(rounds, means, f'{color}-', label=label, )
        ax.fill_between(rounds, means - .5*std_devs/np.sqrt(len(df)), means + .5*std_devs/np.sqrt(len(df)), alpha=0.3, color=color)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel(f'Episode Return')
    ax.legend(fontsize=18)

    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'figures/{datetime.now().strftime("%Y%m%d_%H%M%S")}-{uuid.uuid4()}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def get_cost(path):
    """
    for each env:
        round_cost = 0
        for each round, for each messages in raw_prompts:
            input_count = len(all except the last one)
            output_count = len(the last one)
            cost = cost_input * input_count + cost_output * output_count
            round_cost += cost
    """
    cost_input = .4e-6
    cost_output = 1.6e-6
    # load raw_prompts
    path = find_sciworld_file(path, raw_prompts=True)
    data = json.load(open(path), object_hook=convert_keys_to_int)
    costs = []
    for env_id in data.keys():
        for round_idx in data[env_id]['round_attempts'].keys():
            round_cost = 0
            for step_idx in range(len(data[env_id]['round_attempts'][round_idx][0])):
                for message_idx, message in enumerate(data[env_id]['round_attempts'][round_idx][0][step_idx]):
                    if message_idx < len(data[env_id]['round_attempts'][round_idx][0][step_idx]) - 1:
                        round_cost += len(message['content']) * cost_input
                    else:
                        round_cost += len(message['content']) * cost_output
            costs.append({'env_id': env_id, 'round_idx': round_idx, 'cost': round_cost})
    df_cost = pd.DataFrame(costs)
    df_cost = df_cost.pivot(index='round_idx', columns='env_id', values='cost')
    return df_cost

def plot_cost_reward_sum(*args, **kwargs):
    """
    Plot cost on x-axis, reward sum on y-axis.
    Input should be pairs of (cost_df, reward_df) for each method.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Add more colors if needed
    markers = ['o', 's', '^', 'D', '*', 'x', '+']  # Different markers for each method
    
    # Process pairs of dataframes (cost_df, reward_df)
    for i in range(0, len(args), 2):
        if i+1 >= len(args):
            break
            
        cost_df = args[i]
        reward_df = args[i+1]
        
        # Calculate mean cost and running max reward
        mean_costs = cost_df.cumsum(axis=0).mean(axis=1)
        running_max_rewards = reward_df.cummax(axis=0).mean(axis=1)
        # smooth reward
        running_max_rewards = pd.Series(gaussian_filter(running_max_rewards, sigma=1), index=running_max_rewards.index)
        
        color = colors[(i//2) % len(colors)]
        marker = markers[(i//2) % len(markers)]
        label = kwargs.get(f'label_{i//2}', f'Method {i//2+1}')
        
        ax.plot(mean_costs, running_max_rewards, color=color, marker=marker, 
                markersize=8, label=label, linewidth=2)
        
        # Annotate some points with round numbers
        # for round_idx in range(0, len(mean_costs), 5):
        #     if round_idx < len(mean_costs):
        #         ax.annotate(f'T{round_idx}', 
        #                    (mean_costs.iloc[round_idx], running_max_rewards.iloc[round_idx]),
        #                    textcoords="offset points", 
        #                    xytext=(0,10), 
        #                    ha='center')
    
    ax.set_xlabel('Cumulative Cost (in USD for 4.1-mini)')
    ax.set_ylabel('Running Max Episode Return')
    ax.legend()
    
    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'figures/{datetime.now().strftime("%Y%m%d_%H%M%S")}-{uuid.uuid4()}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

#%%
cols_to_drop = [1, 8, 10, 11, 24, 25]
cols_to_drop = []
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_0111_pos_reward/sciworld_data_round_49_final.json") # OG
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1933") # better prompt, shared on slack
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_2104") # 29 envs
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_2111_4.1-mini") # 4.1-mini
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1114_e&e_but_not") # 29 envs rerun
df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250511_0031_29_4.1_alsoObs")
df_pos_reward = df_pos_reward[(df_pos_reward.index.astype(int) % 2 == 0) | (df_pos_reward.index.astype(int) < 1)]

# df_random_sampling = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250507_1517_random_sampling")
df_random_sampling = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250510_2108_29_4.1_mini")
# df_3_attempts = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1510_3_attempts")
df_3_attempts = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250511_1327_29_4.1_3_icl")
# df_zero_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1636_zero_rewards")
# df_zero_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1751_zero_rewards")
# df_exploration_only = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1653_explore_only")
df_exploration_only = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1112_only_explore") # 29 envs
# df_e_and_e = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1730_e_and_e")
# df_e_and_e = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_2005_e_and_e")
df_e_and_e = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1446_e&e") # 29 envs
# df_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250510_1609_reflexion_29_4.1mini")
df_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250511_1754_reflexion_4.1mini_obsfix")
# df_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250510_2332_reflexion_4.1mini_concise")
# df_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250511_1817_29_4.1_reflexion_3")

#%%
plot_per_step_running_max(df_exploration_only, df_pos_reward, df_e_and_e, df_reflexion, df_random_sampling, df_3_attempts, label_0='Explore Only', label_1='Pos Reward', label_2='E&E', label_3='Reflexion', label_4='Random Sampling', label_5='3 Attempts')
plot_per_step_gaussian_smoothed(df_exploration_only, df_pos_reward, df_e_and_e, df_reflexion, df_random_sampling, df_3_attempts, param=1, label_0='Explore Only', label_1='Pos Reward', label_2='E&E', label_3='Reflexion', label_4='Random Sampling', label_5='3 Attempts')
# plot_per_step(df_exploration_only, df_pos_reward, df_e_and_e, df_reflexion, df_random_sampling, df_3_attempts, label_0='Explore Only', label_1='Pos Reward', label_2='E&E', label_3='Reflexion', label_4='Random Sampling', label_5='3 Attempts')

#%%
# plot_per_step_running_max(df_reflexion, df_pos_reward, df_random_sampling, label_0='Reflexion', label_1='ICRL', label_2='Best-of-N', save=True)
plot_per_step_gaussian_smoothed(df_reflexion, df_pos_reward, df_random_sampling, param=1, label_0='Reflexion', label_1='ICRL', label_2='Random Sampling', save=True)
# %%
path = find_sciworld_file("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250510_2332_reflexion_4.1mini_concise", raw_prompts=True)
# path = find_sciworld_file("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1114_e&e_but_not", raw_prompts=True)
data = json.load(open(path), object_hook=convert_keys_to_int)

#%%
round = 15
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
df_cost_reflexion = get_cost("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250511_1754_reflexion_4.1mini_obsfix")
df_reward_sum_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250511_1754_reflexion_4.1mini_obsfix")
df_cost_pos_reward = get_cost("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250511_0031_29_4.1_alsoObs")
df_reward_sum_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250511_0031_29_4.1_alsoObs")
df_cost_random_sampling = get_cost("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250510_2108_29_4.1_mini")
df_reward_sum_random_sampling = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250510_2108_29_4.1_mini")
# %%
plot_cost_reward_sum(df_cost_reflexion, df_reward_sum_reflexion, df_cost_pos_reward, df_reward_sum_pos_reward, df_cost_random_sampling, df_reward_sum_random_sampling, label_0='Reflexion', label_1='ICRL', label_2='Random Sampling', save=True)
# %%
df_cost_reflexion.cumsum(axis=0).mean(axis=1)

# %%
df_cost_pos_reward.cumsum(axis=0).mean(axis=1)
