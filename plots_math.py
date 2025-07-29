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
import pickle
import re
from math_bench import DataStore
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

def find_math_file(folder_path):
    """Find the math data file in a given folder."""
    # Look for the most recent round file first
    pattern = os.path.join(folder_path, "data_round_*_final.pkl")
    files = glob.glob(pattern)
    
    if files:
        # Sort by round number to get the latest
        def extract_round_num(filepath):
            match = re.search(r"data_round_(\d+)_final\.pkl", os.path.basename(filepath))
            return int(match.group(1)) if match else -1
        
        files.sort(key=extract_round_num, reverse=True)
        return files[0]
    
    # If no round files found, look for initial attempts file
    initial_file = os.path.join(folder_path, "data_initial_attempts.pkl")
    if os.path.exists(initial_file):
        return initial_file
    
    raise FileNotFoundError(f"No math data file found in {folder_path}")

def get_sum_df(path):
    """Get sum dataframe for math_bench.py data format."""
    data_file = find_math_file(path)
    
    # Load pickle data
    data_store = pickle.load(open(data_file, "rb"))
    
    dict_data = defaultdict(dict)
    
    # Get all unique round indices
    all_rounds = set()
    for problem_history in data_store.problem_histories:
        for attempt in problem_history.attempts:
            all_rounds.add(attempt.round_idx)
    
    all_rounds = sorted(list(all_rounds))
    
    # Organize data by round and problem
    for problem_idx, problem_history in enumerate(data_store.problem_histories):
        # Group attempts by round
        attempts_by_round = defaultdict(list)
        for attempt in problem_history.attempts:
            reward = 1 if attempt.reward > .9 else 0
            attempts_by_round[attempt.round_idx].append(reward)
        
        # For each round, take the sum/mean of rewards (depending on what makes sense)
        for round_idx in all_rounds:
            if round_idx in attempts_by_round:
                # Take the mean reward if multiple attempts in the same round
                dict_data[problem_idx][round_idx] = np.mean(attempts_by_round[round_idx])
            else:
                # No attempt in this round for this problem
                dict_data[problem_idx][round_idx] = 0
    
    df = pd.DataFrame(dict_data)
    
    # Handle negative rewards (set to 0)
    df = df.applymap(lambda x: max(0, x) if isinstance(x, (int, float)) else x)
    
    # Filter to reasonable number of rounds
    df = df[df.index.astype(int) < 40]
    
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
    
    first_perf = []
    for idx, df in enumerate(dfs):
        # create running max df
        df_running_max = df.cummax(axis=0)
        means = df_running_max.mean(axis=1)
        means = pd.Series(gaussian_filter(means, sigma=1), index=means.index)
        std_devs = df_running_max.std(axis=1)/np.sqrt(len(df))/4
        rounds = df.index
        color = colors[idx % len(colors)]
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        print(label, means.iloc[-1])
        first_perf.append(means.iloc[0])
        ax.plot(rounds, means, f'{color}-', label=label)
        ax.fill_between(rounds, means - std_devs, means + std_devs, alpha=0.3, color=color)
    print('first perf', np.mean(first_perf))

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
    Calculate costs for math_bench.py data.
    For math problems, we'll estimate cost based on token counts in the prompts.
    """
    # This would need to be implemented based on how you want to calculate costs
    # for math problems. For now, returning a placeholder.
    raise NotImplementedError("Cost calculation for math data not yet implemented")

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
    
    ax.set_xlabel('Cumulative Cost (in USD)')
    ax.set_ylabel('Running Max Reward')
    ax.legend()
    
    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'figures/{datetime.now().strftime("%Y%m%d_%H%M%S")}-{uuid.uuid4()}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

#%%
# qwen3.32b
df_aime = get_sum_df("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/icrl/20250728_0008_aime25_formatted_weird")
df_aime_local = get_sum_df("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/icrl/20250728_2109_aime_local")
df_aime_local_notee = get_sum_df("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/icrl/20250729_0118_aime_local_notee")
df_aime_reflexion_fair = get_sum_df("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/reflexion/20250728_2240_aime_reflexion")
df_aime_selfrefine = get_sum_df("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/selfrefine/20250728_2308_aime_selfrefine")

df_hmmt_local = get_sum_df("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/icrl/20250728_2235_hmmt")
df_hmmt_reflexion = get_sum_df("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/reflexion/20250728_2025_hmmt_reflexion")
df_hmmt_selfrefine_fair = get_sum_df("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/selfrefine/20250728_2308_hmmt_selfrefine")
df_hmmt_reflexion_fair = get_sum_df("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/reflexion/20250728_2325_hmmt_reflexion")

# qwen3.32b reasoning
df_aime_reason = get_sum_df("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/icrl/20250728_1719_aime_reason")
df_aime_reason_reflexion = get_sum_df("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/reflexion/20250728_2127_aime_reason_reflexion")
#%%
data = DataStore.load_data_snapshot("/home/jovyan/shared/amoeini/neurips/ICRL-for-LLM-Agent/ICL/math/icrl/20250728_0008_aime25_formatted_weird")
#%%
df_stat = df_aime
quantiles = df_stat.quantile([0.25, 0.5, 0.75], axis=1)
print(quantiles)
print()
print(df_stat.mean(axis=1))
#%%
# Print all raw prompts and model outputs for question 0
problem_idx = 10  # You can change this to select a different problem
print("="*100)
print(f"QUESTION {problem_idx} - PROBLEM:")
print("="*100)
problem_text = data.problem_histories[problem_idx].problem.problem
print(problem_text)
print(f"\nCorrect Answer: {data.problem_histories[problem_idx].problem.answer}")
print("\n" + "="*100)

print("\nALL ATTEMPTS FOR QUESTION 0:")
print("="*100)

for i, attempt in enumerate(data.problem_histories[problem_idx].attempts):
    print(f"\n{'='*50} ATTEMPT {i+1} (Round {attempt.round_idx}) {'='*50}")
    print(f"Reward: {attempt.reward}")
    print("\nRAW PROMPT:")
    print("-" * 80)
    for j, message in enumerate(attempt.raw_prompt):
        print(f"Message {j+1} ({message['role']}):")
        print(message['content'])
        print("-" * 40)
    
    print("\nMODEL OUTPUT:")
    print("-" * 80)
    print(attempt.model_output)
    print("-" * 80)
#%%
plot_per_step_running_max(
    df_aime, df_aime_local, df_aime_local_notee, df_aime_reflexion_fair, df_aime_selfrefine,
    label_0="AIME", label_1="AIME Local", label_2="AIME Local Notee", label_3="AIME Reflexion Fair", label_4="AIME Selfrefine", param=1)
#%%
plot_per_step_running_max(
    df_hmmt_reflexion, df_hmmt_selfrefine_fair, df_hmmt_reflexion_fair, df_hmmt_local,
    label_0="HMMT Reflexion", label_1="HMMT Selfrefine Fair", label_2="HMMT Reflexion Fair", label_3="HMMT Local", param=1)
#%%
plot_per_step_running_max(
    df_aime_reason, df_aime_reason_reflexion,
    label_0="AIME Reason", label_1="AIME Reason Reflexion", param=1)
#%%