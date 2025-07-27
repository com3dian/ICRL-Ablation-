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
    pattern = os.path.join(folder_path, "data_round_*_final.json")
    files = glob.glob(pattern)
    
    if files:
        # Sort by round number to get the latest
        def extract_round_num(filepath):
            match = re.search(r"data_round_(\d+)_final\.json", os.path.basename(filepath))
            return int(match.group(1)) if match else -1
        
        files.sort(key=extract_round_num, reverse=True)
        return files[0]
    
    # If no round files found, look for initial attempts file
    initial_file = os.path.join(folder_path, "data_initial_attempts.json")
    if os.path.exists(initial_file):
        return initial_file
    
    raise FileNotFoundError(f"No math data file found in {folder_path}")

def get_sum_df(path):
    """Get sum dataframe for math_bench.py data format."""
    data_file = find_math_file(path)
    
    # Load pickle data
    with open(data_file, "rb") as f:
        data_store = pickle.load(f)
    
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
            attempts_by_round[attempt.round_idx].append(attempt.reward)
        
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
# Example usage (update these paths to your math experiment results):
# df_icrl = get_sum_df("/path/to/math/icrl/results/")
# df_reflexion = get_sum_df("/path/to/math/reflexion/results/")
# df_random = get_sum_df("/path/to/math/random_sampling/results/")

# Plot comparisons:
# plot_per_step_running_max(df_icrl, df_reflexion, df_random, 
#                          label_0='ICRL', label_1='Reflexion', label_2='Random Sampling')
# plot_per_step_gaussian_smoothed(df_icrl, df_reflexion, df_random, param=1,
#                                label_0='ICRL', label_1='Reflexion', label_2='Random Sampling', save=True)
