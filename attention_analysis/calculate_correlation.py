#!/usr/bin/env python3
"""
Calculate correlation between adjusted attention values and rewards
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import sys


def load_attention_data(json_path):
    """Load the attention analysis results from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def calculate_correlations(data):
    """Calculate correlation between adjusted attentions and rewards.

    Args:
        data: JSON data containing adjusted_attentions and test_rewards

    Returns:
        Dictionary with correlation statistics
    """
    # Extract data
    adjusted_attentions_all = data['adjusted_attentions']  # List of lists

    # Check if rewards are per-question or single list
    if 'test_rewards_per_question' in data:
        test_rewards_all = data['test_rewards_per_question']  # List of lists (one per question)
    else:
        # Fallback for old format
        test_rewards_all = [data['test_rewards']] * len(adjusted_attentions_all)

    # Flatten all adjusted attentions across questions and trials
    all_attentions = []
    all_rewards = []

    for q_idx, question_attentions in enumerate(adjusted_attentions_all):
        if question_attentions is None:
            continue

        test_rewards = test_rewards_all[q_idx]  # Get this question's rewards

        # Each question has 5 trials with corresponding rewards
        for i, attention in enumerate(question_attentions):
            if attention is not None:  # Skip None values
                all_attentions.append(attention)
                all_rewards.append(test_rewards[i])

    # Convert to numpy arrays
    all_attentions = np.array(all_attentions)
    all_rewards = np.array(all_rewards)

    print(f"Total data points: {len(all_attentions)}")
    print(f"Unique reward values: {np.unique(all_rewards)}")

    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(all_attentions, all_rewards)
    spearman_r, spearman_p = stats.spearmanr(all_attentions, all_rewards)

    # Calculate mean attention by reward group
    # Handle multiple reward values (1, 5, 10)
    unique_rewards = np.unique(all_rewards)
    reward_stats = {}

    for reward in unique_rewards:
        mask = all_rewards == reward
        reward_stats[reward] = {
            'mean': float(np.mean(all_attentions[mask])) if np.any(mask) else 0,
            'std': float(np.std(all_attentions[mask])) if np.any(mask) else 0,
            'count': int(np.sum(mask))  # Convert to regular int
        }

    # Compare low (1) and high (10) reward groups
    # Change these values to compare different groups
    comparison_low = 1   # Compare 1
    comparison_high = 10  # vs 10

    # Find the actual rewards to use (in case they don't exist)
    if comparison_low in unique_rewards:
        low_reward = comparison_low
    else:
        low_reward = min(unique_rewards)  # Fallback
        print(f"Warning: Reward {comparison_low} not found, using {low_reward}")

    if comparison_high in unique_rewards:
        high_reward = comparison_high
    else:
        high_reward = max(unique_rewards)  # Fallback
        print(f"Warning: Reward {comparison_high} not found, using {high_reward}")

    low_reward_mask = all_rewards == low_reward
    high_reward_mask = all_rewards == high_reward

    mean_attention_low = reward_stats[low_reward]['mean']
    mean_attention_high = reward_stats[high_reward]['mean']
    std_attention_low = reward_stats[low_reward]['std']
    std_attention_high = reward_stats[high_reward]['std']

    # T-test between selected reward groups (now 5 vs 10)
    if np.sum(high_reward_mask) > 0 and np.sum(low_reward_mask) > 0:
        t_stat, t_pval = stats.ttest_ind(
            all_attentions[high_reward_mask],
            all_attentions[low_reward_mask]
        )
        print(f"\nT-test comparing rewards {low_reward} vs {high_reward}")
    else:
        t_stat, t_pval = 0, 1.0
        print(f"\nInsufficient data for t-test between rewards {low_reward} and {high_reward}")

    results = {
        'num_data_points': int(len(all_attentions)),
        'num_questions': data['num_questions_processed'],
        'unique_rewards': unique_rewards.tolist(),
        'reward_stats': {str(k): v for k, v in reward_stats.items()},  # Convert keys to strings for JSON
        'pearson_correlation': float(pearson_r),
        'pearson_p_value': float(pearson_p),
        'spearman_correlation': float(spearman_r),
        'spearman_p_value': float(spearman_p),
        'comparison_rewards': [int(low_reward), int(high_reward)],  # Convert to regular ints
        'mean_attention_low_reward': float(mean_attention_low),  # Now this is reward 5
        'std_attention_low_reward': float(std_attention_low),
        'mean_attention_high_reward': float(mean_attention_high),  # This is reward 10
        'std_attention_high_reward': float(std_attention_high),
        'attention_difference': float(mean_attention_high - mean_attention_low),
        't_statistic': float(t_stat),
        't_test_p_value': float(t_pval),
        'all_attentions': all_attentions.tolist(),
        'all_rewards': all_rewards.tolist()
    }

    return results


def plot_correlation(results):
    """Create visualization of correlation between attention and rewards."""

    attentions = np.array(results['all_attentions'])
    rewards = np.array(results['all_rewards'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(rewards, attentions, alpha=0.5)
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Adjusted Attention')
    ax1.set_title(f'Attention vs Reward\nPearson r={results["pearson_correlation"]:.3f}, p={results["pearson_p_value"]:.3e}')
    ax1.set_xticks([1, 10])
    ax1.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(rewards, attentions, 1)
    p = np.poly1d(z)
    x_line = np.array([1, 10])
    ax1.plot(x_line, p(x_line), "r-", alpha=0.8, label='Trend line')
    ax1.legend()

    # Box plot - comparing 1 vs 10
    ax2 = axes[1]
    low_reward_attn = attentions[rewards == 1]
    high_reward_attn = attentions[rewards == 10]

    bp = ax2.boxplot([low_reward_attn, high_reward_attn],
                      labels=['Low Reward (1)', 'High Reward (10)'],
                      patch_artist=True)

    # Color the boxes
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')

    ax2.set_ylabel('Adjusted Attention')
    ax2.set_title(f'Attention Distribution by Reward\nt-test p={results["t_test_p_value"]:.3e}')
    ax2.grid(True, alpha=0.3)

    # Add mean markers
    ax2.scatter([1], [results['mean_attention_low_reward']],
                color='red', marker='D', s=100, zorder=5, label='Mean')
    ax2.scatter([2], [results['mean_attention_high_reward']],
                color='red', marker='D', s=100, zorder=5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('attention_reward_correlation.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: attention_reward_correlation.png")
    plt.show()


def print_results(results):
    """Print correlation results in a formatted way."""

    print("\n" + "="*60)
    print("CORRELATION ANALYSIS RESULTS")
    print("="*60)

    print(f"\nDataset Summary:")
    print(f"  - Number of questions: {results['num_questions']}")
    print(f"  - Total data points: {results['num_data_points']}")

    print(f"\nCorrelation Metrics:")
    print(f"  - Pearson correlation: r = {results['pearson_correlation']:.4f} (p = {results['pearson_p_value']:.4e})")
    print(f"  - Spearman correlation: ρ = {results['spearman_correlation']:.4f} (p = {results['spearman_p_value']:.4e})")

    print(f"\nGroup Statistics:")

    # Print all reward groups if available
    if 'reward_stats' in results:
        for reward, stats in sorted(results['reward_stats'].items()):
            print(f"  Reward {reward} attention:")
            print(f"    - Mean:  {stats['mean']:.6f}")
            print(f"    - Std:   {stats['std']:.6f}")
            print(f"    - Count: {stats['count']}")

    print(f"\n  Comparison (1 vs 10):")
    print(f"    - Reward 1 mean:  {results['mean_attention_low_reward']:.6f}")
    print(f"    - Reward 10 mean: {results['mean_attention_high_reward']:.6f}")
    print(f"    - Difference:     {results['attention_difference']:.6f}")

    print(f"\nStatistical Test:")
    print(f"  - T-statistic: {results['t_statistic']:.4f}")
    print(f"  - P-value: {results['t_test_p_value']:.4e}")

    # Interpretation
    print(f"\nInterpretation:")
    if results['pearson_p_value'] < 0.05:
        direction = "positive" if results['pearson_correlation'] > 0 else "negative"
        print(f"  ✓ Significant {direction} correlation between reward and attention (p < 0.05)")
    else:
        print(f"  ✗ No significant correlation between reward and attention (p >= 0.05)")

    if results['t_test_p_value'] < 0.05:
        if results['attention_difference'] > 0:
            print(f"  ✓ Significant difference: Reward 10 has higher attention than Reward 1 (p < 0.05)")
        else:
            print(f"  ✓ Significant difference: Reward 1 has higher attention than Reward 10 (p < 0.05)")
    else:
        print(f"  ✗ No significant difference between Reward 1 and Reward 10 groups (p >= 0.05)")

    print("="*60 + "\n")


def main():
    # Load the attention analysis results
    json_path = "attention_analysis_all_questions.json"

    try:
        data = load_attention_data(json_path)
        print(f"Loaded data from: {json_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {json_path}")
        print("Please run analyze_all_questions.py first to generate the data.")
        return

    # Calculate correlations
    results = calculate_correlations(data)

    # Save correlation results
    output_path = "correlation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Correlation results saved to: {output_path}")

    # Print results
    print_results(results)

    # Create visualization
    plot_correlation(results)


if __name__ == "__main__":
    main()