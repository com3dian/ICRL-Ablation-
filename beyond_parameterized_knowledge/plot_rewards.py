import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_rewards(json_file):
    # Load the output data
    with open(json_file, 'r') as f:
        output_list = json.load(f)

    n_rounds = len(output_list[0])  # Number of rounds
    n_samples = len(output_list)     # Number of samples

    # Initialize arrays for storing rewards
    avg_rewards_per_round = []
    running_max_rewards_per_round = []

    # Process each round
    for round_idx in range(n_rounds):
        round_rewards = []
        running_max_rewards = []

        # Process each sample
        for sample_outputs in output_list:
            # Get reward for current round
            current_reward = sample_outputs[round_idx]["reward"]
            round_rewards.append(current_reward)

            # Calculate running max up to current round
            rewards_up_to_now = [sample_outputs[i]["reward"] for i in range(round_idx + 1)]
            running_max = max(rewards_up_to_now)
            running_max_rewards.append(running_max)

        # Calculate averages across all samples
        avg_rewards_per_round.append(np.mean(round_rewards))
        running_max_rewards_per_round.append(np.mean(running_max_rewards))

    # Create the plot
    plt.figure(figsize=(12, 6))
    rounds = np.arange(1, n_rounds + 1)

    # Plot average reward per round
    plt.plot(rounds, avg_rewards_per_round, 'b-', label='Average Reward', linewidth=2, alpha=0.7)

    # Plot running max average
    plt.plot(rounds, running_max_rewards_per_round, 'r-', label='Average of Running Max', linewidth=2, alpha=0.7)

    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Reward Progress Over 40 Rounds', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig('reward_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print(f"Final average reward: {avg_rewards_per_round[-1]:.2f}")
    print(f"Final running max average: {running_max_rewards_per_round[-1]:.2f}")
    print(f"Maximum average reward reached: {max(avg_rewards_per_round):.2f} at round {np.argmax(avg_rewards_per_round) + 1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str,
                       default='/sfs/weka/scratch/ks8vf/code_submission/ICL/beyond_parameterized_knowledge/gpt-4.1-mini/ICRL_rolling_window_evalnum_10_n_40/output_list.json',
                       help='Path to output_list.json file')
    args = parser.parse_args()

    plot_rewards(args.json_file)