#!/usr/bin/env python3
"""
Plot stacked bar chart of significant reward-sensitive attention heads
across Qwen3-32B layers.
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "results_all_layers")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "fig_significant_heads_bar.png")

LAYERS = list(range(-1, -33, -1))  # -1 to -32
NUM_HEADS = 64
ALPHA = 0.05


def compute_correlation(filepath):
    """Compute Pearson r and p-value from one layer/head JSON."""
    with open(filepath) as f:
        data = json.load(f)

    adjusted = data["adjusted_attentions"]  # 100 questions x 5 trials
    rewards = data["test_rewards_per_question"]  # 100 questions x 5 trials

    all_att = []
    all_rew = []
    for q_idx, q_att in enumerate(adjusted):
        if q_att is None:
            continue
        for i, a in enumerate(q_att):
            if a is not None:
                all_att.append(a)
                all_rew.append(rewards[q_idx][i])

    if len(all_att) < 3:
        return 0.0, 1.0

    r, p = stats.pearsonr(all_att, all_rew)
    return r, p


def main():
    # Collect results: for each layer, count positive-significant and negative-significant heads
    pos_counts = []  # positive correlation, p < alpha
    neg_counts = []  # negative correlation, p < alpha

    for layer in tqdm(LAYERS, desc="Processing layers"):
        n_pos = 0
        n_neg = 0
        for head in range(NUM_HEADS):
            filepath = os.path.join(DATA_DIR, f"attention_analysis_layer_{layer}_head_{head}.json")
            if not os.path.exists(filepath):
                continue
            r, p = compute_correlation(filepath)
            if p < ALPHA:
                if r > 0:
                    n_pos += 1
                else:
                    n_neg += 1
        pos_counts.append(n_pos)
        neg_counts.append(n_neg)

    pos_counts = np.array(pos_counts)
    neg_counts = np.array(neg_counts)
    total_counts = pos_counts + neg_counts

    # Plot
    fig, ax = plt.subplots(figsize=(16, 5))

    x = np.arange(len(LAYERS))
    labels = [str(l) for l in LAYERS]

    ax.bar(x, neg_counts, color="#c75544", label="Negative (attend to low-reward trials)")
    ax.bar(x, pos_counts, bottom=neg_counts, color="#3889b7", label="Positive (attend to high-reward trials)")

    # Chance level line
    chance = NUM_HEADS * ALPHA  # 64 * 0.05 = 3.2
    ax.axhline(y=chance, color="black", linestyle="--", linewidth=1, label=f"Chance level (p = {ALPHA})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Layer (relative to output; \u22121 = last layer)", fontsize=12)
    ax.set_ylabel("Number of significant heads (out of 64)", fontsize=12)
    ax.set_title("Reward-sensitive attention heads across Qwen3-32B layers", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, max(total_counts) + 5)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH}")

    # Print summary
    total_sig = int(pos_counts.sum() + neg_counts.sum())
    print(f"\nTotal significant heads: {total_sig} / {len(LAYERS) * NUM_HEADS} "
          f"({100 * total_sig / (len(LAYERS) * NUM_HEADS):.1f}%)")
    print(f"Positive: {int(pos_counts.sum())}, Negative: {int(neg_counts.sum())}")


if __name__ == "__main__":
    main()
