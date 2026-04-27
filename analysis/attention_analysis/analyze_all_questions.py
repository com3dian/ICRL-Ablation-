#!/usr/bin/env python3
"""
Extract and analyze attention across multiple questions from ICRL output.
Optimized: one forward pass extracts ALL layer/head combos at once.
Supports two-GPU parallelism by splitting questions.
"""

import json
import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from visualize_qwen3_attention import Qwen3AttentionVisualizer


def load_and_analyze_json(json_path, num_questions):
    """Load the JSON file and extract the last iteration for each question."""
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    prompt_texts = []
    generated_texts = []
    for question_idx in range(min(num_questions, len(data))):
        prompt_texts.append(data[question_idx][-1]['prompt'])
        generated_texts.append(data[question_idx][-1]['generated_text'])

    print(f"Loaded {len(prompt_texts)} questions")
    return prompt_texts, generated_texts


def extract_responses(json_path, question_idx=0):
    """Extract trials and rewards from the JSON data for a specific question."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    trials = []
    rewards = []

    if question_idx >= len(data):
        print(f"Question {question_idx} not found (only {len(data)} questions available)")
        return trials, rewards

    for i, iteration in enumerate(data[question_idx]):
        generated_text = iteration['generated_text']
        reward = iteration['reward']
        if len(generated_text.strip()) < 10:
            continue
        trials.append(generated_text)
        rewards.append(reward)

    return trials, rewards


def find_trial_boundaries(texts, rewards, tokens, tokenizer):
    """Find token boundaries for each trial in the concatenated sequence."""
    trial_boundaries = []
    current_start = 0

    for i, (text, reward) in enumerate(zip(texts, rewards)):
        trial_text = f"<attempt>{text}**Reward**: {reward}</attempt>"
        trial_tokens = tokenizer(trial_text, return_tensors="pt", truncation=False)
        expected_num_tokens = trial_tokens['input_ids'].shape[1]
        current_end = min(current_start + expected_num_tokens, len(tokens))
        trial_boundaries.append((current_start, current_end))
        current_start = current_end
        if current_end >= len(tokens):
            if i + 1 < len(texts):
                tqdm.write(f"  WARNING: Sequence truncated at trial {i+1}/{len(texts)}")
            break

    return trial_boundaries


def extract_trial_attentions(attention_head_np, trial_boundaries, num_texts):
    """Extract average attention per trial from a single head's attention matrix."""
    avg_attentions = []
    for i, (start, end) in enumerate(trial_boundaries):
        if end <= start:
            avg_attentions.append(None)
            continue
        last_token_idx = end - 1
        trial_attention = attention_head_np[last_token_idx, start:end]
        avg_attentions.append(float(np.mean(trial_attention)))

    # Fill remaining with None if truncated
    for _ in range(len(trial_boundaries), num_texts):
        avg_attentions.append(None)

    return avg_attentions


def forward_pass_all_heads(texts, rewards, visualizer, target_layers):
    """One forward pass, extract all heads from target layers only (memory efficient).

    Returns:
        dict mapping (layer_idx, head_idx) -> list of avg attentions per trial
    """
    # Build concatenated prompt
    prompt = ""
    for text, reward in zip(texts, rewards):
        prompt += f"<attempt>{text}**Reward**: {reward}</attempt>"

    # ONE forward pass — only target layers use eager attention
    max_length = 131072
    attentions, tokens = visualizer.get_attention_weights(
        prompt, max_length=max_length, target_layers=target_layers
    )

    # Find trial boundaries (same for all heads since prompt is identical)
    trial_boundaries = find_trial_boundaries(
        texts, rewards, tokens, visualizer.tokenizer
    )

    # Extract every target layer/head
    # attentions is a dict: {layer_idx: tensor [1, num_heads, seq_len, seq_len]}
    results = {}
    for layer_idx in target_layers:
        attention_layer = attentions[layer_idx]  # [1, num_heads, seq_len, seq_len]
        num_heads = attention_layer.shape[1]

        for head_idx in range(num_heads):
            attention_head = attention_layer[0, head_idx].cpu().float().numpy()
            avg_atts = extract_trial_attentions(
                attention_head, trial_boundaries, len(texts)
            )
            results[(layer_idx, head_idx)] = avg_atts

    # Free GPU memory
    del attentions
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze attention across questions (all heads)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU index to use (default: auto)')
    parser.add_argument('--start-q', type=int, default=0,
                        help='Start question index (inclusive)')
    parser.add_argument('--end-q', type=int, default=100,
                        help='End question index (exclusive)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output JSONs')
    parser.add_argument('--target-layers', type=int, nargs='+', default=[-1, -2, -3, -4],
                        help='Layer indices to analyze (default: -1 -2 -3 -4)')
    parser.add_argument('--json-path', type=str, required=True,
                        help='Path to ICRL output_list.json')
    parser.add_argument('--num-questions', type=int, default=100,
                        help='Number of questions to process (default: 100)')
    args = parser.parse_args()

    # Pin to specific GPU if requested
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    json_path = args.json_path
    num_questions = args.num_questions
    num_trials = 5
    target_layers = args.target_layers

    # Load all questions
    prompt_texts, generated_texts = load_and_analyze_json(json_path, num_questions)
    if not prompt_texts:
        print("Failed to extract data from JSON")
        return

    # Initialize model ONCE
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print(f"Questions: {args.start_q} to {args.end_q}")
    print(f"Target layers: {target_layers}")
    print("="*60)
    visualizer = Qwen3AttentionVisualizer(
        model_name="Qwen/Qwen3-32B",
        load_in_4bit=True
    )
    print("Model loaded!")

    # Structure: all_results[(layer, head)] = {lists of per-question data}
    all_results = {}

    # Process assigned questions
    for question_idx in tqdm(range(args.start_q, min(args.end_q, len(prompt_texts))),
                             desc="Processing questions"):
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"QUESTION {question_idx + 1}/{len(prompt_texts)}")
        tqdm.write(f"{'='*60}")

        trials, original_rewards = extract_responses(json_path, question_idx=question_idx)
        if len(trials) < num_trials:
            tqdm.write(f"  Skipping: only {len(trials)} trials (need {num_trials})")
            continue

        # Random synthetic rewards for test pass
        test_rewards = np.random.choice([1, 10], size=num_trials).tolist()
        baseline_rewards = [1, 1, 1, 1, 1]

        # FORWARD PASS 1: test rewards (extracts ALL heads at once)
        tqdm.write(f"  Test pass (rewards={test_rewards})...")
        test_results = forward_pass_all_heads(
            trials[:num_trials], test_rewards, visualizer, target_layers
        )

        # FORWARD PASS 2: baseline (extracts ALL heads at once)
        tqdm.write(f"  Baseline pass (rewards={baseline_rewards})...")
        baseline_results = forward_pass_all_heads(
            trials[:num_trials], baseline_rewards, visualizer, target_layers
        )

        # Accumulate per layer/head
        for key in test_results:
            if key not in all_results:
                all_results[key] = {
                    "test_attentions": [],
                    "baseline_attentions": [],
                    "adjusted_attentions": [],
                    "test_rewards": []
                }

            test_att = np.array(test_results[key])
            base_att = np.array(baseline_results[key])
            adjusted = (test_att - base_att).tolist()

            all_results[key]["test_attentions"].append(test_results[key])
            all_results[key]["baseline_attentions"].append(baseline_results[key])
            all_results[key]["adjusted_attentions"].append(adjusted)
            all_results[key]["test_rewards"].append(test_rewards)

    # Save one JSON per layer/head (same format as before for calculate_correlation.py)
    os.makedirs(args.output_dir, exist_ok=True)

    for (layer_idx, head_idx), data in all_results.items():
        output = {
            "num_questions_processed": len(data["adjusted_attentions"]),
            "num_trials_per_question": num_trials,
            "test_rewards_per_question": data["test_rewards"],
            "baseline_rewards": [1, 1, 1, 1, 1],
            "raw_attentions": data["test_attentions"],
            "baseline_attentions": data["baseline_attentions"],
            "adjusted_attentions": data["adjusted_attentions"],
        }
        filename = os.path.join(
            args.output_dir,
            f"attention_analysis_layer_{layer_idx}_head_{head_idx}.json"
        )
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Questions processed: {args.start_q} to {min(args.end_q, len(prompt_texts))}")
    print(f"Layer/head combos saved: {len(all_results)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
