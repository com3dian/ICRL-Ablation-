#!/usr/bin/env python3
"""
Analyze attention patterns of Qwen3-32B model on large ICRL contexts.
This version analyzes multiple questions and computes average attention distributions.
"""

import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import gc

def load_icrl_prompts(json_path, num_questions=10, round_idx=79):
    """Load ICRL prompts for multiple questions."""
    print(f"Loading prompts from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    prompts = []
    for q_idx in range(min(num_questions, len(data))):
        # Use last round if round_idx is -1
        if round_idx == -1:
            actual_round_idx = len(data[q_idx]) - 1
        else:
            actual_round_idx = round_idx
            
        if actual_round_idx < len(data[q_idx]) and actual_round_idx >= 0:
            prompt = data[q_idx][actual_round_idx]['prompt']
            prompts.append(prompt)
            print(f"Loaded prompt {q_idx+1} (round {actual_round_idx}) with {len(prompt)} characters")
        else:
            print(f"Warning: Question {q_idx+1} doesn't have round {actual_round_idx}")
    
    return prompts

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer with attention outputs enabled."""
    print(f"Loading model {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set truncation to keep the END of the prompt (most recent trials)
    tokenizer.truncation_side = 'left'
    
    # Load model with attention outputs
    model_kwargs = {
        "output_attentions": True,
        "torch_dtype": torch.float16,  # Use fp16 for memory efficiency
        "device_map": "auto",  # Automatically distribute across GPUs
        "trust_remote_code": True
    }
    
    # Add specific settings for Qwen models
    if "Qwen" in model_name:
        model_kwargs["attn_implementation"] = "eager"  # Use eager attention for explicit weights
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    print(f"Model loaded on devices: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}")
    return model, tokenizer

def process_attention_weights(attention_tuple, aggregation='mean'):
    """Process attention weights from model output."""
    # Stack all layers: (num_layers, batch_size, num_heads, seq_len, seq_len)
    attention_stack = torch.stack(attention_tuple)
    
    # Average across layers and heads
    if aggregation == 'mean':
        # Mean across layers (dim=0) and heads (dim=2)
        aggregated = attention_stack.mean(dim=[0, 2])
    elif aggregation == 'max':
        # Max across layers and heads
        aggregated = attention_stack.max(dim=0)[0].max(dim=1)[0]
    
    # Remove batch dimension
    aggregated = aggregated.squeeze(0)
    
    return aggregated

def get_relative_attention_distribution(attention_matrix):
    """
    Get attention distribution as percentages for each position.
    Returns array of attention percentages for each position.
    """
    # Get attention from last token (what the model attends to when generating)
    last_token_attention = attention_matrix[-1, :].cpu().numpy()
    
    # Convert to percentages
    attention_percentages = (last_token_attention / last_token_attention.sum()) * 100
    
    return attention_percentages

def analyze_single_prompt(prompt, model, tokenizer, max_length=2048):
    """Analyze attention for a single prompt."""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    
    # Get model outputs with attention
    try:
        with torch.no_grad():
            torch.cuda.empty_cache()  # Clear cache before forward pass
            outputs = model(**inputs)
        
        # Process attention weights
        attention_matrix = process_attention_weights(outputs.attentions, aggregation='mean')
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"Memory error, reducing sequence length...")
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Reduce length and retry
        max_length = max_length // 4
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            outputs = model(**inputs)
        
        attention_matrix = process_attention_weights(outputs.attentions, aggregation='mean')
    
    # Get attention distribution
    attention_percentages = get_relative_attention_distribution(attention_matrix)
    
    return attention_percentages, input_ids.shape[1]

def compute_position_group_averages(all_attention_distributions):
    """Compute average attention for position groups across all questions."""
    all_position_groups = []
    
    for attention_dist in all_attention_distributions:
        seq_len = len(attention_dist)
        position_groups = {
            'start': float(attention_dist[:seq_len//10].mean()),
            'early': float(attention_dist[seq_len//10:seq_len//4].mean()),
            'middle': float(attention_dist[seq_len//4:3*seq_len//4].mean()),
            'late': float(attention_dist[3*seq_len//4:9*seq_len//10].mean()),
            'end': float(attention_dist[9*seq_len//10:].mean())
        }
        all_position_groups.append(position_groups)
    
    # Compute averages
    avg_position_groups = {}
    for group in ['start', 'early', 'middle', 'late', 'end']:
        values = [pg[group] for pg in all_position_groups]
        avg_position_groups[group] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return avg_position_groups, all_position_groups

def visualize_average_attention(all_attention_distributions, avg_position_groups, all_position_groups, output_prefix='attention_avg'):
    """Create visualizations of average attention patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Average attention by position group with error bars
    ax = axes[0, 0]
    groups = list(avg_position_groups.keys())
    means = [avg_position_groups[g]['mean'] for g in groups]
    stds = [avg_position_groups[g]['std'] for g in groups]
    
    bars = ax.bar(groups, means, yerr=stds, capsize=5, alpha=0.7, color='blue')
    ax.set_xlabel('Position Group')
    ax.set_ylabel('Average Attention (%)')
    ax.set_title('Average Attention by Position Group (with std error)')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}%', ha='center', va='bottom')
    
    # 2. Attention distribution across positions (averaged)
    ax = axes[0, 1]
    # Average across all questions
    max_len = max(len(ad) for ad in all_attention_distributions)
    padded_distributions = []
    for ad in all_attention_distributions:
        if len(ad) < max_len:
            # Pad with zeros if needed
            padded = np.pad(ad, (0, max_len - len(ad)), 'constant')
        else:
            padded = ad[:max_len]
        padded_distributions.append(padded)
    
    avg_distribution = np.mean(padded_distributions, axis=0)
    positions = np.arange(len(avg_distribution))
    ax.plot(positions, avg_distribution, alpha=0.7)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Average Attention (%)')
    ax.set_title('Average Attention Distribution Across Positions')
    
    # 3. Heatmap of all questions
    ax = axes[1, 0]
    # Create heatmap data (questions x position groups)
    heatmap_data = []
    for pg in all_position_groups:
        heatmap_data.append([pg['start'], pg['early'], pg['middle'], pg['late'], pg['end']])
    
    im = ax.imshow(heatmap_data, cmap='hot', aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['start', 'early', 'middle', 'late', 'end'])
    ax.set_yticks(range(len(heatmap_data)))
    ax.set_yticklabels([f'Q{i+1}' for i in range(len(heatmap_data))])
    ax.set_xlabel('Position Group')
    ax.set_ylabel('Question')
    ax.set_title('Attention Heatmap by Question and Position')
    plt.colorbar(im, ax=ax)
    
    # 4. Box plot of position groups
    ax = axes[1, 1]
    box_data = []
    box_labels = []
    for group in groups:
        values = [pg[group] for pg in all_position_groups]
        box_data.append(values)
        box_labels.append(group)
    
    ax.boxplot(box_data, labels=box_labels)
    ax.set_xlabel('Position Group')
    ax.set_ylabel('Attention (%)')
    ax.set_title('Attention Distribution by Position Group')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_prefix}_distribution.png")

def save_detailed_results(all_attention_distributions, avg_position_groups, all_position_groups, 
                         token_counts, output_prefix='attention_avg'):
    """Save detailed analysis results."""
    
    # Summary statistics
    summary = {
        'num_questions_analyzed': len(all_attention_distributions),
        'average_tokens_per_question': float(np.mean(token_counts)),
        'token_counts': token_counts,
        'average_position_groups': avg_position_groups,
        'per_question_position_groups': all_position_groups
    }
    
    # Save summary
    with open(f'{output_prefix}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed per-position averages
    # Average across all questions with padding
    max_len = max(len(ad) for ad in all_attention_distributions)
    padded_distributions = []
    for ad in all_attention_distributions:
        if len(ad) < max_len:
            padded = np.pad(ad, (0, max_len - len(ad)), 'constant')
        else:
            padded = ad[:max_len]
        padded_distributions.append(padded)
    
    avg_dist = np.mean(padded_distributions, axis=0)
    df = pd.DataFrame({
        'position': range(len(avg_dist)),
        'avg_attention_percent': avg_dist,
        'relative_position': np.arange(len(avg_dist)) / len(avg_dist)
    })
    df.to_csv(f'{output_prefix}_position_details.csv', index=False)
    
    print(f"Saved results to {output_prefix}_summary.json and {output_prefix}_position_details.csv")

def main():
    # Configuration
    # Allow passing json_path as argument or use default
    import sys
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'attention_avg'
    else:
        json_path = '/sfs/weka/scratch/ks8vf/code_submission/ICL/creative_writing_api/Qwen/Qwen3-32B/ICRL_different_prompts_evalnum_100_n_100/output_list.json'
        output_prefix = 'attention_avg'
    
    model_name = "Qwen/Qwen3-32B"
    max_length = 2048  # Conservative to avoid OOM
    num_questions = 10  # Analyze first 10 questions
    
    # Load all prompts
    # Use last round (-1) for self-refine, round 79 for ICRL
    round_idx = -1 if "self_refine" in json_path else 79
    prompts = load_icrl_prompts(json_path, num_questions=num_questions, round_idx=round_idx)
    print(f"\nLoaded {len(prompts)} prompts for analysis")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Analyze each prompt
    all_attention_distributions = []
    all_position_groups = []
    token_counts = []
    
    print("\nAnalyzing attention patterns...")
    for i, prompt in enumerate(tqdm(prompts, desc="Processing questions")):
        print(f"\nAnalyzing question {i+1}/{len(prompts)}...")
        print(f"Using LEFT truncation - keeping the LAST {max_length} tokens")
        
        attention_dist, num_tokens = analyze_single_prompt(prompt, model, tokenizer, max_length)
        all_attention_distributions.append(attention_dist)
        token_counts.append(num_tokens)
        
        # Clear cache between questions
        torch.cuda.empty_cache()
        gc.collect()
    
    # Compute averages
    print("\nComputing average attention patterns...")
    avg_position_groups, all_position_groups_list = compute_position_group_averages(all_attention_distributions)
    
    # Print summary
    method_name = "Self-Refine" if "self_refine" in output_prefix else "ICRL"
    print(f"\n=== Average Attention Analysis Summary ({method_name}) ===")
    print(f"Model: {model_name}")
    print(f"Questions analyzed: {len(prompts)}")
    print(f"Average tokens per question: {np.mean(token_counts):.0f}")
    print("\nAverage attention by position group:")
    for group, stats in avg_position_groups.items():
        print(f"  {group}: {stats['mean']:.4f}% (±{stats['std']:.4f}%)")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_average_attention(all_attention_distributions, avg_position_groups, all_position_groups_list, 
                               output_prefix=output_prefix)
    
    # Save results
    print("\nSaving results...")
    save_detailed_results(all_attention_distributions, avg_position_groups, 
                         all_position_groups_list, token_counts,
                         output_prefix=output_prefix)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()