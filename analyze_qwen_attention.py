#!/usr/bin/env python3
"""
Analyze attention patterns of Qwen3-32B model on large ICRL contexts.
This script loads the model, processes a long prompt, and identifies
which positions receive the most attention.
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

def load_icrl_prompt(json_path, question_idx=0, round_idx=79):
    """Load the ICRL prompt from the output JSON file."""
    print(f"Loading prompt from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get the specified question and round
    prompt = data[question_idx][round_idx]['prompt']
    print(f"Loaded prompt with {len(prompt)} characters")
    return prompt

def load_model_and_tokenizer(model_name="gpt2-large"):
    """Load GPT2 model and tokenizer with attention outputs enabled."""
    print(f"Loading model {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with attention outputs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        torch_dtype=torch.float16,  # Use fp16 for memory efficiency
        device_map="auto",  # Automatically distribute across GPUs
        trust_remote_code=True
    )
    
    print(f"Model loaded on devices: {model.hf_device_map}")
    return model, tokenizer

def process_attention_weights(attention_tuple, aggregation='mean'):
    """
    Process attention weights from model output.
    
    Args:
        attention_tuple: Tuple of attention tensors from each layer
        aggregation: How to aggregate across layers and heads ('mean', 'max')
    
    Returns:
        Aggregated attention matrix of shape (seq_len, seq_len)
    """
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

def get_top_attention_positions(attention_matrix, top_percent=0.1):
    """
    Get the top X% of attention positions.
    
    Args:
        attention_matrix: Attention matrix of shape (seq_len, seq_len)
        top_percent: Percentage of top positions to return (0.1 = 10%)
    
    Returns:
        List of (position, attention_score) tuples
    """
    # Get attention scores for the last token (what the model attends to when generating)
    last_token_attention = attention_matrix[-1, :].cpu().numpy()
    
    # Calculate threshold for top X%
    num_positions = int(len(last_token_attention) * top_percent)
    threshold = np.percentile(last_token_attention, 100 * (1 - top_percent))
    
    # Get top positions
    top_indices = np.argsort(last_token_attention)[-num_positions:][::-1]
    top_scores = last_token_attention[top_indices]
    
    return list(zip(top_indices, top_scores))

def analyze_attention_patterns(attention_matrix, tokens, top_positions):
    """Analyze patterns in attention distribution."""
    seq_len = attention_matrix.shape[0]
    
    # Get attention from last token
    last_token_attention = attention_matrix[-1, :].cpu().numpy()
    
    # Analyze position bias
    position_groups = {
        'start': last_token_attention[:seq_len//10].mean(),
        'early': last_token_attention[seq_len//10:seq_len//4].mean(),
        'middle': last_token_attention[seq_len//4:3*seq_len//4].mean(),
        'late': last_token_attention[3*seq_len//4:9*seq_len//10].mean(),
        'end': last_token_attention[9*seq_len//10:].mean()
    }
    
    # Find trial boundaries in the text
    trial_positions = []
    for i, token in enumerate(tokens):
        if '<trial>' in token or '**Task Query**' in token:
            trial_positions.append(i)
    
    return {
        'position_groups': position_groups,
        'trial_positions': trial_positions,
        'num_trials': len(trial_positions)
    }

def visualize_attention(attention_matrix, top_positions, tokens, output_prefix='attention'):
    """Create visualizations of attention patterns."""
    seq_len = attention_matrix.shape[0]
    last_token_attention = attention_matrix[-1, :].cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Attention distribution histogram
    ax = axes[0, 0]
    ax.hist(last_token_attention, bins=100, alpha=0.7, color='blue')
    ax.axvline(np.percentile(last_token_attention, 90), color='red', 
               linestyle='--', label='90th percentile')
    ax.set_xlabel('Attention Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Attention Scores')
    ax.legend()
    
    # 2. Attention by position
    ax = axes[0, 1]
    ax.scatter(range(seq_len), last_token_attention, alpha=0.5, s=1)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Attention Score')
    ax.set_title('Attention Score by Position')
    
    # 3. Top attention positions
    ax = axes[1, 0]
    top_pos, top_scores = zip(*top_positions[:50])  # Show top 50
    ax.bar(range(len(top_pos)), top_scores)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Attention Score')
    ax.set_title('Top 50 Attention Scores')
    
    # 4. Attention heatmap (sample)
    ax = axes[1, 1]
    # Sample evenly from the sequence for visualization
    sample_size = min(100, seq_len)
    sample_indices = np.linspace(0, seq_len-1, sample_size, dtype=int)
    sampled_attention = attention_matrix[sample_indices][:, sample_indices].cpu().numpy()
    
    im = ax.imshow(sampled_attention, cmap='hot', interpolation='nearest')
    ax.set_xlabel('Token Position (sampled)')
    ax.set_ylabel('Token Position (sampled)')
    ax.set_title('Attention Heatmap (sampled)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_prefix}_distribution.png")

def save_results(top_positions, tokens, analysis_results, output_prefix='attention'):
    """Save analysis results to files."""
    
    # Create detailed results
    results_data = []
    for pos, score in top_positions:
        # Get context around the position
        context_start = max(0, pos - 20)
        context_end = min(len(tokens), pos + 20)
        context_tokens = tokens[context_start:context_end]
        
        results_data.append({
            'position': int(pos),
            'score': float(score),
            'token': tokens[pos] if pos < len(tokens) else 'N/A',
            'context': ''.join(context_tokens),
            'relative_position': pos / len(tokens)
        })
    
    # Save to CSV
    df = pd.DataFrame(results_data)
    df.to_csv(f'{output_prefix}_top_positions.csv', index=False)
    
    # Save full analysis results
    analysis_results['num_top_positions'] = len(top_positions)
    analysis_results['total_tokens'] = len(tokens)
    
    with open(f'{output_prefix}_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Saved results to {output_prefix}_top_positions.csv and {output_prefix}_analysis_results.json")

def main():
    # Configuration
    json_path = '/sfs/weka/scratch/ks8vf/code_submission/ICL/creative_writing_api/Qwen/Qwen3-32B/ICRL_different_prompts_evalnum_100_n_100/output_list.json'
    model_name = "gpt2-large"
    max_length = 1024  # GPT2 context window
    
    # Load the ICRL prompt
    prompt = load_icrl_prompt(json_path)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Tokenize the prompt
    print("Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    
    print(f"Tokenized to {input_ids.shape[1]} tokens (truncated to {max_length} if necessary)")
    
    # Get model outputs with attention
    print("Running model forward pass...")
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process attention weights
        print("Processing attention weights...")
        attention_matrix = process_attention_weights(outputs.attentions, aggregation='mean')
    except torch.cuda.OutOfMemoryError:
        print("Out of memory! Trying with reduced sequence length...")
        # Reduce to half the length and retry
        max_length = max_length // 2
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"]
        print(f"Retrying with {input_ids.shape[1]} tokens")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        attention_matrix = process_attention_weights(outputs.attentions, aggregation='mean')
    
    # Get top 10% attention positions
    top_positions = get_top_attention_positions(attention_matrix, top_percent=0.1)
    print(f"Found {len(top_positions)} positions in top 10% of attention")
    
    # Convert token IDs to tokens for analysis
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Analyze attention patterns
    print("Analyzing attention patterns...")
    analysis_results = analyze_attention_patterns(attention_matrix, tokens, top_positions)
    
    # Print summary
    print("\n=== Attention Analysis Summary ===")
    print(f"Total tokens analyzed: {len(tokens)}")
    print(f"Top 10% positions: {len(top_positions)}")
    print("\nAttention by position group:")
    for group, score in analysis_results['position_groups'].items():
        print(f"  {group}: {score:.4f}")
    print(f"\nNumber of trials detected: {analysis_results['num_trials']}")
    
    # Top 10 attended positions
    print("\nTop 10 most attended positions:")
    for i, (pos, score) in enumerate(top_positions[:10]):
        token = tokens[pos] if pos < len(tokens) else 'N/A'
        print(f"  {i+1}. Position {pos} (score: {score:.4f}): '{token}'")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_attention(attention_matrix, top_positions, tokens)
    
    # Save results
    print("\nSaving results...")
    save_results(top_positions, tokens, analysis_results)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()