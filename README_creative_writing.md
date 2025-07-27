# Creative Writing Experiments with ICRL

This guide explains how to run creative writing experiments using In-Context Reinforcement Learning (ICRL) and compare with baseline methods.

## Prerequisites

1. Install required packages:
```bash
pip install -r requirements_creative_writing.txt
```

2. Set up API keys:
```bash
export OPENAI_API_KEY=sk-C8z62BDhmo4EW1bqOn2TTmdFR29ocUeZXLExkdmGS1T3BlbkFJQcA3zOug-aNTm98KC0Wjsv549b3OgxEGn9TKJknXMA
```

## Running Experiments

### 1. Main ICRL Method

The main ICRL implementation is in `llm_creative_writing_api.py`. 

**Configuration**: Edit the flags at the top of the file (lines 22-42) to control the experiment:
```python
api_eval = False  # Set to True to use OpenAI API, False for vLLM
ICRL = 1  # Enable ICRL method
exploitation_only = 0  # Set to 1 for exploitation-only ablation
exploration_only_no_reward = 0  # Set to 1 for exploration without rewards
# ... other flags
```

**Run the experiment**:
```bash
python llm_creative_writing_api.py
```

### 2. Baseline Methods

Run each baseline with its dedicated script:

```bash
# Reflexion baseline
python llm_creative_writing_api_reflexion.py

# Self-Refine baseline
python llm_creative_writing_api_self-refine.py

# Chain-of-Thought (CoT) baseline
python llm_creative_writing_api_CoT.py

# Rejection Sampling (Best-of-N)
# For this, set rejection_sampling=1 in llm_creative_writing_api.py
```

### 3. ICRL Ablations

To run ablations, modify the flags in `llm_creative_writing_api.py`:

- **Exploitation-only**: Set `exploitation_only = 1`
- **Exploration-only (no reward)**: Set `exploration_only_no_reward = 1`
- **Exploration and exploitation**: Set `exploration_and_exploitation = 1`
- **No reward signal**: Set `no_reward = 1`
- **Zero reward**: Set `zero_reward = 1`

## Generating Data for AlpacaEval

After running experiments, use `gen_data_for_AlpacaEval.py` to prepare the outputs for evaluation:

```bash
python gen_data_for_AlpacaEval.py
```

This script will:
1. Read the experimental outputs from pickle files
2. Format them according to AlpacaEval requirements
3. Generate two JSON files for comparison

## Running AlpacaEval Pairwise Evaluation

AlpacaEval performs pairwise comparisons between two methods to determine which produces better outputs.

### Install AlpacaEval
```bash
pip install alpaca-eval
```

### Run Pairwise Evaluation

1. Ensure your OpenAI API key is set:
```bash
export OPENAI_API_KEY=sk-C8z62BDhmo4EW1bqOn2TTmdFR29ocUeZXLExkdmGS1T3BlbkFJQcA3zOug-aNTm98KC0Wjsv549b3OgxEGn9TKJknXMA
```

2. Run the evaluation comparing two methods:
```bash
alpaca_eval --model_outputs ours50_responses.json --reference_outputs bon2_responses.json
```

Where:
- `ours50_responses.json`: Your ICRL method outputs
- `bon2_responses.json`: Baseline method outputs (e.g., Best-of-N)

### Understanding AlpacaEval Output

AlpacaEval will produce:
- **Win rate**: Percentage of times your method was preferred
- **Standard error**: Statistical uncertainty in the win rate
- **Detailed comparisons**: Individual judgments for each example

Results are typically saved in the `AlpacaEval/` directory.

## Output Files

The experiments generate several output files:
- `intermediate_round_creative_writing_*.pkl`: Raw experimental results
- `*_responses.json`: Formatted outputs for AlpacaEval
- `AlpacaEval/*/leaderboard.csv`: Evaluation results

## Tips for Running Experiments

1. **GPU Usage**: When using vLLM (api_eval=False), ensure you have sufficient GPU memory. The code uses `tensor_parallel_size=2` for multi-GPU setups.

2. **API Costs**: When using OpenAI API (api_eval=True), be aware of token usage and costs.

3. **Batch Processing**: The code uses ThreadPoolExecutor for parallel processing. Adjust the number of workers if needed.

4. **Monitoring Progress**: All scripts use tqdm for progress bars. Watch the console output to track experiment progress.

## Troubleshooting

- **Out of Memory**: Reduce `max_model_len` in vLLM configuration or use fewer samples
- **API Errors**: Check your API key and rate limits
- **Missing Files**: Ensure all required modules (like `tot`) are properly installed

## Example Workflow

```bash
# 1. Run ICRL method
python llm_creative_writing_api.py

# 2. Run a baseline for comparison
python llm_creative_writing_api_reflexion.py

# 3. Generate evaluation files
python gen_data_for_AlpacaEval.py

# 4. Run pairwise evaluation
export OPENAI_API_KEY=sk-C8z62BDhmo4EW1bqOn2TTmdFR29ocUeZXLExkdmGS1T3BlbkFJQcA3zOug-aNTm98KC0Wjsv549b3OgxEGn9TKJknXMA
alpaca_eval --model_outputs icrl_responses.json --reference_outputs reflexion_responses.json
```