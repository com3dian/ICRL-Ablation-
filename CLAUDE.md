# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements "Reward Is Enough: LLMs Are In-Context Reinforcement Learners" - a research project demonstrating in-context reinforcement learning (ICRL) across three domains:

1. **Game of 24**: Mathematical puzzle solving where players use arithmetic operations on 4 numbers to reach 24
2. **Creative Writing**: Text generation tasks with quality evaluation
3. **ScienceWorld**: Interactive text-based science simulation environment

## Architecture

The codebase is organized into three main experiment domains with parallel implementations:

### Core Implementation Pattern
Each domain follows a consistent structure:
- **Main ICRL file**: `llm_{domain}_api.py` - Primary implementation with configurable flags
- **Baseline implementations**: Separate files for each comparison method (reflexion, self-refine, CoT, rejection sampling)
- **Configuration flags**: At the top of each main file, control which method/ablation to run

### Key Configuration Flags (in main API files)
- `ICRL`: Enable/disable main ICRL method
- `exploration_only`: Run exploration-only ablation  
- `exploitation_only`: Run exploitation-only ablation
- `no_reward_exploration`: Exploration without reward signal
- `exploration_or_exploitation`: Alternative exploration strategy
- `rejection_sampling`: Enable Best-of-N baseline
- `api_eval`: Use OpenAI API vs local models

### ScienceWorld Domain
- Uses `sciworld.py` with OmegaConf for configuration
- Integrates external ScienceWorld environment via `sciworld_armap/` module
- Supports multiple environments and ablations through command-line args

## Common Development Commands

### Environment Setup
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Game of 24 Experiments
```bash
# Main ICRL method
python llm_game24_api.py

# Baselines
python llm_game24_api_reflexion.py
python llm_game24_api_self-refine.py  
python llm_game24_api_rejection.py
python llm_game24_api_CoT.py
```

### Creative Writing Experiments  
```bash
# Main ICRL method
python llm_creative_writing_api.py

# Baselines (same pattern as Game of 24)
python llm_creative_writing_api_reflexion.py
# ... etc
```

### ScienceWorld Experiments
```bash
# ICRL method
python3 sciworld.py icrl_mode=ICRL num_envs=29

# Ablations
python3 sciworld.py icrl_mode=ICRL num_envs=29 explore_only=true

# Baselines
python3 sciworld.py icrl_mode=RANDOM_SAMPLING num_envs=29 max_env_steps=200
```

## Important Configuration Notes

### API Keys
- OpenAI API key must be configured in each main file: `client = OpenAI(api_key="Your_API_Key")`
- For ScienceWorld, API key should be set via environment variable or .env file

### Method Selection
- **Game of 24 & Creative Writing**: Edit flags at the top of the main API file before running
- **ScienceWorld**: Use command-line arguments with OmegaConf syntax

### ScienceWorld Dependencies
Requires Java 1.8+ and external ScienceWorld repository:
```bash
git clone https://github.com/allenai/ScienceWorld.git
cd ScienceWorld  
pip install -e .
```

## Key Files to Understand

- `llm_game24_api.py:26` - ICRL flag configuration
- `llm_game24_api.py:210` - Main prompt construction logic
- `sciworld.py:39-50` - SciWorldConfig class with all available options
- `sciworld_armap/` - Modified ScienceWorld environment integration