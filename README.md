# Reward Is Enough: LLMs Are In-Context Reinforcement Learners

This repository contains the code for reproducing the results in the paper "Reward Is Enough: LLMs Are In-Context Reinforcement Learners".

## Shared Setup

Install the dependencies. We recommend using `uv`.
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```
## Game of 24
Configure the OpenAI API key and specify which ICRL method or ablation to run in the file, then run:
```python llm_game_24_api.py
```

Run other baseline methods
```python llm_game_24_reflexion.py
```

## Creative Writing
Configure the OpenAI API key and specify which ICRL method or ablation to run in the file, then run:
```python llm_game_24_api.py
```

## ScienceWorld

### Setup

Make sure you have Java 1.8+ installed
```bash
javac -version
```

Clone the ScienceWorld repository and install it
```bash
git clone https://github.com/allenai/ScienceWorld.git
cd ScienceWorld
pip install -e .
```

### Running the experiments

Run ICRL preset:
```bash
python3 sciworld.py icrl_mode=ICRL num_envs=29 
```

Run ICRL ablations, e.g. explore_only:
```bash
python3 sciworld.py icrl_mode=ICRL num_envs=29 explore_only=true
```

Run other baselines, e.g. random sampling:
```bash
python3 sciworld.py icrl_mode=RANDOM_SAMPLING num_envs=29 max_env_steps=200 
```

For all the other options available inlcuding the ablations and baselines, refer to the `SciWorldConfig` class in `sciworld.py`.

> Acknowledgement:
> We have borrowed code from the [ScienceWorld](https://github.com/allenai/ScienceWorld), [ARMAP](https://github.com/heaplax/ARMAP), and [CLIN](https://github.com/allenai/clin) repositories.
