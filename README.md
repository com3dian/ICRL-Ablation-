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
```
python llm_game24_api.py
```

Run reflexion baseline:
```
python llm_game24_api_reflexion.py
```
Run self-refine baseline:
```
python llm_game24_api_self-refine.py
```
Run Best-of-N baseline:
```
python llm_game24_api_rejection.py
```
Run long CoT baseline:
```
python llm_game24_api_CoT.py
```


## Creative Writing

Configure the OpenAI API key and specify which ICRL method or ablation to run in the file, then run:
```
python llm_creative_writing_api.py
```

Run reflexion baseline:
```
python llm_creative_writing_api_reflexion.py
```
Run self-refine baseline:
```
python llm_creative_writing_api_self-refine.py
```
Run Best-of-N baseline:
```
python llm_creative_writing_api_rejection.py
```
Run long CoT baseline:
```
python llm_creative_writing_api_CoT.py
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

For all the other options available including the ablations and baselines, refer to the `SciWorldConfig` class in `sciworld.py`.

## Math Competitions (AIME/HMMT)

```
python math_bench.py
```

## Beyond Parametric Knowledge (ArXiv Abstract Generation)

Run ICRL:
```
python beyond_parameterized_knowledge/beyond_parametric_knowledge.py
```

Run Best-of-N baseline:
```
python beyond_parameterized_knowledge/beyond_parametric_knowledge.py --rejection_sampling
```

Run exploitation-only ablation:
```
python beyond_parameterized_knowledge/beyond_parametric_knowledge.py --exploitation_only
```

Run exploration-only ablation:
```
python beyond_parameterized_knowledge/beyond_parametric_knowledge.py --explore_only
```

## Attention Analysis (Reward-Sensitive Heads)

Analyzes attention patterns in Qwen3-32B to identify reward-sensitive heads. Requires 2 GPUs.

Run the initial analysis (layers -1 to -4, 64 heads each):
```bash
cd attention_analysis
bash test_layers_heads.sh <path_to_output_list.json>
```

Run the extended analysis across all 32 layers:
```bash
bash run_all_layers.sh <path_to_output_list.json>
```

Generate the significant heads figure:
```
python plot_significant_heads_bar.py
```

> Acknowledgement:
> We have borrowed code from the [ScienceWorld](https://github.com/allenai/ScienceWorld), [ARMAP](https://github.com/heaplax/ARMAP), and [CLIN](https://github.com/allenai/clin) repositories.
