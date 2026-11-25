#!/usr/bin/env bash
# Submit from the repository root (random feedback seed defaults to 0):
#   sbatch jobs/sciworld_luna_random.sbatch
# Small connectivity test (makes model API calls):
#   SAMPLES=1 ROUNDS=2 BOOTSTRAP_ATTEMPTS=0 MAX_ENV_STEPS=3 sbatch jobs/sciworld_luna_random.sbatch
# Continue an interrupted random-reward run with its saved configuration and seed:
#   RESUME_DIR=ablation/outputs/sciworld/gpt6luna/numeric/random/<run-id> sbatch jobs/sciworld_luna_random.sbatch
# Use RANDOM_REWARD_SEED to choose a seed for a new run.
#SBATCH --job-name=sciworld-luna-random
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-sciworld-luna-random-%j.out
#SBATCH --error=slurm-sciworld-luna-random-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"
if [[ ! -f ablation/run.py ]]; then
    echo "Submit from the repository root or set REPO_DIR to its absolute path." >&2
    exit 1
fi

ENV_PREFIX="${ENV_PREFIX:-$PWD/.conda/ablation}"
if [[ ! -x "$ENV_PREFIX/bin/python" || ! -x "$ENV_PREFIX/bin/java" ]]; then
    echo "Missing Python/Java environment at $ENV_PREFIX; follow the README setup first." >&2
    exit 1
fi
export PATH="$ENV_PREFIX/bin:$PATH"
export JAVA_HOME="$ENV_PREFIX"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# Match the baseline's 29 simulator JVMs and offline action matcher.
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--XX:ActiveProcessorCount=1 -Xmx1024m}"
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/icrl-matplotlib-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${RESUME_DIR:-}" ]]; then
    # Resume only this condition, preserving its saved configuration and seed.
    "$ENV_PREFIX/bin/python" - "$RESUME_DIR" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
expected = {"task": "sciworld", "provider": "cliproxy", "model_alias": "gpt6luna",
            "feedback": "numeric", "reward_polarity": "random"}
if any(config.get(key) != value for key, value in expected.items()):
    raise SystemExit("RESUME_DIR must be a random-reward numeric ScienceWorld gpt6luna Cliproxy run")
PY
    run_args=(--resume "$RESUME_DIR" --sciworld-resume-mode "${RESUME_MODE:-auto}")
else
    run_args=(
        --task sciworld --provider cliproxy --model gpt6luna
        --feedback numeric --reward-polarity random
        --random-reward-seed "${RANDOM_REWARD_SEED:-0}"
        --samples "${SAMPLES:-29}" --rounds "${ROUNDS:-40}"
        --bootstrap-attempts "${BOOTSTRAP_ATTEMPTS:-2}"
        --max-env-steps "${MAX_ENV_STEPS:-15}"
        --max-tokens "${MAX_TOKENS:-1000}"
        --variation-idx 0 --simplifications easy
    )
fi

# Check local prerequisites and random feedback without calling the API.
"$ENV_PREFIX/bin/python" - <<'PY'
import os
from pathlib import Path

import nltk
from dotenv import load_dotenv

from ablation.feedback import feedback_language, format_feedback
from ablation.reward_polarity import shown_reward

# Draws must depend on position and seed, independently of the true reward.
with feedback_language("numeric"):
    for step in range(1, 16):
        position = dict(step=step, random_seed=0, sample_id=0, round_idx=0)
        expected = shown_reward("sciworld", 0, "random", **position)
        if not isinstance(expected, int) or not -100 <= expected <= 100:
            raise SystemExit("ScienceWorld random-reward range check failed")
        for reward in (100, 30, 0, -3, -100, -130):
            shown = shown_reward("sciworld", reward, "random", **position)
            if shown != expected or format_feedback(shown) != str(expected):
                raise SystemExit("ScienceWorld random-reward feedback check failed")
load_dotenv(Path.cwd() / ".env", override=False)
if not os.getenv("CLIPROXY_API_KEY"):
    raise SystemExit("CLIPROXY_API_KEY is missing; set it in the private root .env")
for resource in ("corpora/wordnet.zip", "tokenizers/punkt_tab/english/"):
    nltk.data.find(resource)
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True)
print("Local prerequisites and random-reward feedback ready; credentials were not printed.", flush=True)
PY

printf 'Job %s on %s; repository: %s\n' "${SLURM_JOB_ID:-local}" "$(hostname)" "$PWD"
printf 'Command: python -u -m ablation.run'
printf ' %q' "${run_args[@]}"
printf '\n'
exec "$ENV_PREFIX/bin/python" -u -m ablation.run "${run_args[@]}"
