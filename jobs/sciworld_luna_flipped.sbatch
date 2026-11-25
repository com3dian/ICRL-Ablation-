#!/usr/bin/env bash
# Submit from the repository root:
#   sbatch jobs/sciworld_luna_flipped.sbatch
# Small connectivity test (makes model API calls):
#   SAMPLES=1 ROUNDS=2 BOOTSTRAP_ATTEMPTS=0 MAX_ENV_STEPS=3 sbatch jobs/sciworld_luna_flipped.sbatch
# Continue an interrupted flipped-reward run with its saved configuration:
#   RESUME_DIR=ablation/outputs/sciworld/gpt6luna/numeric/flipped/<run-id> sbatch jobs/sciworld_luna_flipped.sbatch
#SBATCH --job-name=sciworld-luna-flipped
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-sciworld-luna-flipped-%j.out
#SBATCH --error=slurm-sciworld-luna-flipped-%j.err

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
    # Resume only this condition, preserving its saved configuration.
    "$ENV_PREFIX/bin/python" - "$RESUME_DIR" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
expected = {"task": "sciworld", "provider": "cliproxy", "model_alias": "gpt6luna",
            "feedback": "numeric", "reward_polarity": "flipped"}
if any(config.get(key) != value for key, value in expected.items()):
    raise SystemExit("RESUME_DIR must be a flipped-reward numeric ScienceWorld gpt6luna Cliproxy run")
PY
    run_args=(--resume "$RESUME_DIR" --sciworld-resume-mode "${RESUME_MODE:-auto}")
else
    run_args=(
        --task sciworld --provider cliproxy --model gpt6luna
        --feedback numeric --reward-polarity flipped
        --samples "${SAMPLES:-29}" --rounds "${ROUNDS:-40}"
        --bootstrap-attempts "${BOOTSTRAP_ATTEMPTS:-2}"
        --max-env-steps "${MAX_ENV_STEPS:-15}"
        --max-tokens "${MAX_TOKENS:-1000}"
        --variation-idx 0 --simplifications easy
    )
fi

# Check local prerequisites and numeric sign inversion without calling the API.
"$ENV_PREFIX/bin/python" - <<'PY'
import os
from pathlib import Path

import nltk
from dotenv import load_dotenv

from ablation.feedback import feedback_language, format_feedback
from ablation.reward_polarity import shown_reward

# Include a corrected failure penalty after 30 points: -100 - 30 = -130.
with feedback_language("numeric"):
    for reward in (100, 30, 0, -3, -100, -130):
        shown = shown_reward("sciworld", reward, "flipped")
        if shown != -reward or format_feedback(shown) != str(-reward):
            raise SystemExit("ScienceWorld flipped-reward feedback check failed")
    if sum(shown_reward("sciworld", reward, "flipped") for reward in (30, 0, -130)) != 100:
        raise SystemExit("ScienceWorld flipped attempt-total check failed")
load_dotenv(Path.cwd() / ".env", override=False)
if not os.getenv("CLIPROXY_API_KEY"):
    raise SystemExit("CLIPROXY_API_KEY is missing; set it in the private root .env")
for resource in ("corpora/wordnet.zip", "tokenizers/punkt_tab/english/"):
    nltk.data.find(resource)
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True)
print("Local prerequisites and flipped-reward feedback ready; credentials were not printed.", flush=True)
PY

printf 'Job %s on %s; repository: %s\n' "${SLURM_JOB_ID:-local}" "$(hostname)" "$PWD"
printf 'Command: python -u -m ablation.run'
printf ' %q' "${run_args[@]}"
printf '\n'
exec "$ENV_PREFIX/bin/python" -u -m ablation.run "${run_args[@]}"
