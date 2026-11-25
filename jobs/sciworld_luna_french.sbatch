#!/usr/bin/env bash
# Submit from the repository root:
#   sbatch jobs/sciworld_luna_french.sbatch
#SBATCH --job-name=sciworld-luna-french
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-sciworld-luna-french-%j.out
#SBATCH --error=slurm-sciworld-luna-french-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SCIWORLD_FEEDBACK=fr exec bash "$REPO_DIR/jobs/sciworld_luna_normal.sbatch"
