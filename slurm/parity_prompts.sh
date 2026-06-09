#!/usr/bin/env bash
#SBATCH --job-name=seceval_parity
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/export/home/aberriche/BenchBench/seceval-harness/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/seceval-harness/slurm/logs/%x_%j.err
set -uo pipefail

# Tier-1 parity: new loaders vs the original collectors' committed outputs.
# Deterministic, no API; just needs `datasets` + HF network (compute node).

ROOT=/export/home/aberriche/BenchBench/seceval-harness
ENV_BIN=/export/home/aberriche/miniconda3/envs/vllm/bin
export PATH="$ENV_BIN:$PATH"
PY="$ENV_BIN/python"
mkdir -p "$ROOT/slurm/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "Job ${SLURM_JOB_ID:-local}  Node ${SLURMD_NODENAME:-$(hostname)}  $(date)"
"$PY" tests/parity_prompts.py
echo "Done: $(date)"
