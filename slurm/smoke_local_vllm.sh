#!/usr/bin/env bash
#SBATCH --job-name=seceval_smoke
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/export/home/aberriche/BenchBench/seceval-harness/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/seceval-harness/slurm/logs/%x_%j.err
set -uo pipefail

# Self-contained live validation of the local-vLLM-behind-LiteLLM path:
#   1. pip install the two missing deps (litellm, pytest) into the `vllm` env
#   2. serve a tiny instruct model OpenAI-compatibly with vLLM
#   3. run the real pytest suite
#   4. run an end-to-end smoke: model-under-test AND judge are the served model,
#      reached purely through LiteLLM via base_url (no API keys, no Azure quirks)
# Validates: pip install in a real env, litellm import, the OpenAI-compatible
# local transport, judge-as-Model, full pipeline, real dataset load + metrics.
#
# NOTE: the env's torch/vLLM needs GPU compute capability sm_70+. The gpu-short
# partition may assign Tesla P100 (sm_60), which fails engine init. Submit
# excluding the P100 nodes so the scheduler picks a V100/T4/A16/A100:
#   sbatch --partition=gpu-all \
#     --exclude=crimv3mgpu003,crimv3mgpu021,crimv3mgpu022,crimv3mgpu023,crimv3srv040,crimv3srv041,crimv3srv042,crimv3srv047 \
#     slurm/smoke_local_vllm.sh
# (Validated PASS on gpu-all / A16, job 311007.)

ROOT=/export/home/aberriche/BenchBench/seceval-harness
MODEL_HF=Qwen/Qwen2.5-0.5B-Instruct
SERVED=qwen-smoke
PORT=8011
SAMPLES=3
mkdir -p "$ROOT/slurm/logs" "$ROOT/outputs"

echo "============================================================"
echo "Job ${SLURM_JOB_ID:-local}  Node ${SLURMD_NODENAME:-$(hostname)}  $(date)"
echo "Model/judge: $MODEL_HF  port $PORT  samples $SAMPLES"
echo "============================================================"

# ── Environment ─────────────────────────────────────────────────────────────
# Compute nodes don't have `conda` on PATH; use the env's interpreter directly
# and prepend its bin (so vLLM's helper binaries resolve).
ENV_BIN=/export/home/aberriche/miniconda3/envs/vllm/bin
export PATH="$ENV_BIN:$PATH"
PY="$ENV_BIN/python"

echo ">> installing litellm + pytest (compute node has network egress)"
"$PY" -m pip install -q litellm pytest || { echo "pip install FAILED"; exit 1; }
"$PY" -c "import litellm, pytest, cvss, datasets; from importlib.metadata import version; print('deps ok: litellm', version('litellm'))"

# ── Serve the model (background) ──────────────────────────────────────────────
echo ">> starting vLLM server"
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_HF" \
    --served-model-name "$SERVED" \
    --port "$PORT" \
    --gpu-memory-utilization 0.30 \
    --max-model-len 4096 \
    --enforce-eager \
    > "$ROOT/slurm/logs/vllm_server_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
trap 'echo ">> stopping vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true' EXIT

echo ">> waiting for /health (up to 15 min; slow/older GPUs need it)"
for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo ">> server healthy after ${i}x5s"; break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM died during startup; tail of server log:"
        tail -40 "$ROOT/slurm/logs/vllm_server_${SLURM_JOB_ID:-local}.log"
        exit 1
    fi
    sleep 5
done
curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "server never became healthy"; exit 1; }

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# ── Unit tests (real env) ─────────────────────────────────────────────────────
echo ">> pytest"
"$PY" -m pytest tests/ -q || echo "WARN: pytest reported failures (see above)"

# ── Live end-to-end smoke ─────────────────────────────────────────────────────
echo ">> live smoke: seceval run (model + judge = local $SERVED)"
OUT="$ROOT/outputs/smoke_${SLURM_JOB_ID:-local}"
"$PY" -m seceval.cli run \
    --tasks mcq seceval \
    --model "hosted_vllm/$SERVED" --base-url "http://localhost:$PORT/v1" --api-key EMPTY \
    --judge "hosted_vllm/$SERVED" --judge-base-url "http://localhost:$PORT/v1" --judge-api-key EMPTY \
    --output-dir "$OUT" \
    --max-samples "$SAMPLES" \
    --concurrency 4

echo "============================================================"
echo ">> summary.json:"
cat "$OUT/summary.json"
echo ""
echo ">> output files:"; ls -la "$OUT"
echo ">> sanity: detailed row count per task"
for t in mcq seceval; do
    n=$(wc -l < "$OUT/${t}_detailed.jsonl" 2>/dev/null || echo 0)
    echo "   $t: $n detailed rows"
done
echo "Done: $(date)"
echo "============================================================"
