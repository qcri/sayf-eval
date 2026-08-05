#!/usr/bin/env bash
# Phase-3 first-eval baselines for the MCQ knowledge set: EN->EN and Arabic
# (translate-test), across the multiple-choice knowledge benchmarks.
#
# The Arabic rendering is a STUDYABLE VARIABLE, chosen with AR_RENDER:
#   seedmini   - letter render (Question:/A:/..) + Arabic SYS_AR system prompt,
#                reproducing seed-mini's eval_tri_mcq.py. The EN run is rendered
#                the same way (--mcq-render letter) for a matched comparison.
#   harness    - keep each task's English wrapper, swap ONLY question+choices to
#                Arabic (one manipulated variable). EN run is untouched.
#   fullprompt - translate the whole rendered prompt live via the translator
#                (fully Arabic, incl. wrapper). EN run untouched.
# seedmini/harness use your pre-built Gemma3 field translations (GEMMA_MAP) and
# fall back to a live Gemma translation (TRANSLATOR_MODEL) for items not covered.
#
# Controlled-study note: every model condition must see the SAME Arabic items, so
# pre-built translations (the Gemma files) are the shared source; live fallback
# write-through is cached under AR_CACHE for reuse.
#
# Prereqs (run where your infra lives; NOT in this scaffold's sandbox):
#   pip install -e .            # litellm + datasets
#   export OPENAI_API_KEY=...   ANTHROPIC_API_KEY=...
#   export SAYF_EVAL_CISSP_PATH=/path/to/cissp.jsonl   # optional, to include CISSP
#
# Usage:
#   MODEL=openai/gpt-4o JUDGE=anthropic/claude-sonnet-4-20250514 \
#   AR_RENDER=harness GEMMA_MAP=scripts/gemma_map.example.json \
#   TRANSLATOR_MODEL=hosted_vllm/gemma-3-27b-it \
#   TRANSLATOR_BASE_URL=http://10.141.11.88:7797/v1 \
#   ./scripts/run_mcq_baselines.sh
set -euo pipefail

MODEL="${MODEL:?set MODEL, e.g. openai/gpt-4o}"
JUDGE="${JUDGE:?set JUDGE, e.g. anthropic/claude-sonnet-4-20250514}"
AR_RENDER="${AR_RENDER:-harness}"                 # seedmini | harness | fullprompt
GEMMA_MAP="${GEMMA_MAP:-scripts/gemma_map.example.json}"
TRANSLATOR_MODEL="${TRANSLATOR_MODEL:-}"          # Gemma model for live fallback / fullprompt
TRANSLATOR_BASE_URL="${TRANSLATOR_BASE_URL:-}"    # Gemma endpoint (else reuses model base-url)
MAX_SAMPLES="${MAX_SAMPLES:-}"                    # set e.g. 25 for a quick smoke run
AR_CACHE="${AR_CACHE:-cache/ar-$AR_RENDER}"       # live translations written here for reuse
OUT_ROOT="${OUT_ROOT:-outputs}"

case "$AR_RENDER" in seedmini|harness|fullprompt) ;; *) echo "bad AR_RENDER=$AR_RENDER"; exit 2;; esac

# MCQ knowledge tasks. CISSP only if its (private) dataset path is set.
TASKS=(cybermetric secbench seceval mmlu-cs
       redsage_frameworks redsage_generals redsage_skills redsage_cli redsage_kali)
[ -n "${SAYF_EVAL_CISSP_PATH:-}" ] && TASKS+=(cissp)

SLUG="$(printf '%s' "$MODEL" | tr '/ ' '__')"
COMMON=(--tasks "${TASKS[@]}")
[ -n "$MAX_SAMPLES" ] && COMMON+=(--max-samples "$MAX_SAMPLES")

# Arabic-side flags built from AR_RENDER.
AR_FLAGS=(--ar-render "$AR_RENDER" --translator-write-cache "$AR_CACHE")
EN_FLAGS=()
if [ "$AR_RENDER" = "seedmini" ] || [ "$AR_RENDER" = "harness" ]; then
  AR_FLAGS+=(--gemma-map "$GEMMA_MAP")
fi
[ -n "$TRANSLATOR_MODEL" ]   && AR_FLAGS+=(--translator-model "$TRANSLATOR_MODEL")
[ -n "$TRANSLATOR_BASE_URL" ] && AR_FLAGS+=(--translator-base-url "$TRANSLATOR_BASE_URL")
# seedmini: render the EN run the same (letter) way for a matched comparison.
[ "$AR_RENDER" = "seedmini" ] && EN_FLAGS+=(--mcq-render letter)

echo "== tasks: ${TASKS[*]}"
echo "== model=$MODEL judge=$JUDGE ar_render=$AR_RENDER gemma_map=$GEMMA_MAP translator=${TRANSLATOR_MODEL:-<none>}"

# 1) EN->EN baseline.
echo "== [EN->EN] $MODEL"
sayf-eval run "${COMMON[@]}" "${EN_FLAGS[@]}" \
  --model "$MODEL" --judge "$JUDGE" \
  --output-dir "$OUT_ROOT/$SLUG/en"

# 2) Arabic run under the chosen rendering.
echo "== [AR ($AR_RENDER)] $MODEL"
sayf-eval run "${COMMON[@]}" "${AR_FLAGS[@]}" \
  --model "$MODEL" --judge "$JUDGE" \
  --output-dir "$OUT_ROOT/$SLUG/ar-$AR_RENDER"

echo
echo "== done. Summaries:"
echo "   EN->EN          : $OUT_ROOT/$SLUG/en/summary.json"
echo "   AR ($AR_RENDER) : $OUT_ROOT/$SLUG/ar-$AR_RENDER/summary.json"
echo "   (per-task accuracy = MCQ exact accuracy; EN - AR = the AR-degradation signal."
echo "    Re-run with a different AR_RENDER to measure the rendering's own effect.)"
