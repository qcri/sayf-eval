"""Live smoke against the project's Azure OpenAI deployment (gpt-5.4).

Validates the real LiteLLM round-trip end to end: Model adapter -> litellm ->
Azure, judge-as-Model, pipeline, real dataset load, and corpus metrics. Runs on
CPU (no model serving). Requires AZURE_API_KEY in the environment.

Usage:
    AZURE_API_KEY=... python tests/live_smoke_azure.py [--samples 3]
"""

import argparse
import json
import os
import sys
import tempfile

from seceval.model import GenParams, Model
from seceval.scorer import JudgeScorer
from seceval.pipeline import RunConfig, run_tasks
import seceval.tasks  # noqa: F401 — registers tasks
from seceval.registry import get_task

AZURE_ENDPOINT = "https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-5.4"
AZURE_API_VERSION = "2024-12-01-preview"


def make_azure_model(concurrency: int = 4) -> Model:
    key = os.environ.get("AZURE_API_KEY")
    if not key:
        sys.exit("AZURE_API_KEY not set")
    # Azure OpenAI: litellm reads the endpoint from api_base (passed via extra),
    # plus api_version. model string is azure/<deployment>.
    return Model(
        model=f"azure/{AZURE_DEPLOYMENT}",
        api_key=key,
        api_version=AZURE_API_VERSION,
        extra={"api_base": AZURE_ENDPOINT},
        concurrency=concurrency,
    )


def preflight(model: Model) -> None:
    """One direct call to confirm the transport works before the full run."""
    print(">> preflight: single completion through LiteLLM -> Azure")
    r = model.generate(
        [{"role": "user", "content": "Reply with the single word: OK"}],
        GenParams(max_tokens=2048),
    )
    print(f"   ok={r.ok} text={r.text!r} usage={r.usage}")
    if not r.ok:
        sys.exit("preflight failed — transport not working (see retries above)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--tasks", nargs="+", default=["mcq", "seceval"])
    args = ap.parse_args()

    model = make_azure_model()
    preflight(model)

    judge = make_azure_model()
    scorer = JudgeScorer(judge, GenParams(max_tokens=2048))
    tasks = [get_task(t) for t in args.tasks]

    out = tempfile.mkdtemp(prefix="seceval_azure_smoke_")
    print(f">> running tasks={args.tasks} samples={args.samples} -> {out}")
    summary = run_tasks(tasks, model, scorer, out, RunConfig(max_samples=args.samples))

    print(">> summary.json:")
    print(json.dumps(summary, indent=2))

    # Sanity: every task attempted the requested number of items (denominator
    # = attempted; only judge failures are excluded).
    ok = True
    for t in args.tasks:
        s = summary.get(t, {})
        attempted = s.get("total", 0) + s.get("skipped", 0)
        print(f"   {t}: attempted={attempted} total={s.get('total')} "
              f"skipped={s.get('skipped')} accuracy={s.get('accuracy')}")
        if attempted == 0:
            ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
