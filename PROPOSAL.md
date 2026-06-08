# Task: Turn the cybersecurity eval harness into a lightweight framework

## Goal

Expose our existing cybersecurity LLM benchmark harness through **one interface
common to all LLMs**, packaged as a small, reusable framework rather than a
script.

The design rests on a single decision:

> **LiteLLM for the common interface; lighteval as the structural reference for
> how to wrap tasks and metrics around it.**

This document explains that idea and the architecture it implies.

---

## Where we are today

Our harness is already a coherent pipeline:

- **Unified prompt** — one prompt template applied across benchmarks.
- **Unified inference params** — one config (temperature, max tokens, etc.)
  applied across models.
- **LLM-as-a-judge for extraction** — a judge model pulls the answer out of the
  raw generation.
- **Unified scoring metric** — a single metric turns judged samples into a score.

What's missing is a clean **model-agnostic boundary** and a **framework shape**
so any LLM (hosted API or local) can be run through the same harness without
per-provider glue.

---

## The core idea: two layers, kept separate

The phrase "interface common to all LLMs" actually hides two distinct concerns.
We solve each with a different existing tool instead of hand-rolling either.

### Layer 1 — Transport: LiteLLM (the common interface)

"Talk to any provider the same way" is a solved problem. **LiteLLM** exposes
every provider through the OpenAI-style call format — OpenAI, Anthropic,
Bedrock, Vertex, Azure, Together, Groq, and OpenAI-compatible local servers
(e.g. vLLM via a `base_url`).

Consequence for us:

- We maintain **one** model adapter, not a provider matrix.
- "Unified inference params" become a **single config object** handed to LiteLLM.
- A local model is just another endpoint — no special-casing.
- The **judge is not special**: it's another call through the same interface
  with its own prompt + params.

### Layer 2 — Structure: lighteval (the reference, not necessarily a dependency)

lighteval is the lightweight end of the eval-framework spectrum, and its whole
design goal is ours: evaluate any model — local checkpoint or API — through a
common interface. We mirror its structure rather than re-inventing it. Two
patterns worth lifting directly:

1. **Backend abstraction.** lighteval already treats LiteLLM as one of its
   supported backends. That confirms the layering (framework on top, LiteLLM
   underneath) and gives us a reference for the `Model` boundary.
2. **Two-level metrics.** lighteval splits scoring into a **sample-level**
   function and a **corpus-level** aggregation (e.g. exact-match per sample,
   then averaged over the corpus). This maps directly onto our pipeline:
   *judge/extract = sample level*, *unified metric = corpus level*.

We can either depend on lighteval or just copy its shape — start with the shape;
the boundary stays compatible if we adopt it later.

---

## Proposed architecture

Three thin pieces over LiteLLM:

```
+-------------------------------------------------------------+
|  Task / harness                                             |
|    - unified prompt template                                |
|    - inference-param config                                 |
|    - dataset of samples (input, target)                     |
+----------------------------+--------------------------------+
                             |
                             v
+-------------------------------------------------------------+
|  Model interface  (one adapter)                             |
|    generate(messages, params) -> Response                   |
|    backed by LiteLLM  ->  "common interface to all LLMs"    |
+----------------------------+--------------------------------+
                             |
                             v
+-------------------------------------------------------------+
|  Scorer                                                     |
|    judge call (SAME Model interface)  -> extracted answer   |
|    sample_metric(answer, target)      -> per-sample score   |
|    corpus_metric([sample scores])     -> final score        |
+-------------------------------------------------------------+
```

### Interface sketch (illustrative, not final)

```python
# Layer 1: the common interface (LiteLLM under the hood)
class Model(Protocol):
    def generate(self, messages: list[dict], params: GenParams) -> Response: ...

class GenParams:        # "unified inference params"
    temperature: float
    max_tokens: int
    # ...

# Layer 2: structure (lighteval-shaped)
class Task:
    prompt_template: str         # "unified prompt"
    params: GenParams
    dataset: list[Sample]        # Sample = (input, target)
    scorer: "Scorer"

class Scorer:
    judge: Model                 # judge is just another Model
    def extract(self, generation: str) -> str: ...        # LLM-as-judge
    def sample_metric(self, answer: str, target: str) -> float: ...
    def corpus_metric(self, scores: list[float]) -> dict: ...
```

The model-under-test and the judge are **the same `Model` type**, so either can
be swapped to any provider via LiteLLM with no code change.

---

## How today's harness maps on

| Existing piece               | Lands in           | Notes                                   |
|------------------------------|--------------------|-----------------------------------------|
| Unified prompt               | `Task.prompt_template` | unchanged content                   |
| Unified inference params     | `GenParams`        | one config, passed to LiteLLM           |
| LLM-as-judge extraction      | `Scorer.extract` + `Scorer.judge` | judge = a `Model`        |
| Unified scoring metric       | `Scorer.sample_metric` / `corpus_metric` | two-level, lighteval-style |

---

## Build vs. reuse

- **Reuse:** LiteLLM for all provider transport. Do **not** write provider
  adapters.
- **Reference (copy the shape):** lighteval's backend boundary + two-level
  metric design.
- **Build (ours):** the cybersecurity-specific prompt, judge logic, scoring
  metric, and the thin `Task`/`Scorer` wrappers above.

---

## Out of scope (for now)

- Agentic / tool-use / CTF-style tasks and sandboxed execution. If we go there
  later, the upgrade path is **Inspect AI** (the standard for cyber/agentic
  evals), and lighteval can already use Inspect as a backend — so this design
  doesn't lock us out.
- Multi-turn conversations.
- Leaderboard hosting / dashboards.

---

## Open questions

- Depend on lighteval directly, or just mirror its structure with our own thin
  classes? (Lean: start by mirroring; keep the boundary compatible.)
- Where do per-benchmark prompt variations live — in `Task`, or a registry?
- Do we need log-prob / multiple-choice scoring, or is generative + judge enough
  for our benchmarks?

---

## References

- LiteLLM — unified OpenAI-format interface across providers.
- lighteval — lightweight evaluation framework; LiteLLM backend; two-level
  (sample/corpus) metrics.
- Inspect AI — heavier, agentic/sandboxed; future upgrade path only.
