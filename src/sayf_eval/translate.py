"""Translation layer — render benchmark prompts in a target language.

The harness is English-first: loaders in :mod:`sayf_eval.datasets` produce a
fully-rendered English ``Sample.prompt`` plus a per-task English ``system_prompt``
(in the registry). A :class:`Translator` sits at a single seam in
:func:`sayf_eval.pipeline.run_inference` — between ``task.load()`` and inference —
and rewrites the prompts the model-under-test actually sees, leaving ground truth,
scoring, and metrics untouched (MCQ targets are language-agnostic letters).

Three interchangeable backends, selected by the user:

* :class:`IdentityTranslator` — no-op; the English baseline.
* :class:`LLMTranslator` — translate at run time with *any* LiteLLM-reachable
  model (a local vLLM server, Azure, OpenAI…). The same :class:`~sayf_eval.model.Model`
  adapter as the model-under-test, so "pick the translator" == pick a model string.
  System prompts are the exact ones used to build the project's translated sets
  (see :mod:`sayf_eval.prompts_translate`).
* :class:`CacheTranslator` — read pre-translated prompts from disk (e.g. a
  QE-checked translation set produced offline). Reproducible and free at run time;
  falls back to identity on a miss so a partial cache still runs.

What gets translated:

* **User prompt** — the rendered question/options, machine translated with the
  MCQ or generic system prompt depending on whether the sample carries choices.
* **System prompt** — the model-under-test's task system prompt is translated once
  (generic prompt) and cached; identity keeps it English.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from typing import Protocol, runtime_checkable

from sayf_eval import prompts_translate as P
from sayf_eval.model import GenParams, Model
from sayf_eval.task import Sample, Task


logger = logging.getLogger(__name__)


@runtime_checkable
class Translator(Protocol):
    """Rewrites the prompts a model-under-test sees, in a target language."""

    name: str
    target_lang: str

    def system_prompt(self, task: Task) -> str | None:
        """The system prompt to use for ``task`` (possibly translated)."""
        ...

    def translate_samples(self, task: Task, samples: list[Sample]) -> list[Sample]:
        """Return new samples whose ``prompt`` is in the target language."""
        ...


def _has_choices(samples: list[Sample]) -> bool:
    """MCQ family iff most samples carry choices (selects the MCQ prompt)."""
    if not samples:
        return False
    with_choices = sum(1 for s in samples if s.choices)
    return with_choices >= max(1, len(samples) // 2)


# ── Identity (English baseline) ──────────────────────────────────────────────


class IdentityTranslator:
    """No-op translator; preserves the English baseline behavior."""

    name = "none"

    def __init__(self, target_lang: str = "en") -> None:
        self.target_lang = target_lang

    def system_prompt(self, task: Task) -> str | None:
        return task.system_prompt

    def translate_samples(self, task: Task, samples: list[Sample]) -> list[Sample]:
        return samples


# ── LLM-backed (translate at run time) ───────────────────────────────────────


class LLMTranslator:
    """Translate prompts at run time with a LiteLLM-reachable model.

    The translator model is an ordinary :class:`~sayf_eval.model.Model`, so any
    local vLLM server (``base_url``) or hosted provider works. Translations are
    cached in memory for the run (deterministic at temperature 0) and, optionally,
    written through to ``write_cache_dir`` for later reuse via :class:`CacheTranslator`.

    Args:
        model: the translator :class:`Model`.
        target_lang: ``"ar"`` (EN→AR) or ``"en"`` (AR→EN translate-test).
        model_aware: opt into the per-model MCQ prompts (Fanar/LLaMA/GPT-OSS) from
            the original pipeline. Off by default — those embed raw chat-template
            tokens meant for a non-chat serving path.
        retry: append the original pipeline's MCQ retry refinement clause.
    """

    def __init__(
        self,
        model: Model,
        target_lang: str = "ar",
        max_tokens: int = 2048,
        model_aware: bool = False,
        retry: bool = False,
        write_cache_dir: str | None = None,
    ) -> None:
        if target_lang not in ("ar", "en"):
            raise ValueError(f"target_lang must be 'ar' or 'en', got {target_lang!r}.")
        self.model = model
        self.target_lang = target_lang
        self.max_tokens = max_tokens
        self.model_aware = model_aware
        self.retry = retry
        self.write_cache_dir = write_cache_dir
        self.name = f"llm:{model.model}->{target_lang}"
        self._sys_cache: dict[str, str] = {}

    def _translate_system_prompt(self, mcq: bool) -> str:
        if self.target_lang == "ar":
            return P.select_en2ar(self.model.model, mcq=mcq, retry=self.retry, model_aware=self.model_aware)
        return P.select_ar2en(mcq=mcq)

    # -- system prompt (of the model-under-test) -----------------------------

    def system_prompt(self, task: Task) -> str | None:
        if not task.system_prompt:
            return None
        if task.system_prompt in self._sys_cache:
            return self._sys_cache[task.system_prompt]
        # Translate the task's own system prompt with the generic instruction.
        sys = P.GENERIC_EN2AR if self.target_lang == "ar" else P.GENERIC_AR2EN
        out = self._translate_one(task.system_prompt, sys)
        self._sys_cache[task.system_prompt] = out
        return out

    # -- user prompts --------------------------------------------------------

    def translate_samples(self, task: Task, samples: list[Sample]) -> list[Sample]:
        if not samples:
            return samples
        sys = self._translate_system_prompt(mcq=_has_choices(samples))
        translated = self._translate_batch([s.prompt for s in samples], sys)
        out: list[Sample] = []
        for s, t in zip(samples, translated):
            meta = {**s.metadata, "prompt_en": s.prompt, "translator": self.name}
            out.append(replace(s, prompt=t, metadata=meta))
        if self.write_cache_dir:
            _write_cache(self.write_cache_dir, task, out)
        return out

    # -- internals -----------------------------------------------------------

    def _messages(self, text: str, system: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]

    def _translate_one(self, text: str, system: str) -> str:
        if not text.strip():
            return text
        resp = self.model.generate(self._messages(text, system), GenParams(max_tokens=self.max_tokens))
        return resp.text.strip() if resp.ok and resp.text.strip() else text

    def _translate_batch(self, texts: list[str], system: str) -> list[str]:
        batch = [self._messages(t, system) for t in texts]
        responses = self.model.generate_batch(batch, GenParams(max_tokens=self.max_tokens))
        out: list[str] = []
        fails = 0
        for src, r in zip(texts, responses):
            if r.ok and r.text.strip():
                out.append(r.text.strip())
            else:
                out.append(src)  # fall back to source so the eval still runs
                fails += 1
        if fails:
            logger.warning("%d/%d prompts fell back to source (translation failed).", fails, len(texts))
        return out


# ── Cache-backed (pre-translated offline) ────────────────────────────────────


class CacheTranslator:
    """Serve pre-translated prompts from disk; identity on a miss.

    Cache layout (``cache_dir/<task.name>.jsonl``), one row per sample::

        {"index": 0, "prompt": "<translated user prompt>",
         "system_prompt": "<optional translated system prompt>"}

    Rows match samples by ``index``. A missing file or index falls back to the
    English text so a partial cache still produces a complete run; misses are
    logged.
    """

    def __init__(self, cache_dir: str, target_lang: str = "ar") -> None:
        self.cache_dir = cache_dir
        self.target_lang = target_lang
        self.name = f"cache:{os.path.basename(cache_dir.rstrip('/'))}"
        self._cache: dict[str, dict[int, dict]] = {}

    def _load_task(self, task: Task) -> dict[int, dict]:
        if task.name in self._cache:
            return self._cache[task.name]
        path = os.path.join(self.cache_dir, f"{task.name}.jsonl")
        rows: dict[int, dict] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    rows[int(r["index"])] = r
        else:
            logger.warning("No translation cache for task %r at %s", task.name, path)
        self._cache[task.name] = rows
        return rows

    def system_prompt(self, task: Task) -> str | None:
        if not task.system_prompt:
            return None
        for r in self._load_task(task).values():
            if r.get("system_prompt"):
                return r["system_prompt"]
        return task.system_prompt

    def translate_samples(self, task: Task, samples: list[Sample]) -> list[Sample]:
        rows = self._load_task(task)
        out: list[Sample] = []
        misses = 0
        for s in samples:
            r = rows.get(s.index)
            if r and r.get("prompt"):
                meta = {**s.metadata, "prompt_en": s.prompt, "translator": self.name}
                out.append(replace(s, prompt=r["prompt"], metadata=meta))
            else:
                misses += 1
                out.append(s)
        if misses:
            logger.warning("%d/%d samples missing from cache for %r; used English.", misses, len(samples), task.name)
        return out


# ── helpers ──────────────────────────────────────────────────────────────────


def _write_cache(cache_dir: str, task: Task, samples: list[Sample]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{task.name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"index": s.index, "prompt": s.prompt}, ensure_ascii=False) + "\n")


def build_translator(
    kind: str,
    target_lang: str = "ar",
    *,
    backend: str = "endpoint",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    concurrency: int = 8,
    cache_dir: str | None = None,
    write_cache_dir: str | None = None,
    model_aware: bool = False,
    offline: dict | None = None,
) -> Translator:
    """Construct a translator from CLI-style options.

    ``kind`` is one of ``none`` (identity), ``llm`` (run-time MT; needs ``model``),
    or ``cache`` (offline cache; needs ``cache_dir``). For ``llm``, ``backend`` is
    ``endpoint`` (LiteLLM/vLLM server) or ``offline-vllm`` (in-process vLLM,
    server-less — same loading as the offline batched jobs); ``offline`` carries
    the vLLM kwargs (tensor_parallel_size, gpu_memory_utilization, max_model_len,
    trust_remote_code).
    """
    if kind == "none":
        return IdentityTranslator(target_lang="en")
    if kind == "llm":
        if not model:
            raise ValueError("--translator llm requires --translator-model.")
        if backend == "offline-vllm":
            from sayf_eval.backends import OfflineVLLMModel

            m = OfflineVLLMModel(model=model, **(offline or {}))
        else:
            m = Model(model=model, base_url=base_url, api_key=api_key, concurrency=concurrency)
        return LLMTranslator(m, target_lang=target_lang, model_aware=model_aware, write_cache_dir=write_cache_dir)
    if kind == "cache":
        if not cache_dir:
            raise ValueError("--translator cache requires --translator-cache-dir.")
        return CacheTranslator(cache_dir, target_lang=target_lang)
    raise ValueError(f"Unknown translator kind {kind!r}; choose none|llm|cache.")
