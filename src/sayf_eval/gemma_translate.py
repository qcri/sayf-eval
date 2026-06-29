"""Gemma-backed Arabic MCQ translation with selectable rendering.

The Arabic side of the translate-test study can be produced three ways, chosen by
``--ar-render`` so the *rendering* itself becomes a measurable variable:

* ``seedmini`` — reproduce the project's ``eval_tri_mcq.py`` pipeline: render the
  Arabic question + choices with :func:`_render_mcq_letter` (``Question:\\n.. / A: ..
  / Reply ONLY ..``) under the Arabic system prompt ``SYS_AR``. For a fair EN vs AR
  comparison the EN run should use the same letter rendering (``--mcq-render letter``).
* ``harness`` — keep each task's own English wrapper and substitute *only* the
  question and choice text with Arabic (the one manipulated variable). EN run is
  untouched.
* ``fullprompt`` — translate the whole rendered prompt live (handled by the
  existing :class:`~sayf_eval.translate.LLMTranslator`, not this module).

Source of the Arabic fields, in order: (1) the user's pre-built Gemma3 files
(field-level ``translated_question`` / ``translated_choices``, content-matched to
the English item), else (2) a live Gemma translation of the fields, else (3)
identity (English) so a partial run still completes. Live hits can be written
through to a cache for reuse.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import replace

from sayf_eval.datasets import SYS_AR, _render_mcq_letter
from sayf_eval.task import Sample, Task


logger = logging.getLogger(__name__)

RENDER_MODES = ("seedmini", "harness", "fullprompt")
_LETTERS = ("A", "B", "C", "D")


def _norm(s: str) -> str:
    """Normalize a question for content matching (case/whitespace-insensitive)."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ── pull the English question + choices back out of a rendered MCQ sample ────

_Q_PATTERNS = (
    re.compile(r"Question:\s*\n?(.+?)\n\s*(?:Options:|A[:)\.])", re.DOTALL),
    re.compile(r"Question:\s*\n?(.+?)\n\s*\n", re.DOTALL),
)


def mcq_fields(sample: Sample) -> tuple[str, dict[str, str]]:
    """Best-effort ``(question, {A..D: text})`` from a rendered MCQ sample.

    Choices come from the structured ``sample.choices`` (``"A: text"`` lines);
    the question is parsed from the rendered prompt, falling back to the first
    non-empty line.
    """
    choices: dict[str, str] = {}
    for line in sample.choices or []:
        m = re.match(r"\s*([A-D])\s*[:)\.\-]\s*(.+)", str(line))
        if m:
            choices[m.group(1).upper()] = m.group(2).strip()
    q = ""
    for pat in _Q_PATTERNS:
        m = pat.search(sample.prompt or "")
        if m:
            q = m.group(1).strip()
            break
    if not q:
        q = next((ln.strip() for ln in (sample.prompt or "").splitlines() if ln.strip()), "")
    return q, choices


def substitute_fields(prompt: str, en_q: str, ar_q: str, en_choices: dict, ar_choices: dict) -> str:
    """Exact-replace the English question + each choice text with Arabic, in place.

    Keeps the task's own wrapper untouched (the ``harness`` render mode). Longest
    strings are replaced first so a short choice that is a substring of another
    is not clobbered.
    """
    out = prompt
    if en_q and ar_q and en_q in out:
        out = out.replace(en_q, ar_q)
    pairs = [(en_choices[L], ar_choices[L]) for L in _LETTERS if en_choices.get(L) and ar_choices.get(L)]
    for en_c, ar_c in sorted(pairs, key=lambda p: len(p[0]), reverse=True):
        if en_c in out:
            out = out.replace(en_c, ar_c)
    return out


# ── content-keyed store over the user's pre-built Gemma3 files ───────────────


class GemmaFieldStore:
    """Per-task lookup of pre-translated Arabic fields, keyed by English question.

    ``task_map`` maps a registered task name to a Gemma3 JSONL whose rows carry
    ``question`` / ``answers`` (English) and ``translated_question`` /
    ``translated_choices`` (Arabic), e.g. ``CyberMetric-500-Arabic-gemma3.jsonl``.
    """

    def __init__(self, task_map: dict[str, str]) -> None:
        self.task_map = task_map
        self._loaded: dict[str, dict[str, dict]] = {}

    def _load(self, task_name: str) -> dict[str, dict]:
        if task_name in self._loaded:
            return self._loaded[task_name]
        rows: dict[str, dict] = {}
        path = self.task_map.get(task_name)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    q = r.get("question") or ""
                    tq = r.get("translated_question") or ""
                    tc = r.get("translated_choices") or {}
                    if not q or not tq or not tc:
                        continue
                    rows[_norm(q)] = {
                        "ar_q": tq.strip(),
                        "ar_choices": {k.upper(): str(v).strip() for k, v in tc.items()},
                        "en_choices": {k.upper(): str(v).strip() for k, v in (r.get("answers") or {}).items()},
                    }
        elif path:
            logger.warning("Gemma file for task %r not found at %s", task_name, path)
        self._loaded[task_name] = rows
        return rows

    def lookup(self, task_name: str, en_question: str) -> dict | None:
        return self._load(task_name).get(_norm(en_question))


# ── the translator ───────────────────────────────────────────────────────────


class GemmaArabicTranslator:
    """Field-level Arabic MCQ translator with selectable rendering + live fallback.

    ``render`` is ``"seedmini"`` (letter render + ``SYS_AR``) or ``"harness"``
    (substitute fields into the task's own wrapper). ``store`` provides pre-built
    Gemma translations; ``live`` (optional) is called ``live(question, choices)
    -> (ar_q, ar_choices)`` on a store miss.
    """

    target_lang = "ar"

    def __init__(self, render: str, store: GemmaFieldStore | None = None, live=None, write_cache_dir: str | None = None):
        if render not in ("seedmini", "harness"):
            raise ValueError(f"GemmaArabicTranslator render must be seedmini|harness, got {render!r}.")
        self.render = render
        self.store = store
        self.live = live
        self.write_cache_dir = write_cache_dir
        self.name = f"gemma:{render}"
        self._misses = 0
        self._live_used = 0

    def system_prompt(self, task: Task) -> str | None:
        # seedmini swaps in the Arabic system prompt; harness keeps the task's own
        # (English) system prompt — only the question/choices change.
        return SYS_AR if self.render == "seedmini" else task.system_prompt

    def _fields(self, task: Task, en_q: str, en_choices: dict) -> dict | None:
        if self.store is not None:
            hit = self.store.lookup(task.name, en_q)
            if hit:
                return hit
        if self.live is not None and en_q:
            try:
                ar_q, ar_choices = self.live(en_q, en_choices)
                self._live_used += 1
                return {"ar_q": ar_q, "ar_choices": ar_choices, "en_choices": en_choices}
            except Exception as e:  # noqa: BLE001
                logger.warning("live Gemma translation failed for one item: %s", e)
        return None

    def translate_samples(self, task: Task, samples: list[Sample]) -> list[Sample]:
        out: list[Sample] = []
        for s in samples:
            en_q, en_choices = mcq_fields(s)
            fields = self._fields(task, en_q, en_choices)
            if not fields:
                self._misses += 1
                out.append(s)  # identity: keep English so the run still completes
                continue
            ar_q = fields["ar_q"]
            ar_choices = fields["ar_choices"]
            if self.render == "seedmini":
                prompt = _render_mcq_letter(ar_q, ar_choices)
            else:  # harness: substitute into the task's own rendered prompt
                prompt = substitute_fields(s.prompt, en_q, ar_q, fields.get("en_choices") or en_choices, ar_choices)
            meta = {**s.metadata, "prompt_en": s.prompt, "translator": self.name}
            out.append(replace(s, prompt=prompt, metadata=meta))
        if self._misses:
            logger.warning(
                "%d/%d %s items had no Gemma translation (kept English).", self._misses, len(samples), task.name
            )
        if self.live is not None and self._live_used:
            logger.info("%d items translated live via Gemma for %s.", self._live_used, task.name)
        if self.write_cache_dir:
            self._write_cache(task, out)
        return out

    def _write_cache(self, task: Task, samples: list[Sample]) -> None:
        os.makedirs(self.write_cache_dir, exist_ok=True)
        path = os.path.join(self.write_cache_dir, f"{task.name}.jsonl")
        sys_p = self.system_prompt(task)
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                row = {"index": s.index, "prompt": s.prompt}
                if sys_p:
                    row["system_prompt"] = sys_p
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def re_render_letter(samples: list[Sample]) -> list[Sample]:
    """Re-render MCQ samples with the letter layout (EN-side parity for seedmini).

    Used for the English run so it matches the ``seedmini`` Arabic layout; the
    only change is the prompt rendering — target/choices are untouched.
    """
    out: list[Sample] = []
    for s in samples:
        q, choices = mcq_fields(s)
        if q and choices:
            out.append(replace(s, prompt=_render_mcq_letter(q, choices)))
        else:
            out.append(s)
    return out


def load_gemma_map(path: str) -> dict[str, str]:
    """Load a ``{task_name: gemma_jsonl_path}`` JSON map."""
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError(f"--gemma-map must be a JSON object {{task: path}}, got {type(m).__name__}.")
    return {str(k): str(v) for k, v in m.items()}


_LIVE_SYS = (
    "You are a professional Arabic cybersecurity translator. Translate the given English "
    "multiple-choice question and its options into Modern Standard Arabic (MSA). Preserve "
    "English technical terms, acronyms, code, CVE/CWE IDs, commands, and identifiers. "
    'Output ONLY valid JSON: {"question": "<ar>", "choices": {"A": "<ar>", "B": "<ar>", '
    '"C": "<ar>", "D": "<ar>"}}. No commentary.'
)


def make_live_field_translator(model, max_tokens: int = 512):
    """Return a ``live(question, choices) -> (ar_q, ar_choices)`` backed by ``model``.

    One chat call per item; on any failure raises so the caller logs and keeps the
    English item. Used as the store-miss fallback (e.g. a Gemma endpoint).
    """
    from sayf_eval.model import GenParams

    def live(question: str, choices: dict[str, str]):
        opts = "\n".join(f"{L}: {choices[L]}" for L in _LETTERS if choices.get(L))
        user = f"Question: {question}\n{opts}"
        resp = model.generate(
            [{"role": "system", "content": _LIVE_SYS}, {"role": "user", "content": user}],
            GenParams(max_tokens=max_tokens),
        )
        if not resp.ok or not resp.text.strip():
            raise RuntimeError("empty live translation")
        m = re.search(r"\{.*\}", resp.text, re.DOTALL)
        data = json.loads(m.group(0) if m else resp.text)
        ar_q = str(data.get("question", "")).strip()
        ar_choices = {k.upper(): str(v).strip() for k, v in (data.get("choices") or {}).items()}
        if not ar_q or not ar_choices:
            raise RuntimeError("incomplete live translation JSON")
        return ar_q, ar_choices

    return live
