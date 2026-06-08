"""Layer 2 — task structure (lighteval-shaped).

A :class:`Sample` is one ``(input, target)`` item; a :class:`Task` bundles the
unified prompt/system prompt, the calibrated token budget, a dataset loader, and
the scorer kind that selects judge rules + metrics. Per-benchmark variation lives
as *data* in the registry (``registry.py``), not as ``Task`` subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Sample:
    """One benchmark item.

    Attributes:
        index: Stable position in the dataset; used to align judge runs.
        prompt: The full benchmark question text (already rendered).
        target: Ground truth. May be ``""`` where the benchmark ships none
            (e.g. ``cti_taa``); such items are still attempted and judged.
        choices: MCQ options when the benchmark provides them, else ``None``.
        metadata: Anything else carried through to the detailed output.
    """

    index: int
    prompt: str
    target: str = ""
    choices: list[str] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Task:
    """A benchmark sub-task.

    Attributes:
        name: Registry key / ``--tasks`` flag name (e.g. ``"mcq"``).
        task_type: Scoring family that selects judge ``format_hint`` /
            ``compare_rule`` and the corpus metric (e.g. ``mcq``, ``seceval``,
            ``vsp``, ``taa``). Ported from ``get_task_type``.
        system_prompt: System message applied where the benchmark specifies one.
        max_tokens: Per-task calibrated generation budget.
        loader: Zero-arg callable returning the list of :class:`Sample`. Kept
            lazy so importing the registry does not trigger dataset downloads.
        scorer_kind: Usually equal to ``task_type``; kept separate so several
            tasks can share one scoring family.
    """

    name: str
    task_type: str
    loader: Callable[[], list["Sample"]]
    system_prompt: str | None = None
    max_tokens: int = 1024
    scorer_kind: str = ""

    def __post_init__(self) -> None:
        if not self.scorer_kind:
            self.scorer_kind = self.task_type

    def load(self, max_samples: int | None = None) -> list["Sample"]:
        samples = self.loader()
        if max_samples is not None:
            samples = samples[:max_samples]
        return samples
