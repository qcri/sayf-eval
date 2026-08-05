"""Task registry — per-benchmark configuration as data.

The registry maps a task name to a :class:`~sayf_eval.task.Task`. Each entry binds
the unified prompt/system prompt, the calibrated token budget, a lazy dataset
loader (from ``datasets.py``), and the scorer kind. The full suite is registered
by importing ``sayf_eval.tasks``. This module stays import-light: loaders are
referenced, not invoked, so importing the registry triggers no dataset downloads.
"""

from __future__ import annotations

from sayf_eval.task import Task


# Populated by importing ``sayf_eval.tasks`` (which calls ``register`` for each
# task). Kept empty here so importing the registry alone pulls in no loaders.
TASKS: dict[str, Task] = {}


def register(task: Task) -> Task:
    """Add a task to the registry (idempotent overwrite by name)."""
    TASKS[task.name] = task
    return task


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"Unknown task {name!r}. Registered: {sorted(TASKS)}") from None


def available_tasks() -> list[str]:
    return sorted(TASKS)
