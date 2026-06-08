"""seceval — lightweight, model-agnostic cybersecurity LLM benchmark framework.

Two layers, kept separate (see PROPOSAL.md):
  - Transport: LiteLLM. One ``Model`` adapter for every provider; local models
    are just a ``base_url``. The judge is another ``Model``.
  - Structure: lighteval-shaped. A ``Task``/``Scorer`` boundary with a two-level
    metric split (sample-level extract+verdict, corpus-level aggregation).
"""

from seceval.model import GenParams, Model, Response

__version__ = "0.1.0"

__all__ = ["GenParams", "Model", "Response", "__version__"]
