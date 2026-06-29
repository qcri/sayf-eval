"""Offline in-process model backend (server-less vLLM).

The default :class:`sayf_eval.model.Model` reaches every model through a LiteLLM
endpoint (a vLLM/SGLang server or a hosted provider). This module adds an
*offline* alternative that loads weights into the eval process and batches with
vLLM's in-process ``LLM.chat`` — the same pattern as the project's offline batched
jobs (``seed_mini/translate_en_gemma_offline.py``): construct one ``LLM`` with a
tensor-parallel / gpu-memory config, then ``llm.chat(list_of_conversations,
SamplingParams(...))``.

It exposes the same ``model`` attribute and ``generate`` / ``generate_batch``
methods as :class:`~sayf_eval.model.Model`, so it is a drop-in for the
model-under-test, the judge, or the translator (``LLMTranslator`` takes any object
with that shape). No HTTP server required; fully air-gapped.
"""

from __future__ import annotations

import logging

from sayf_eval.model import GenParams, Response


logger = logging.getLogger(__name__)


class OfflineVLLMModel:
    """In-process vLLM backend, API-compatible with :class:`~sayf_eval.model.Model`.

    Args:
        model: HF id or local path to the checkpoint (``LLM(model=...)``).
        tensor_parallel_size: GPUs to shard across (``TP_SIZE``).
        gpu_memory_utilization: vLLM KV-cache fraction (``GPU_MEM_UTIL``).
        max_model_len: context window; ``None`` lets vLLM read the model config.
        trust_remote_code: pass through for models with custom code.
        dtype: ``"auto"`` by default; override for e.g. ``"bfloat16"``.
        extra_llm_kwargs: any additional ``vllm.LLM`` kwargs (quantization, etc.).

    The constructor loads the weights eagerly (like the offline jobs' module-level
    ``llm = LLM(...)``), so build it once and reuse across tasks.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        extra_llm_kwargs: dict | None = None,
    ) -> None:
        self.model = model
        # Imported lazily so the package (and the endpoint path) costs nothing
        # when vLLM isn't installed — mirrors Model's lazy litellm import.
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        kwargs: dict = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            **(extra_llm_kwargs or {}),
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        logger.info("[vLLM offline] loading %s (tp=%d, gpu_mem=%.2f, max_len=%s)",
                    model, tensor_parallel_size, gpu_memory_utilization, max_model_len)
        self._llm = LLM(**kwargs)

    # -- params --------------------------------------------------------------

    def _sampling(self, params: GenParams):
        return self._SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            stop=params.stop or None,
            seed=params.seed,
        )

    # -- single / batched ----------------------------------------------------

    def generate(self, messages: list[dict], params: GenParams) -> Response:
        return self.generate_batch([messages], params)[0]

    def generate_batch(self, batch: list[list[dict]], params: GenParams) -> list[Response]:
        """Batch a list of conversations through ``llm.chat`` (input order kept)."""
        if not batch:
            return []
        outs = self._llm.chat(batch, sampling_params=self._sampling(params))
        responses: list[Response] = []
        for o in outs:
            text = o.outputs[0].text if getattr(o, "outputs", None) else ""
            responses.append(Response(text=text, ok=bool(text), raw=o))
        return responses
