"""Layer 1 — the common interface to all LLMs, backed by LiteLLM.

One adapter, not a provider matrix. Any provider (OpenAI, Anthropic, Azure) and
any local OpenAI-compatible server (vLLM via ``base_url``) is reached through the
same :class:`Model`. The judge is not special — it is another :class:`Model`.

Mirrors the structure of lighteval's ``LiteLLMClient``: build a kwargs dict for
``litellm.completion(**kwargs)``, run a batch through a ``ThreadPoolExecutor``,
retry with exponential backoff, and turn a content-filter / exhausted-retry into
an explicit not-ok response rather than a crash.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)

# Substring LiteLLM surfaces on an Azure/OpenAI content-management rejection.
_CONTENT_FILTER_MARKER = "content management policy"


@dataclass
class GenParams:
    """Unified inference params, handed straight to LiteLLM.

    These are the standardized pipeline choices (temperature 0, top_p 1, fixed
    seed) the harness applies across all models. ``response_format`` is set by
    the judge to request strict JSON.
    """

    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    stop: list[str] | None = None
    seed: int | None = 42
    response_format: dict | None = None

    def to_litellm_kwargs(self) -> dict:
        """Render the params as LiteLLM ``completion`` keyword arguments."""
        kwargs: dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            # Newer param name; LiteLLM maps it to legacy ``max_tokens`` per
            # provider. ``drop_params`` removes it where unsupported.
            "max_completion_tokens": self.max_tokens,
        }
        if self.stop:
            kwargs["stop"] = self.stop
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format
        return kwargs


@dataclass
class Response:
    """A single generation result.

    ``ok`` is ``False`` when the model produced nothing usable — a content-filter
    rejection or an exhausted-retry failure. Downstream the scorer treats a
    not-ok judge response as ``skipped`` (excluded from numerator and
    denominator); a not-ok model-under-test response is an empty answer (counts
    as incorrect under the denominator policy).
    """

    text: str
    ok: bool = True
    usage: dict | None = None
    raw: object = field(default=None, repr=False)


class Model:
    """One model adapter over ``litellm.completion``.

    Args:
        model: LiteLLM model string with provider prefix, e.g.
            ``"openai/gpt-4o"``, ``"anthropic/claude-sonnet-4-20250514"``,
            ``"azure/<deployment>"``. For a local vLLM server use
            ``"openai/<served-name>"`` together with ``base_url``.
        base_url: Endpoint override (set this for a local vLLM server).
        api_key: Key override; if ``None``, LiteLLM reads provider env vars.
        concurrency: Max parallel requests in :meth:`generate_batch`.
        num_retries / retry_sleep / retry_multiplier: Exponential-backoff retry
            policy (defaults mirror lighteval's LiteLLM client).
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        concurrency: int = 8,
        num_retries: int = 8,
        retry_sleep: float = 1.0,
        retry_multiplier: float = 2.0,
        timeout: float | None = 120.0,
        api_version: str | None = None,
        extra: dict | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        # api_version is required by Azure OpenAI; extra passes any additional
        # provider-specific completion kwargs straight through to LiteLLM.
        self.api_version = api_version
        self.extra = extra or {}
        self.concurrency = concurrency
        self.num_retries = num_retries
        self.retry_sleep = retry_sleep
        self.retry_multiplier = retry_multiplier
        self.timeout = timeout

        # Imported lazily so importing the package costs nothing until a real
        # call is made (and so tests can monkeypatch ``litellm``).
        import litellm

        self._litellm = litellm
        # Silently drop params a given provider does not support (e.g. ``seed``,
        # ``max_completion_tokens`` on some backends) instead of erroring.
        litellm.drop_params = True

    # -- single call ---------------------------------------------------------

    def generate(self, messages: list[dict], params: GenParams) -> Response:
        """Generate one completion with retries. Never raises on API failure."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "timeout": self.timeout,
            **({"api_version": self.api_version} if self.api_version else {}),
            **params.to_litellm_kwargs(),
            **self.extra,
        }

        last_err: Exception | None = None
        for attempt in range(self.num_retries):
            try:
                resp = self._litellm.completion(**kwargs)
                content = resp.choices[0].message.content or ""
                usage = _usage_dict(resp)
                return Response(text=content, ok=bool(content), usage=usage, raw=resp)
            except Exception as e:  # noqa: BLE001 — normalize all provider errors
                # Content-filter rejection: do not retry, return not-ok.
                if _is_content_filter(e):
                    logger.warning("Content filtered for %s; returning empty.", self.model)
                    return Response(text="", ok=False, raw=e)
                last_err = e
                wait = min(64.0, self.retry_sleep * (self.retry_multiplier**attempt))
                logger.warning(
                    "API error (%s), retry %d/%d in %.1fs",
                    e,
                    attempt + 1,
                    self.num_retries,
                    wait,
                )
                time.sleep(wait)

        logger.error("API call failed after %d attempts: %s", self.num_retries, last_err)
        return Response(text="", ok=False, raw=last_err)

    # -- batched calls -------------------------------------------------------

    def generate_batch(self, batch: list[list[dict]], params: GenParams) -> list[Response]:
        """Generate for many message-lists concurrently, preserving order.

        Mirrors lighteval's ``__call_api_parallel``: a thread pool over the
        per-request :meth:`generate`, results collected in input order.
        """
        if not batch:
            return []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(lambda m: self.generate(m, params), batch))


def _usage_dict(resp: object) -> dict | None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _is_content_filter(err: Exception) -> bool:
    msg = str(getattr(err, "message", "")) or str(err)
    return _CONTENT_FILTER_MARKER in msg.lower()
