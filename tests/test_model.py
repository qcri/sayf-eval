"""Model adapter tests — mock ``litellm`` so no network is needed."""

import sys
import types

import pytest

from seceval.model import GenParams, Model


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = types.SimpleNamespace(
            prompt_tokens=1, completion_tokens=2, total_tokens=3
        )


@pytest.fixture
def fake_litellm(monkeypatch):
    mod = types.ModuleType("litellm")
    mod.drop_params = False
    mod.calls = []

    def completion(**kwargs):
        mod.calls.append(kwargs)
        return mod._next(**kwargs)

    mod.completion = completion
    mod._next = lambda **kw: _Resp("A")
    monkeypatch.setitem(sys.modules, "litellm", mod)
    return mod


def test_generate_ok(fake_litellm):
    m = Model("openai/gpt-x")
    r = m.generate([{"role": "user", "content": "hi"}], GenParams(max_tokens=16))
    assert r.ok and r.text == "A"
    assert r.usage["total_tokens"] == 3
    # params propagated
    sent = fake_litellm.calls[-1]
    assert sent["model"] == "openai/gpt-x"
    assert sent["max_completion_tokens"] == 16
    assert sent["temperature"] == 0.0


def test_empty_response_is_not_ok(fake_litellm):
    fake_litellm._next = lambda **kw: _Resp("")
    m = Model("openai/gpt-x", num_retries=1)
    r = m.generate([{"role": "user", "content": "hi"}], GenParams())
    assert not r.ok and r.text == ""


def test_content_filter_short_circuits(fake_litellm):
    def boom(**kw):
        raise RuntimeError("blocked by Microsoft content management policy")

    fake_litellm._next = boom
    m = Model("openai/gpt-x", num_retries=5)
    r = m.generate([{"role": "user", "content": "x"}], GenParams())
    assert not r.ok
    # No retry storm: exactly one attempt for a content filter.
    assert len(fake_litellm.calls) == 1


def test_base_url_routes_local_vllm(fake_litellm):
    m = Model("openai/qwen", base_url="http://localhost:8000/v1", api_key="EMPTY")
    m.generate([{"role": "user", "content": "x"}], GenParams())
    sent = fake_litellm.calls[-1]
    assert sent["base_url"] == "http://localhost:8000/v1"
    assert sent["api_key"] == "EMPTY"


def test_generate_batch_preserves_order(fake_litellm):
    seq = iter(["A", "B", "C"])
    fake_litellm._next = lambda **kw: _Resp(next(seq))
    m = Model("openai/gpt-x", concurrency=1)
    out = m.generate_batch([[{"role": "user", "content": str(i)}] for i in range(3)], GenParams())
    assert [r.text for r in out] == ["A", "B", "C"]
