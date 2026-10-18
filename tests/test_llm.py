from types import SimpleNamespace
import pytest
from finpilot import config
from finpilot.llm import LLM, LLMUnavailable


def make_client(responses):
    calls = {"n": 0}

    def create(**kw):
        calls["n"] += 1
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


def test_json_parses():
    client, _ = make_client(['{"a": 1}'])
    assert LLM(client=client).json("s", "u") == {"a": 1}


def test_bad_json_raises_unavailable():
    client, _ = make_client(["not json"])
    with pytest.raises(LLMUnavailable):
        LLM(client=client).json("s", "u")


def test_retries_once_then_raises(monkeypatch):
    monkeypatch.setattr("finpilot.llm.time.sleep", lambda s: None)
    client, calls = make_client([RuntimeError("x"), RuntimeError("y")])
    with pytest.raises(LLMUnavailable):
        LLM(client=client).complete("s", "u")
    assert calls["n"] == 2


def test_retry_recovers(monkeypatch):
    monkeypatch.setattr("finpilot.llm.time.sleep", lambda s: None)
    client, _ = make_client([RuntimeError("x"), "ok"])
    assert LLM(client=client).complete("s", "u") == "ok"


def test_missing_key(monkeypatch):
    monkeypatch.delenv("MESH_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable):
        LLM().complete("s", "u")


def test_routes_through_mesh():
    assert "api.meshapi.ai" in config.MESH_BASE_URL
