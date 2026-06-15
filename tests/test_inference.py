import json
import pytest

from localevidence import inference


class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_parse_model():
    assert inference.parse_model("ollama:qwen2.5:14b") == ("ollama", "qwen2.5:14b")
    assert inference.parse_model("") == (None, None)
    assert inference.parse_model(None) == (None, None)  # falls to DEFAULT_MODEL ("")


def test_generate_no_model_raises():
    with pytest.raises(inference.InferenceError):
        inference.generate("hi", model="")


def test_generate_unknown_backend_raises():
    with pytest.raises(inference.InferenceError):
        inference.generate("hi", model="openai:gpt")


def test_generate_ollama_path(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        return _FakeResp({"message": {"content": "  grounded answer [s1#0]  "}})

    monkeypatch.setattr(inference.urllib.request, "urlopen", fake_urlopen)
    out = inference.generate("Q?", system="sys", model="ollama:qwen2.5:14b")
    assert out == "grounded answer [s1#0]"
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["model"] == "qwen2.5:14b"
    assert captured["body"]["messages"][0]["role"] == "system"


def test_anthropic_backend(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cap = {}
    def fake_urlopen(req, timeout=0):
        cap["url"] = req.full_url
        cap["key"] = req.get_header("X-api-key")
        cap["body"] = json.loads(req.data)
        return _FakeResp({"content": [{"type": "text", "text": "grounded reply"}]})
    monkeypatch.setattr(inference.urllib.request, "urlopen", fake_urlopen)
    out = inference.generate("Q?", system="sys", model="anthropic:claude-sonnet-4-6")
    assert out == "grounded reply"
    assert cap["url"].endswith("/v1/messages") and cap["key"] == "sk-test"
    assert cap["body"]["model"] == "claude-sonnet-4-6" and cap["body"]["system"] == "sys"


def test_anthropic_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(inference.InferenceError):
        inference.generate("Q?", model="anthropic:claude-sonnet-4-6")


def test_openai_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cap = {}
    def fake_urlopen(req, timeout=0):
        cap["url"] = req.full_url
        cap["auth"] = req.get_header("Authorization")
        return _FakeResp({"choices": [{"message": {"content": "gpt reply"}}]})
    monkeypatch.setattr(inference.urllib.request, "urlopen", fake_urlopen)
    out = inference.generate("Q?", model="openai:gpt-4o")
    assert out == "gpt reply"
    assert cap["url"].endswith("/v1/chat/completions") and cap["auth"] == "Bearer sk-test"


def test_openai_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(inference.InferenceError):
        inference.generate("Q?", model="openai:gpt-4o")


def test_format_passages_numbers_and_cites():
    txt = inference._format_passages(
        [{"id": "s1#3", "paper": "Paper One", "doi": "10.1/x", "text": "the dose is 50 mg/kg"}])
    assert "[s1#3]" in txt and "10.1/x" in txt and "50 mg/kg" in txt


def test_synthesize_answer_builds_grounded_prompt(monkeypatch):
    seen = {}

    def fake_generate(prompt, *, system=None, model=None, **kw):
        seen["prompt"], seen["system"] = prompt, system
        return "Ceftriaxone is appropriate [s1#3]."

    monkeypatch.setattr(inference, "generate", fake_generate)
    out = inference.synthesize_answer(
        "Empiric therapy for paediatric bacterial meningitis?",
        [{"id": "s1#3", "paper": "P", "doi": "10.1/x", "text": "ceftriaxone 50 mg/kg"}],
        model="ollama:qwen2.5:14b")
    assert out["grounded"] is True and out["n_passages"] == 1
    assert "[s1#3]" in out["answer"]
    assert "ceftriaxone 50 mg/kg" in seen["prompt"]    # passages went into the prompt
    assert "ONLY" in seen["system"]                     # grounding instruction present


def test_synthesize_answer_no_passages_is_not_grounded():
    out = inference.synthesize_answer("Q?", [], model="ollama:qwen2.5:14b")
    assert out["grounded"] is False and out["n_passages"] == 0
