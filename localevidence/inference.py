"""Optional local-model synthesis — answer grounded clinical questions with a
free, on-device open-weight model instead of (or alongside) Claude-in-the-loop.

The whole point of the surrounding tool is that the *corpus and grounding* carry
the safety, not a particular model. So the model is a swappable backend: point it
at a local Ollama server and a 14B open-weight model on a laptop produces the same
grounded, cited answer a frontier API would — fully on-device, no paid API, no
data leaving the machine. If no model is configured, synthesis stays
Claude-in-the-loop (you read the evidence pack and write the answer); this module
just makes the autonomous, local path a first-class option.

Stdlib only. Ollama is reached over its local HTTP API; nothing here phones home.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional, Sequence

from . import config  # noqa: F401 — importing loads the gitignored .env (keys/model)

DEFAULT_MODEL = os.environ.get("LOCALEVIDENCE_MODEL", "")   # e.g. "ollama:qwen2.5:14b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Local generation can be slow under memory pressure (big model, long prompt). A
# generous default; override with LOCALEVIDENCE_TIMEOUT for very large models.
DEFAULT_TIMEOUT = int(os.environ.get("LOCALEVIDENCE_TIMEOUT", "600"))

_SYSTEM = (
    "You are a clinical evidence assistant. Answer the question USING ONLY the "
    "numbered evidence passages provided. Cite the passages you use by their id "
    "in square brackets, e.g. [slug#3]. If the passages do not support an answer, "
    "say so plainly rather than guessing. Do not introduce facts that are not in "
    "the passages. Be concise and clinically precise."
)


class InferenceError(RuntimeError):
    """Raised when a backend is misconfigured or unreachable (callers may fall
    back to Claude-in-the-loop)."""


def parse_model(spec: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """'ollama:qwen2.5:14b' -> ('ollama', 'qwen2.5:14b'). Empty -> (None, None)."""
    spec = (spec if spec is not None else DEFAULT_MODEL) or ""
    if not spec:
        return None, None
    backend, _, name = spec.partition(":")
    return backend.strip().lower() or None, name.strip() or None


def generate(prompt: str, *, system: Optional[str] = None,
             model: Optional[str] = None, host: Optional[str] = None,
             timeout: int = DEFAULT_TIMEOUT) -> str:
    """Generate text from the configured backend.

    `ollama:<name>` is local and free (the default stack). `anthropic:<model>` and
    `openai:<model>` are OPTIONAL paid-API backends for the cross-model evaluation
    arm only — off by default, the operator supplies their own key via
    ANTHROPIC_API_KEY / OPENAI_API_KEY. They exist so the same harness can compare
    how any AI handles a grounded question; they do not change the free-local core."""
    backend, name = parse_model(model)
    if backend == "ollama":
        return _ollama_chat(name, prompt, system, host or OLLAMA_HOST, timeout)
    if backend == "anthropic":
        return _anthropic_chat(name, prompt, system, timeout)
    if backend == "openai":
        return _openai_chat(name, prompt, system, timeout)
    if backend == "openrouter":
        return _openrouter_chat(name, prompt, system, timeout)
    if backend in (None, "manual", "claude"):
        raise InferenceError(
            "no model backend configured — set LOCALEVIDENCE_MODEL=ollama:<name> "
            "(e.g. ollama:qwen2.5:14b), or pass model=anthropic:<m> / openai:<m> for "
            "the eval arm; the zero-config default is Claude-in-the-loop synthesis.")
    raise InferenceError(f"unknown inference backend: {backend!r}")


def _ollama_chat(name: Optional[str], prompt: str, system: Optional[str],
                 host: str, timeout: int) -> str:
    if not name:
        raise InferenceError("ollama backend needs a model name, e.g. ollama:qwen2.5:14b")
    messages = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": prompt}]
    body = json.dumps({"model": name, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except Exception as e:  # connection refused, model not pulled, timeout, ...
        raise InferenceError(
            f"ollama request failed (host={host}, model={name}): {e}. "
            f"Is `ollama serve` running and the model pulled (`ollama pull {name}`)?")
    return ((data.get("message") or {}).get("content") or "").strip()


def _post_json(url: str, headers: dict, payload: dict, timeout: int, what: str) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:  # auth, rate-limit, network, bad model
        raise InferenceError(f"{what} request failed (model in spec): {e}")


def _anthropic_chat(name: Optional[str], prompt: str, system: Optional[str],
                    timeout: int, max_tokens: int = 2048) -> str:
    if not name:
        raise InferenceError("anthropic backend needs a model, e.g. anthropic:claude-sonnet-4-6")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise InferenceError("anthropic backend needs ANTHROPIC_API_KEY (operator-supplied; eval arm only)")
    payload = {"model": name, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    if system:
        payload["system"] = system
    data = _post_json("https://api.anthropic.com/v1/messages",
                      {"x-api-key": key, "anthropic-version": "2023-06-01"},
                      payload, timeout, "anthropic")
    parts = [b.get("text", "") for b in (data.get("content") or []) if b.get("type") == "text"]
    return "".join(parts).strip()


def _openai_style_chat(name: Optional[str], prompt: str, system: Optional[str],
                       timeout: int, *, url: str, key: str, label: str,
                       extra_headers: Optional[dict] = None) -> str:
    """Shared OpenAI-compatible Chat Completions call (OpenAI, OpenRouter, …)."""
    if not name:
        raise InferenceError(f"{label} backend needs a model name")
    messages = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": prompt}]
    headers = {"Authorization": f"Bearer {key}", **(extra_headers or {})}
    data = _post_json(url, headers, {"model": name, "messages": messages}, timeout, label)
    return ((data.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()


def _openai_chat(name: Optional[str], prompt: str, system: Optional[str], timeout: int) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise InferenceError("openai backend needs OPENAI_API_KEY (operator-supplied; eval arm only)")
    return _openai_style_chat(name, prompt, system, timeout,
                              url="https://api.openai.com/v1/chat/completions", key=key, label="openai")


def _openrouter_chat(name: Optional[str], prompt: str, system: Optional[str], timeout: int) -> str:
    # One key, ~every model (Claude/GPT/Llama/Qwen/…) — the efficient way to sweep
    # model GRADES (e.g. openrouter:qwen/qwen-2.5-72b-instruct) and simulate what a
    # local model of that size/family would do, without the hardware to run it.
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise InferenceError("openrouter backend needs OPENROUTER_API_KEY (operator-supplied; "
                             "one key, many models — e.g. openrouter:qwen/qwen-2.5-72b-instruct)")
    return _openai_style_chat(name, prompt, system, timeout,
                              url="https://openrouter.ai/api/v1/chat/completions", key=key,
                              label="openrouter", extra_headers={"X-Title": "LocalEvidence"})


def _format_passages(passages: Sequence[dict]) -> str:
    """Render verify-evidence / search passages into a numbered, cited context."""
    blocks = []
    for p in passages:
        pid = p.get("id") or f"{p.get('slug','?')}#{p.get('chunk_idx', 0)}"
        head = " ".join(x for x in (p.get("paper") or p.get("title", ""),
                                    f"({p['doi']})" if p.get("doi") else "") if x)
        blocks.append(f"[{pid}] {head}\n{p.get('text', '')}")
    return "\n\n".join(blocks)


def synthesize_answer(question: str, passages: Sequence[dict], *,
                      model: Optional[str] = None) -> dict:
    """Produce a grounded, cited answer to `question` from `passages` using the
    configured local model. Raises InferenceError if no backend is available."""
    if not passages:
        return {"answer": "No supporting passages were retrieved; cannot answer "
                          "from the local corpus.", "model": model or DEFAULT_MODEL,
                "n_passages": 0, "grounded": False}
    prompt = (f"Question: {question}\n\nEvidence passages:\n"
              f"{_format_passages(passages)}\n\n"
              "Write a grounded, cited answer using only these passages.")
    text = generate(prompt, system=_SYSTEM, model=model)
    return {"answer": text, "model": model or DEFAULT_MODEL,
            "n_passages": len(passages), "grounded": True}
