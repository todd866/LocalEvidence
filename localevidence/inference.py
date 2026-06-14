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

DEFAULT_MODEL = os.environ.get("LOCALEVIDENCE_MODEL", "")   # e.g. "ollama:qwen2.5:14b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

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
             timeout: int = 300) -> str:
    """Generate text from the configured backend. Currently: ollama (local)."""
    backend, name = parse_model(model)
    if backend == "ollama":
        return _ollama_chat(name, prompt, system, host or OLLAMA_HOST, timeout)
    if backend in (None, "manual", "claude"):
        raise InferenceError(
            "no local model configured — set LOCALEVIDENCE_MODEL=ollama:<name> "
            "(e.g. ollama:qwen2.5:14b) or pass model=...; the zero-config default "
            "is Claude-in-the-loop synthesis (read the evidence pack yourself).")
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
