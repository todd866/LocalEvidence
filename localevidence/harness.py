"""A grounded-answer harness that pushes a small local model toward frontier-grade
clinical answers — by structure, not scale.

The bet of the whole tool is that the corpus and the grounding carry the quality
and the safety, not the model's raw capability. This module makes that concrete:
around a modest local model (e.g. qwen2.5:14b on a laptop) it wraps a multi-stage
loop that a frontier model would do implicitly —

    retrieve  ->  (query-expand)  ->  draft  ->  self-critique vs the passages
              ->  revise  ->  programmatic citation verification  ->  report

Each model step is a free, on-device call, so the harness can be run as hard as
you like. Citation verification is deterministic (no model), so "is every claim
actually tied to a retrieved passage?" is checked by code, not vibes.

Model calls go through inference.generate (swappable backend); retrieval is
injected, so the whole loop is unit-testable without a model or a corpus.
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

from . import inference

CITE_RE = re.compile(r"\[([a-z0-9][\w.\-]*#\d+)\]", re.I)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


# ── deterministic citation verification (no model) ──────────────────────────

def verify_citations(answer: str, passage_ids: Sequence[str]) -> dict:
    """Check every [slug#n] in the answer against the retrieved passage ids, and
    measure how much of the answer is actually grounded."""
    ids = set(passage_ids)
    cited = CITE_RE.findall(answer)
    valid = [c for c in cited if c in ids]
    invalid = [c for c in cited if c not in ids]
    sentences = [s for s in _SENT_RE.split(answer.strip()) if s.strip()]
    grounded = [s for s in sentences if any(c in ids for c in CITE_RE.findall(s))]
    coverage = round(len(grounded) / len(sentences), 3) if sentences else 0.0
    return {"cited": cited, "valid": sorted(set(valid)),
            "invalid": sorted(set(invalid)),
            "n_sentences": len(sentences), "n_grounded": len(grounded),
            "coverage": coverage, "hallucinated_citations": len(set(invalid))}


# ── model stages (each a single local call) ─────────────────────────────────

def expand_queries(question: str, *, model: Optional[str] = None,
                   n: int = 3) -> list[str]:
    """Ask the model for focused retrieval sub-queries to broaden grounding."""
    prompt = (f"A clinician asks: {question}\n\nList up to {n} short, distinct "
              "search queries that would retrieve the evidence needed to answer "
              "well (e.g. specific drugs, thresholds, complications). One per line, "
              "no numbering, no prose.")
    try:
        raw = inference.generate(prompt, model=model)
    except inference.InferenceError:
        return []
    qs = [re.sub(r"^[\-\*\d.\)\s]+", "", ln).strip() for ln in raw.splitlines()]
    return [q for q in qs if 3 < len(q) < 120][:n]


def draft_answer(question: str, passages: Sequence[dict], *,
                 model: Optional[str] = None) -> str:
    return inference.synthesize_answer(question, passages, model=model)["answer"]


def critique(question: str, draft: str, passages: Sequence[dict], *,
             model: Optional[str] = None) -> str:
    prompt = (
        f"Question: {question}\n\nEvidence passages:\n"
        f"{inference._format_passages(passages)}\n\nDraft answer:\n{draft}\n\n"
        "Critique the draft STRICTLY against the passages. List, as terse bullet "
        "points, every: (a) claim not supported by any passage, (b) missing "
        "clinically important caveat the passages do support, (c) citation that "
        "does not match a passage id. If the draft is fully grounded and complete, "
        "reply exactly: OK.")
    return inference.generate(prompt, model=model)


def revise(question: str, draft: str, crit: str, passages: Sequence[dict], *,
           model: Optional[str] = None) -> str:
    prompt = (
        f"Question: {question}\n\nEvidence passages:\n"
        f"{inference._format_passages(passages)}\n\nDraft answer:\n{draft}\n\n"
        f"Critique to address:\n{crit}\n\n"
        "Rewrite the answer to fix every point in the critique. Use ONLY the "
        "passages, cite each claim with its [slug#n] id, and drop any claim the "
        "passages do not support. Keep it concise and clinically precise.")
    return inference.generate(prompt, system=inference._SYSTEM, model=model)


# ── reasoning lane (for management / judgment / epidemiological questions) ──
# The grounded loop above optimises citation coverage and, on reasoning questions,
# strips legitimate clinical reasoning that isn't a citable corpus fact (it gave
# NEGATIVE lift on hard vignettes). This lane instead FORCES the reasoning steps a
# small model skips, and lets sound clinical reasoning stand even where the corpus
# is silent — while still grounding factual claims and running a safety self-check.

_REASON_SYSTEM = (
    "You are a careful clinical reasoner. Answer with explicit reasoning: address the "
    "most dangerous / can't-miss possibility FIRST; when a test is involved, reason "
    "about pre-test probability and what a result actually changes (a positive in a "
    "low-probability setting is often a FALSE positive); adjust for EACH comorbidity "
    "named. Ground factual claims in the provided passages and cite them [slug#n], but "
    "DO include sound clinical reasoning even where the passages are silent — state it "
    "plainly as reasoning. Be specific, safe, and decisive.")


def frame(question: str, *, model: Optional[str] = None) -> str:
    """Force the examiner-level framing the model otherwise skips."""
    prompt = (
        "For the following clinical question, state BRIEFLY (a few lines):\n"
        "1. The single most dangerous / can't-miss diagnosis or outcome to address first.\n"
        "2. The key decision and its main trade-off.\n"
        "3. If a test is involved: the rough pre-test probability and what a positive vs "
        "negative result would actually change (weigh false positives when probability is low).\n"
        "4. Any comorbidities and how each changes the standard approach.\n\n"
        f"Question: {question}")
    return inference.generate(prompt, model=model)


def reasoned_draft(question: str, frame_text: str, passages: Sequence[dict], *,
                   model: Optional[str] = None) -> str:
    prompt = (f"Question: {question}\n\nReasoning frame to follow:\n{frame_text}\n\n"
              f"Evidence passages (cite factual claims with [slug#n]):\n"
              f"{inference._format_passages(passages)}\n\n"
              "Write the answer, following the frame.")
    return inference.generate(prompt, system=_REASON_SYSTEM, model=model)


def safety_check(question: str, answer: str, *, model: Optional[str] = None) -> str:
    prompt = (
        f"Question: {question}\n\nAnswer to audit:\n{answer}\n\n"
        "As a demanding examiner, list as terse bullets any: (a) most-dangerous / can't-miss "
        "diagnosis not addressed, (b) test reasoning that ignores pre-test probability / base "
        "rates, (c) comorbidity not accounted for, (d) clinically unsafe or incorrect statement. "
        "If none, reply exactly: OK.")
    return inference.generate(prompt, model=model)


def reasoning_answer(question: str, *,
                     retrieve: Callable[[str, int], list[dict]],
                     model: Optional[str] = None, k: int = 8) -> dict:
    """Scaffolded reasoning loop: frame -> retrieve -> reasoned draft -> safety
    self-check -> revise -> verify. Returns the answer + full stages."""
    stages = ["frame"]
    fr = frame(question, model=model)
    passages = list(retrieve(question, k))
    stages.append("retrieve")
    if not passages:
        # reasoning can still proceed with no corpus support; note it
        passages = []
    draft = reasoned_draft(question, fr, passages, model=model)
    stages.append("draft")
    answer = draft
    chk = safety_check(question, draft, model=model)
    stages.append("safety-check")
    if chk.strip().upper() != "OK":
        prompt = (f"Question: {question}\n\nDraft:\n{draft}\n\n"
                  f"Safety/completeness gaps to fix:\n{chk}\n\n"
                  f"Evidence passages:\n{inference._format_passages(passages)}\n\n"
                  "Rewrite the answer to fix EVERY gap: address the most dangerous possibility "
                  "first, reason about base rates for any test, adjust for each comorbidity, and "
                  "cite factual claims.")
        answer = inference.generate(prompt, system=_REASON_SYSTEM, model=model)
        stages.append("revise")
    report = verify_citations(answer, [p["id"] for p in passages])
    stages.append("verify")
    return {"question": question, "answer": answer, "frame": fr, "safety_check": chk,
            "grounding": report, "grounded": report["coverage"] > 0,
            "passages": passages, "n_passages": len(passages), "stages": stages,
            "model": model or inference.DEFAULT_MODEL}


# ── the loop ────────────────────────────────────────────────────────────────

def grounded_answer(question: str, *,
                    retrieve: Callable[[str, int], list[dict]],
                    model: Optional[str] = None, k: int = 8,
                    expand: bool = True, reflect: bool = True,
                    max_passages: int = 12) -> dict:
    """Run the full harness. `retrieve(query, k) -> [passage dicts]` is injected
    (wrap PassageIndex.search at the call site). Returns the answer plus a full
    grounding report and every intermediate stage for inspection."""
    stages: list[str] = []

    # 1. retrieve (+ optional model-proposed sub-queries, de-duplicated by id)
    passages = list(retrieve(question, k))
    stages.append("retrieve")
    if expand:
        seen = {p["id"] for p in passages}
        for sq in expand_queries(question, model=model):
            for p in retrieve(sq, max(2, k // 2)):
                if p["id"] not in seen and len(passages) < max_passages:
                    seen.add(p["id"]); passages.append(p)
        stages.append("expand")
    pids = [p["id"] for p in passages]

    if not passages:
        # Same key set as the normal return, so callers (eval.run_eval,
        # synthesize --harness) never KeyError on an empty/off-topic corpus.
        return {"question": question,
                "answer": "No supporting passages were retrieved; cannot answer "
                          "from the local corpus.", "grounded": False,
                "grounding": verify_citations("", []), "passages": [], "n_passages": 0,
                "stages": stages, "model": model or inference.DEFAULT_MODEL}

    # 2. draft
    draft = draft_answer(question, passages, model=model)
    stages.append("draft")
    answer, crit = draft, None

    # 3. self-critique + revise
    if reflect:
        crit = critique(question, draft, passages, model=model)
        stages.append("critique")
        if crit.strip().upper() != "OK":
            answer = revise(question, draft, crit, passages, model=model)
            stages.append("revise")

    # 4. deterministic citation verification
    report = verify_citations(answer, pids)
    stages.append("verify")

    return {"question": question, "answer": answer, "draft": draft,
            "critique": crit, "grounding": report, "grounded": report["coverage"] > 0,
            "passages": passages, "n_passages": len(passages), "stages": stages,
            "model": model or inference.DEFAULT_MODEL}
