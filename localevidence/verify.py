"""Judgment-free claim-verification service.

Given a claim's text, retrieve supporting/contradicting passages from the
LocalEvidence corpus and return them with a retrieval-strength confidence, a
citation-provenance check, and a corpus version stamp. The *verdict* (does a
passage entail the claim?) is the caller's job — this module never judges.

Used by `localevidence serve`'s POST /api/verify-evidence and the `verify` CLI.
"""
from __future__ import annotations

from typing import Callable, Optional

_NEG = (" no ", " not ", " without ", " absence of ", " contraindicat",
        " should not ", " avoid ", " ineffective ", " did not ")


def build_query(claim: dict) -> str:
    topics = claim.get("topics") or []
    if isinstance(topics, str):           # a bare string is one topic, not chars
        topics = [topics]
    parts = [claim.get("text", ""), claim.get("context", "") or ""]
    parts += [str(t) for t in topics]
    return " ".join(p for p in parts if p).strip()


def confidence(passages) -> float:
    """Coverage signal: the top passage's dense query·passage cosine, clamped to
    0–1. Discriminative (≈0.5–0.7 on-topic, ≈0.2 off-topic) — unlike the fused RRF
    rank score, which saturates at 1.0 for any double-ranked hit and so can't gate
    acquire-on-miss. A retrieval-strength signal, NOT a calibrated probability."""
    if not passages:
        return 0.0
    top = max(p.cosine for p in passages)
    return round(max(0.0, min(1.0, top)), 3)


def corpus_version(index) -> str:
    s = index.stats()
    return f"le-{s['papers']}-{s['passages']}"


def _negation_hint(text: str) -> str:
    """Whether the passage contains negation language. A crude LEXICAL flag, NOT a
    claim-aware stance/entailment verdict — a passage saying 'no significant
    difference' may well SUPPORT a 'no difference' claim. Named 'negation' (not
    'stance') so callers don't overread it as contradiction. Real entailment is a
    cross-encoder, later."""
    t = f" {text.lower()} "
    return "has_negation" if any(n in t for n in _NEG) else "none"


def _passage_view(p, max_chars: int = 600) -> dict:
    return {"id": f"{p.slug}#{p.chunk_idx}", "paper": p.title, "doi": p.doi,
            "year": p.year, "tier": p.tier,
            "text": " ".join(p.text.split())[:max_chars],
            "score": round(p.score, 5), "negation_hint": _negation_hint(p.text)}


def citation_check(citation: dict, passages, index) -> dict:
    """Citation PROVENANCE only: did the cited source actually surface for this
    claim? `supports` stays None — claim-support is the judge's call (the caller)."""
    from .audit import _norm_doi, _title_tokens
    doi = _norm_doi(citation.get("doi", "")) if citation.get("doi") else ""
    title = citation.get("title", "") or ""
    retrieved_dois = {_norm_doi(p.doi) for p in passages if p.doi}
    if doi and doi in retrieved_dois:
        return {"status": "found", "supports": None, "matched_doi": doi,
                "note": "cited DOI present in retrieved set (presence, not support)"}
    if title:
        want = _title_tokens(title)
        for p in passages:
            got = _title_tokens(p.title)
            if want and len(want & got) / max(1, len(want)) >= 0.6:
                return {"status": "found", "supports": None, "matched_doi": p.doi,
                        "note": "matched cited source by title overlap"}
    return {"status": "absent", "supports": None, "matched_doi": None,
            "note": "cited source not in the retrieved evidence for this claim"}


def verify_evidence(claim: dict, *, index, citation: Optional[dict] = None,
                    k: int = 8, acquire_on_miss: bool = False,
                    min_confidence: float = 0.45, importance: int = 1,
                    acquirer: Optional[Callable[[str], dict]] = None) -> dict:
    q = build_query(claim)
    passages = index.search(q, k=k)
    conf = confidence(passages)
    acquired = {"ran": False, "pulled": 0, "topic": None}

    if (conf < min_confidence and acquire_on_miss and importance >= 2
            and acquirer is not None):
        topics = claim.get("topics") or [q]
        topic = (" ".join(topics) if isinstance(topics, list) else str(topics))[:80]
        res = acquirer(topic) or {}
        if hasattr(index, "reload"):
            index.reload()  # see the freshly-pulled passages before re-searching
        passages = index.search(q, k=k)
        conf = confidence(passages)
        acquired = {"ran": True, "pulled": int(res.get("pulled", 0)), "topic": topic}

    cc = citation_check(citation, passages, index) if citation else \
        {"status": "n/a", "supports": None, "matched_doi": None,
         "note": "no citation supplied"}

    return {"passages": [_passage_view(p) for p in passages],
            "citation_check": cc, "confidence": conf, "acquired": acquired,
            "corpus_version": corpus_version(index)}
