"""Assemble the evidence pack a clinician (or Claude) synthesises from.

Output is deliberately NOT a finished answer — it is grounded raw material:
ranked passages grouped by source paper, a coverage report (the compounding
signal), and a gap log of what we wanted but could not get. The synthesis step
(answer-style.md + a faithfulness check) happens after, against this pack.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config
from .acquire import AcquireReport
from .triage import TriageResult
from .index import Passage


_TIER_ORDER = ["guideline", "systematic-review", "rct", "cohort", "review",
               "article", "other", ""]


def _fmt_authors(authors: str, n: int = 3) -> str:
    parts = [a.strip() for a in (authors or "").replace(";", ",").split(",") if a.strip()]
    if not parts:
        return ""
    if len(parts) <= n:
        return ", ".join(parts)
    return ", ".join(parts[:n]) + " et al."


@dataclass
class EvidencePack:
    markdown: str
    coverage: dict
    gaps: list[dict]
    n_passages: int
    n_papers: int


def _coverage(triage_result: TriageResult, acq: AcquireReport,
              n_candidates: int, n_passages: int) -> dict:
    return {
        "candidates_discovered": n_candidates,
        "relevant_above_floor": len(triage_result.to_acquire) + len(triage_result.in_library),
        "below_relevance_floor": triage_result.below_floor,
        "already_in_library": acq.from_library + acq.already_have,
        "newly_pulled": acq.pulled,
        "could_not_acquire": acq.no_oa + acq.not_found + acq.wrong_paper_only,
        "budget_skipped": acq.budget_skipped,
        "papers_indexed": len(acq.papers),
        "passages_indexed": n_passages,
    }


def build_pack(
    question: str,
    *,
    triage_result: TriageResult,
    acquire_report: AcquireReport,
    passages: list[Passage],
    n_candidates: int,
    n_passages_total: int,
    passages_per_paper: int = 4,
) -> EvidencePack:
    cov = _coverage(triage_result, acquire_report, n_candidates, n_passages_total)

    # Group retrieved passages by source paper, preserving retrieval order.
    by_slug: dict[str, list[Passage]] = {}
    order: list[str] = []
    for p in passages:
        if p.slug not in by_slug:
            by_slug[p.slug] = []
            order.append(p.slug)
        by_slug[p.slug].append(p)

    # paper metadata lookup from the acquire report
    meta = {ap.slug: ap for ap in acquire_report.papers}

    # order papers by best evidence tier then by best passage score
    def paper_sort_key(slug: str):
        ps = by_slug[slug]
        tier = ps[0].tier
        tier_rank = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else len(_TIER_ORDER)
        best = max(p.score for p in ps)
        return (tier_rank, -best)

    order.sort(key=paper_sort_key)

    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L: list[str] = []
    L.append(f"# Evidence pack — {question.strip()}")
    L.append("")
    L.append(f"_Generated {stamp} by LocalEvidence. Grounded source material for "
             f"clinician/Claude synthesis — not a finished answer._")
    L.append("")

    # Coverage / compounding signal
    L.append("## Coverage")
    L.append("")
    L.append(f"- Candidates discovered: **{cov['candidates_discovered']}** "
             f"({cov['below_relevance_floor']} dropped below relevance floor)")
    L.append(f"- Papers indexed: **{cov['papers_indexed']}** "
             f"({cov['already_in_library']} already in library, "
             f"**{cov['newly_pulled']} newly pulled**)")
    L.append(f"- Passages indexed: **{cov['passages_indexed']}**")
    if cov["could_not_acquire"]:
        L.append(f"- Could not acquire: {cov['could_not_acquire']} "
                 f"(see Gaps)")
    L.append("")

    # Evidence, grouped by paper
    L.append("## Evidence")
    L.append("")
    for slug in order:
        ps = by_slug[slug][:passages_per_paper]
        ap = meta.get(slug)
        title = ps[0].title or (ap.title if ap else slug)
        year = ps[0].year or (ap.year if ap else "")
        journal = ps[0].journal or (ap.journal if ap else "")
        doi = ps[0].doi or (ap.doi if ap else "")
        authors = _fmt_authors(ap.authors if ap else "")
        tier = ps[0].tier or "—"
        cite = " · ".join(x for x in [authors, journal, str(year)] if x)
        L.append(f"### {title}")
        head = f"_{tier}_"
        if cite:
            head += f" — {cite}"
        if doi:
            head += f" — doi:[{doi}](https://doi.org/{doi})"
        L.append(head)
        L.append("")
        for p in ps:
            snippet = " ".join(p.text.split())
            if len(snippet) > 600:
                snippet = snippet[:600].rsplit(" ", 1)[0] + " …"
            L.append(f"> {snippet}")
            L.append("")
    if not order:
        L.append("_No passages retrieved — corpus gap. See Gaps._")
        L.append("")

    # Gaps
    gaps: list[dict] = []
    for f in acquire_report.failures:
        gaps.append({"doi": f.get("doi", ""), "reason": "error", "note": f.get("error", "")})
    # surface the to_acquire candidates that did not end up indexed
    indexed_slugs = {ap.slug for ap in acquire_report.papers}
    for c in triage_result.to_acquire:
        if config.slugify_doi(c.doi) not in indexed_slugs:
            gaps.append({"doi": c.doi, "title": c.title, "reason": "not_acquired",
                         "tier": c.tier})
    if gaps:
        L.append("## Gaps (wanted, not retrieved)")
        L.append("")
        for g in gaps[:30]:
            t = g.get("title", "")
            L.append(f"- `{g.get('doi','')}` {('— ' + t[:70]) if t else ''} "
                     f"({g.get('reason','')})")
        L.append("")
        L.append("_These feed the next acquisition pass; re-running `ask` retries them._")
        L.append("")

    # Sources
    L.append("## Sources")
    L.append("")
    for slug in order:
        ap = meta.get(slug)
        ps = by_slug[slug]
        title = ps[0].title or (ap.title if ap else slug)
        doi = ps[0].doi or (ap.doi if ap else "")
        src = ap.source if ap else ""
        link = f" — https://doi.org/{doi}" if doi else ""
        prov = f" _(via {src})_" if src else ""
        L.append(f"- {title}{link}{prov}")
    L.append("")

    return EvidencePack(
        markdown="\n".join(L),
        coverage=cov,
        gaps=gaps,
        n_passages=len(passages),
        n_papers=len(order),
    )
