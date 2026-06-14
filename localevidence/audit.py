"""The audit layer — emit a reconstructable provenance trail for a worked answer.

Auditability is LocalEvidence's differentiator and a safety property: every answer
should ship the chain that produced it, re-checkable by an independent party.
`audit_entry` assembles that trail from the ledger entry plus the run checkpoints,
and runs the **citation provenance check**:

    were the sources the answer cites actually retrieved by THIS session, or
    introduced from the model's parametric memory?

A cited source — by DOI, or by name matched to a retrieved paper's title — that is
NOT in the session's retrieved evidence is flagged. This verifies *retrieval
presence*, NOT that the source supports the specific claim (claim-level support is
the manual step in a full audit); the output says so, to avoid false reassurance.
It also reports a "verification ceiling": how far back the answer can be
independently reconstructed (the `_RUNGS` ladder below). For LocalEvidence with a
full run, that is the top rung — the whole session re-runs from the trail.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from . import config

# DOIs may contain balanced parentheses (Lancet/Elsevier S-series, old Wiley SICI:
# 10.1016/S0140-6736(16)31678-6). Allow them in the body; strip a trailing markdown
# wrapper ")" only when unbalanced. Stop at whitespace, "]" (so [DOI] strips), and
# quote/angle brackets.
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\]\"'<>]+")
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_STOP = {
    "the", "and", "for", "with", "guideline", "guidelines", "update", "statement",
    "position", "review", "systematic", "management", "treatment", "clinical",
    "practice", "study", "trial", "children", "paediatric", "pediatric", "acute",
    "national", "society", "from", "evidence",
}

_RUNGS = {
    0: "no answer", 1: "answer only", 2: "answer + citations",
    3: "sources retrievable", 4: "claim->source map", 5: "retrieval set visible",
    6: "retrieval reproducible", 7: "corpus inspectable", 8: "reasoning trace",
    9: "end-to-end re-runnable",
}


def _norm_doi(d: str) -> str:
    d = (d or "").strip().lower().rstrip(".,;:")
    while d.endswith(")") and d.count("(") < d.count(")"):
        d = d[:-1]
    return d


def _answer_dois(text: str) -> list[str]:
    return [_norm_doi(d) for d in DOI_RE.findall(text or "")]


def _title_tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s or "")
            if len(t) > 3 and t.lower() not in _STOP}


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _ceiling(answer: str, evidence: list, have_run: bool) -> dict:
    # Reachable buckets: 0 (nothing), 1 (answer), 2 (+citations), 5 (+retrieval
    # set visible / sources retrievable / claim->source), 9 (+re-runnable run).
    rung = 0
    if answer:
        rung = 1
    if answer and (DOI_RE.search(answer) or _BRACKET_RE.search(answer)):
        rung = 2
    if evidence:
        rung = max(rung, 5)
    if have_run and answer:
        rung = 9
    return {"rung": rung, "of": 9, "label": _RUNGS[rung]}


def _resolve(dois: list[str]) -> dict:
    """Optional live existence check via doi.org (network). Best-effort."""
    out: dict = {}
    try:
        import requests
    except Exception:
        return {d: None for d in dois}
    for d in dois:
        try:
            r = requests.head(f"https://doi.org/{d}", allow_redirects=False, timeout=15,
                              headers={"User-Agent": f"localevidence-audit (mailto:{config.CONTACT_EMAIL})"})
            out[d] = r.status_code in (200, 301, 302, 303, 307, 308)
        except Exception:
            out[d] = None
    return out


def _provenance(answer: str, evidence: list) -> dict:
    """Match the answer's citations (DOI or named) against the retrieved evidence.

    Returns matched/unmatched for DOIs and for name-style citations. An unmatched
    citation is one whose source was NOT retrieved this session — the candidate for
    a citation introduced from memory. This is a retrieval-presence check, not a
    claim-support check.
    """
    ev_dois = {_norm_doi(e["doi"]) for e in evidence if e.get("doi")}
    ev_titles = [_title_tokens(e.get("title", "")) for e in evidence]
    ev_titles = [t for t in ev_titles if t]

    cited_dois = sorted(set(_answer_dois(answer)))
    doi_matched = [d for d in cited_dois if d in ev_dois]
    doi_unmatched = [d for d in cited_dois if d not in ev_dois]

    # named citations: bracketed source tokens that are not DOIs
    names: list[str] = []
    for chunk in _BRACKET_RE.findall(answer or ""):
        for part in re.split(r";|\|", chunk):
            part = part.strip()
            if part and not DOI_RE.search(part):
                names.append(part)
    name_matched, name_unmatched = [], []
    for nc in names:
        toks = _title_tokens(nc)
        if not toks:
            continue
        hit = any(len(toks & tt) >= 1 and len(toks & tt) / len(toks) >= 0.5
                  for tt in ev_titles)
        (name_matched if hit else name_unmatched).append(nc)

    n_checkable = len(cited_dois) + len(name_matched) + len(name_unmatched)
    n_matched = len(doi_matched) + len(name_matched)
    return {
        "doi_cited": cited_dois, "doi_matched": doi_matched, "doi_unmatched": doi_unmatched,
        "named": names, "name_matched": name_matched, "name_unmatched": name_unmatched,
        "checkable": n_checkable, "matched": n_matched,
        "match_rate": (n_matched / n_checkable) if n_checkable else None,
    }


def audit_entry(entry: dict, *, resolve: bool = False) -> dict:
    """Build the audit record for one ledger entry."""
    project, run_id = entry.get("project"), entry.get("run_id")
    run_dir = (config.PROJECTS / project / "runs" / run_id) if (project and run_id) else None
    have_run = bool(run_dir and run_dir.exists())

    evidence = entry.get("evidence") or []
    answer = entry.get("answer") or ""
    prov = _provenance(answer, evidence)
    ev_dois = {_norm_doi(e["doi"]) for e in evidence if e.get("doi")}

    stages: dict = {}
    if have_run:
        disc = _read_json(run_dir / "discovery.json")
        tri = _read_json(run_dir / "triage.json")
        acq = _read_json(run_dir / "acquire.json")
        summ = _read_json(run_dir / "summary.json") or {}
        stages["discovery"] = {"candidates": len(disc) if isinstance(disc, list)
                               else summ.get("n_candidates")}
        if tri:
            stages["triage"] = {"to_acquire": len(tri.get("to_acquire", [])),
                                "in_library": len(tri.get("in_library", [])),
                                "below_floor": tri.get("below_floor")}
        if acq:
            stages["acquire"] = {k: acq.get(k) for k in
                                 ("pulled", "already_have", "from_library",
                                  "no_oa", "not_found", "wrong_paper_only")}
        acqlog = run_dir / "acquire.jsonl"
        if acqlog.exists():
            by_source: dict = {}
            for line in acqlog.read_text().splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                s = r.get("source") or r.get("status")
                if s:
                    by_source[s] = by_source.get(s, 0) + 1
            stages.setdefault("acquire", {})["by_source"] = by_source

    resolved = _resolve(prov["doi_cited"]) if (resolve and prov["doi_cited"]) else {}

    return {
        "entry_id": entry.get("id"),
        "question": entry.get("question"),
        "ts": entry.get("ts"),
        "confidence": entry.get("confidence"),
        "project": project, "run_id": run_id, "reproducible": have_run,
        "evidence_count": len(evidence),
        "provenance": {**prov, "resolved": resolved},
        "evidence": [{"slug": e.get("slug"), "doi": e.get("doi"),
                      "title": e.get("title"), "tier": e.get("tier"),
                      "cited": _norm_doi(e.get("doi", "")) in prov["doi_matched"]}
                     for e in evidence],
        "stages": stages,
        "gaps": entry.get("gaps") or [],
        "verification_ceiling": _ceiling(answer, evidence, have_run),
    }


def render(a: dict) -> str:
    p = a["provenance"]
    vc = a["verification_ceiling"]
    L = [f"# Audit — ledger #{a['entry_id']}",
         f"{a['question']}", "",
         f"confidence: {a.get('confidence') or '—'}   |   "
         f"verification ceiling: rung {vc['rung']}/{vc['of']} ({vc['label']})   |   "
         f"reproducible: {'yes' if a['reproducible'] else 'no (ledger-only)'}", ""]

    if a["stages"]:
        L.append("## Provenance chain")
        s = a["stages"]
        if "discovery" in s:
            L.append(f"- discovery: {s['discovery'].get('candidates')} candidates considered")
        if "triage" in s:
            t = s["triage"]
            L.append(f"- triage: {t.get('to_acquire')} to acquire, "
                     f"{t.get('in_library')} already in library, "
                     f"{t.get('below_floor')} below relevance floor")
        if "acquire" in s:
            ac = s["acquire"]
            bs = ac.get("by_source")
            L.append("- acquire: " + ", ".join(f"{k}={v}" for k, v in ac.items() if k != "by_source"))
            if bs:
                L.append(f"  by source: {bs}")
        L.append("")

    L.append("## Citation provenance check")
    L.append("_(verifies each cited source was retrieved this session — NOT that it "
             "supports the specific claim; that is the manual claim-support step.)_")
    if p["checkable"] == 0:
        if p["named"] or p["doi_cited"]:
            L.append("- citations present but not checkable against retrieval (no usable tokens)")
        else:
            L.append("- the answer contains no parseable citations (DOI or named source)")
    else:
        L.append(f"- citations checked: {p['checkable']} "
                 f"({len(p['doi_cited'])} DOI, {len(p['name_matched']) + len(p['name_unmatched'])} named)")
        L.append(f"- present in this session's retrieval: {p['matched']}")
        unmatched = p["doi_unmatched"] + p["name_unmatched"]
        if unmatched:
            L.append(f"- ⚠ NOT in this session's retrieval (verify provenance — "
                     f"introduced from memory?): {len(unmatched)}")
            for d in p["doi_unmatched"]:
                rr = p["resolved"].get(d)
                tag = "" if rr is None else ("  [resolves]" if rr else "  [does NOT resolve — likely fabricated]")
                L.append(f"    - {d}{tag}")
            for nc in p["name_unmatched"]:
                L.append(f"    - \"{nc}\" (named source not among retrieved titles)")
        if p["match_rate"] is not None:
            L.append(f"- provenance match rate: {p['matched']}/{p['checkable']} "
                     f"({round(100 * p['match_rate'])}%)")
    L.append("")

    L.append(f"## Evidence retrieved ({a['evidence_count']} sources the answer could draw on)")
    for e in a["evidence"][:40]:
        mark = "✓" if e["cited"] else "·"
        doi = f"  ({e['doi']})" if e.get("doi") else ""
        L.append(f"- [{mark}] [{e.get('tier') or '—'}] {(e.get('title') or e.get('slug') or '')[:70]}{doi}")
    L.append("")

    if a["gaps"]:
        L.append("## Gaps logged")
        for g in a["gaps"][:15]:
            L.append(f"- {g.get('doi','') if isinstance(g, dict) else g} "
                     f"{g.get('reason','') if isinstance(g, dict) else ''}".strip())
        L.append("")

    if a["reproducible"]:
        L.append("## Reproduce")
        L.append(f"- run: `{a['project']}/{a['run_id']}` — re-run: "
                 f"`localevidence ask --project {a['project']} --resume`")
    return "\n".join(L)


def audit_cli(*, entry_id: Optional[int] = None, project: Optional[str] = None,
              as_json: bool = False, resolve: bool = False) -> int:
    from .ledger import Ledger
    led = Ledger()
    if entry_id is not None:
        entry = led.get(entry_id)
    elif project:
        entry = next((e for e in reversed(led.entries) if e.get("project") == project
                      and e.get("answer")), None)
    else:
        entry = next((e for e in reversed(led.entries) if e.get("answer")), None)
    if not entry:
        print("no matching worked answer in the ledger")
        return 2
    rec = audit_entry(entry, resolve=resolve)
    print(json.dumps(rec, indent=2) if as_json else render(rec))
    return 0
