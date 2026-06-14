"""The audit layer — emit a reconstructable provenance trail for a worked answer.

Auditability is LocalEvidence's differentiator and a safety property: every answer
should ship the chain that produced it, re-checkable by an independent party.
`audit_entry` assembles that trail from the ledger entry plus the run checkpoints,
and runs the safety-relevant check at its core:

    are the answer's citations actually grounded in what THIS session retrieved,
    or were they smuggled in from the model's parametric memory?

A cited DOI that is not in the session's retrieved evidence is flagged — that is
the exact failure mode (a confident citation the retrieval never supports). It also
reports a "verification ceiling": how far back the answer can be independently
reconstructed (see study AUDIT_DEPTH ladder). For LocalEvidence with a full run,
that is the top rung — the whole session re-runs from the trail.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from . import config

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\]\)\"'>]+")

_RUNGS = {
    0: "answer only", 1: "answer + citations", 2: "citations resolvable",
    3: "sources retrievable", 4: "claim->source map", 5: "retrieval set visible",
    6: "retrieval reproducible", 7: "corpus inspectable", 8: "reasoning trace",
    9: "end-to-end re-runnable",
}


def _norm_doi(d: str) -> str:
    return (d or "").rstrip(".,;:)").lower()


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _ceiling(answer: str, evidence: list, have_run: bool) -> dict:
    rung = 0
    if answer:
        rung = 1
    if DOI_RE.search(answer or ""):
        rung = 2
    if evidence:
        rung = max(rung, 5)   # retrieval set visible, sources retrievable, claim->source
    if have_run:
        rung = 9              # full chain re-runs from the checkpoints
    return {"rung": rung, "of": 9, "label": _RUNGS[rung]}


def _resolve(dois: list[str]) -> dict:
    """Optional live existence check via doi.org (network). Best-effort."""
    out = {}
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


def audit_entry(entry: dict, *, resolve: bool = False) -> dict:
    """Build the audit record for one ledger entry."""
    project, run_id = entry.get("project"), entry.get("run_id")
    run_dir = (config.PROJECTS / project / "runs" / run_id) if (project and run_id) else None
    have_run = bool(run_dir and run_dir.exists())

    evidence = entry.get("evidence") or []
    ev_dois = {_norm_doi(e["doi"]) for e in evidence if e.get("doi")}

    answer = entry.get("answer") or ""
    cited = sorted({_norm_doi(d) for d in DOI_RE.findall(answer)})
    grounded = [d for d in cited if d in ev_dois]
    ungrounded = [d for d in cited if d not in ev_dois]

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

    resolved = _resolve(cited) if (resolve and cited) else {}

    return {
        "entry_id": entry.get("id"),
        "question": entry.get("question"),
        "ts": entry.get("ts"),
        "confidence": entry.get("confidence"),
        "project": project, "run_id": run_id, "reproducible": have_run,
        "evidence_count": len(evidence),
        "citations": {
            "cited": cited, "grounded": grounded, "ungrounded": ungrounded,
            "grounding_rate": (len(grounded) / len(cited)) if cited else None,
            "resolved": resolved,
        },
        "evidence": [{"slug": e.get("slug"), "doi": e.get("doi"),
                      "title": e.get("title"), "tier": e.get("tier"),
                      "cited": _norm_doi(e.get("doi", "")) in cited} for e in evidence],
        "stages": stages,
        "gaps": entry.get("gaps") or [],
        "verification_ceiling": _ceiling(answer, evidence, have_run),
    }


def render(a: dict) -> str:
    c = a["citations"]
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
            L.append(f"- acquire: " + ", ".join(f"{k}={v}" for k, v in ac.items() if k != "by_source"))
            if bs:
                L.append(f"  by source: {bs}")
        L.append("")

    L.append("## Citation grounding check")
    if not c["cited"]:
        L.append("- no DOI-level citations in the answer")
    else:
        L.append(f"- cited DOIs: {len(c['cited'])}")
        L.append(f"- grounded in this session's retrieved evidence: {len(c['grounded'])}")
        if c["ungrounded"]:
            L.append(f"- ⚠ UNGROUNDED (cited but NOT retrieved this session — verify provenance): "
                     f"{len(c['ungrounded'])}")
            for d in c["ungrounded"]:
                rr = c["resolved"].get(d)
                tag = "" if rr is None else ("  [resolves]" if rr else "  [does NOT resolve — likely fabricated]")
                L.append(f"    - {d}{tag}")
        rate = c["grounding_rate"]
        L.append(f"- grounding rate: {len(c['grounded'])}/{len(c['cited'])}"
                 + (f" ({round(100 * rate)}%)" if rate is not None else ""))
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
