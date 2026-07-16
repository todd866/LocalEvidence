"""Orchestrate the ask loop: discover -> triage -> acquire -> index -> pack.

Each stage writes a JSON checkpoint into the run directory, so a run is
inspectable and `--resume` can skip recompute. Acquisition is idempotent at the
library level (already-held papers resolve instantly), so even a full re-run is
cheap after the first night — which is the whole compounding idea.

Entry point: `ask(question, ...)` -> dict of result paths + summary.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config, discovery, triage as triage_mod, acquire as acquire_mod
from . import index as index_mod, evidence as evidence_mod
from .discovery import Candidate
from .triage import TriageResult
from .acquire import AcquireReport, AcquiredPaper
from .ledger import Ledger


def _slug(text: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen].rstrip("-") or "question"


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def _read_json(path: Path):
    return json.loads(path.read_text())


def _snippet(text: str, maxlen: int = 600) -> str:
    """Whitespace-normalised passage excerpt (matches evidence.py's pack style)."""
    s = " ".join((text or "").split())
    if len(s) > maxlen:
        s = s[:maxlen].rsplit(" ", 1)[0] + " …"
    return s


def evidence_from_passages(passages) -> list[dict]:
    """Collapse retrieved passages into one evidence record per source paper.

    Each record is `{slug, doi, title, tier, passage_ids, snippet, score}`.
    `snippet` (text of the highest-scoring passage for that slug) and `score`
    (the max passage score) are what the EvidenceViewer localevidence loader
    reads to render a source-linked passage; without them the viewer shows
    empty passages. Passage order is not assumed — the best passage wins.
    """
    ev_by_slug: dict[str, dict] = {}
    best_score: dict[str, float] = {}
    for p in passages:
        e = ev_by_slug.setdefault(p.slug, {"slug": p.slug, "doi": p.doi,
                                           "title": p.title, "tier": p.tier,
                                           "passage_ids": [], "snippet": "",
                                           "score": 0.0})
        e["passage_ids"].append(p.passage_id)
        if p.slug not in best_score or p.score > best_score[p.slug]:
            best_score[p.slug] = p.score
            e["score"] = p.score
            e["snippet"] = _snippet(p.text)
    return list(ev_by_slug.values())


def _ensure_project(question: str, project: Optional[str]) -> Path:
    slug = project or _slug(question)
    pdir = config.PROJECTS / slug
    pdir.mkdir(parents=True, exist_ok=True)
    q_md = pdir / "question.md"
    if not q_md.exists():
        q_md.write_text(f"# Question\n\n{question.strip()}\n")
    for sub in ("evidence-log.md", "gap-log.md"):
        f = pdir / sub
        if not f.exists():
            f.write_text(f"# {sub.replace('-', ' ').replace('.md', '').title()}\n")
    (pdir / "private").mkdir(exist_ok=True)
    return pdir


@dataclass
class AskResult:
    project_dir: Path
    run_dir: Path
    evidence_pack_path: Path
    summary: dict
    passages: list


def record_answer(
    *,
    project: Optional[str] = None,
    run_dir: Optional[str] = None,
    answer: Optional[str] = None,
    reasoning: Optional[str] = None,
    confidence: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Persist a synthesised answer into its ledger entry — the writeback that
    turns the ledger from a paper-cache into accumulated worked answers.

    The synthesiser is Claude-in-the-loop (this session), not an API: write the
    grounded answer to `<run_dir>/answer.md`, then call this to fold it into the
    ledger so a future similar question can reuse it. Cited DOIs are extracted as
    the grounding record.
    """
    if run_dir:
        rd = Path(run_dir)
    elif project:
        runs = sorted((config.PROJECTS / project / "runs").glob("*"))
        if not runs:
            raise SystemExit(f"no runs for project {project!r}")
        rd = runs[-1]
    else:
        raise SystemExit("record_answer needs --project or --run")

    summary = _read_json(rd / "summary.json")
    entry_id = summary.get("ledger_entry")
    if entry_id is None:
        raise SystemExit(f"run {rd} has no ledger_entry in summary.json")

    if answer is None:
        af = rd / "answer.md"
        if not af.exists():
            raise SystemExit(f"no answer text: write it to {af} or pass --file")
        answer = af.read_text()

    from .audit import _answer_dois   # shared DOI extraction (handles embedded parens)
    cited = sorted(set(_answer_dois(answer)))
    led = Ledger()
    led.update(entry_id, answer=answer, reasoning=reasoning,
               grounding={"cited_sources": cited, "n_cited": len(cited)},
               confidence=confidence)
    (rd / "answer.md").write_text(answer)
    # human-readable copy at the project root too
    (rd.parent.parent / "answer.md").write_text(answer)
    if verbose:
        print(f"recorded answer → ledger #{entry_id} "
              f"({len(cited)} cited sources, confidence={confidence or 'n/a'})")
    return entry_id


def ask(
    question: str,
    *,
    project: Optional[str] = None,
    run_id: Optional[str] = None,
    extra_queries: Optional[list[str]] = None,
    top_n: int = 20,
    max_per_query: int = 100,
    k_passages: int = 12,
    oa_only: bool = False,
    relevance_floor: float = 0.30,
    resume: bool = False,
    verbose: bool = True,
) -> AskResult:
    project_dir = _ensure_project(question, project)
    run_id = run_id or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = project_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    disc_path = run_dir / "discovery.json"
    triage_path = run_dir / "triage.json"
    acquire_path = run_dir / "acquire.json"
    acquire_log = run_dir / "acquire.jsonl"
    passages_path = run_dir / "passages.json"
    pack_path = run_dir / "evidence-pack.md"

    if verbose:
        print(f"\n=== LocalEvidence ask ===\nQ: {question}\nproject: {project_dir.name}  run: {run_id}\n")

    # 0. Knowledge ledger — have we worked this out before? ------------------
    led = Ledger()
    prior = led.find_similar(question)
    if prior and verbose:
        print("  ledger: similar prior question(s) —")
        for e, s in prior:
            state = "answered" if e.get("answer") else "evidence-only"
            print(f"    [{s:.2f}] #{e['id']} ({state}) {e['question'][:66]}")

    # 1. Discovery -----------------------------------------------------------
    if resume and disc_path.exists():
        candidates = [Candidate.from_dict(d) for d in _read_json(disc_path)]
        if verbose:
            print(f"  discover: loaded {len(candidates)} candidates (resume)")
    else:
        candidates = discovery.discover(
            question, extra_queries=extra_queries,
            max_per_query=max_per_query, verbose=verbose)
        _write_json(disc_path, [c.to_dict() for c in candidates])

    # 2. Triage --------------------------------------------------------------
    tr = triage_mod.triage(
        question, candidates, top_n=top_n,
        relevance_floor=relevance_floor, verbose=verbose)
    _write_json(triage_path, {
        "to_acquire": [c.to_dict() for c in tr.to_acquire],
        "in_library": [c.to_dict() for c in tr.in_library],
        "below_floor": tr.below_floor,
        "ranked_top50": [c.to_dict() for c in tr.ranked[:50]],
    })

    # 3. Acquire -------------------------------------------------------------
    if resume and acquire_path.exists():
        saved = _read_json(acquire_path)
        report = AcquireReport(**{k: v for k, v in saved.items() if k not in ("papers", "_summary")})
        report.papers = [AcquiredPaper(**p) for p in saved.get("papers", [])]
        if verbose:
            print(f"  acquire: loaded {len(report.papers)} indexable papers (resume)")
    else:
        report = acquire_mod.acquire(
            tr, oa_only=oa_only, log_path=acquire_log, verbose=verbose)
        _write_json(acquire_path, {
            **{k: getattr(report, k) for k in
               ("pulled", "already_have", "from_library", "no_oa",
                "not_found", "wrong_paper_only", "no_text", "budget_skipped")},
            "failures": report.failures,
            "papers": [p.to_dict() for p in report.papers],
            "_summary": report.summary(),
        })

    # 4. Index (persistent, incremental) + 5. Retrieve -----------------------
    # The passage index below is the only semantic store that retrieval uses,
    # and `add_papers` embeds newly-pulled papers as it indexes them, so there
    # is nothing else to keep current — the store is self-maintaining.
    pidx = index_mod.PassageIndex()                      # default store: data/passages
    pidx.add_papers(report.papers, verbose=verbose)
    focus = {p.slug for p in report.papers}
    passages = pidx.search(question, k=k_passages, focus_slugs=focus)
    _write_json(passages_path, [p.to_dict() for p in passages])
    n_passages_total = sum(1 for m in pidx.meta if m["slug"] in focus)
    store_stats = pidx.stats()

    # 6. Evidence pack -------------------------------------------------------
    pack = evidence_mod.build_pack(
        question,
        triage_result=tr,
        acquire_report=report,
        passages=passages,
        n_candidates=len(candidates),
        n_passages_total=n_passages_total,
    )
    pack_path.write_text(pack.markdown)

    # append a pointer into the project evidence log + gap log
    with (project_dir / "evidence-log.md").open("a") as fh:
        fh.write(f"\n## Run {run_id}\n\n- {pack.n_papers} papers, "
                 f"{pack.coverage['newly_pulled']} newly pulled, "
                 f"{pack.coverage['passages_indexed']} passages.\n"
                 f"- Pack: `{pack_path.relative_to(project_dir)}`\n")
    if pack.gaps:
        with (project_dir / "gap-log.md").open("a") as fh:
            fh.write(f"\n## Run {run_id}\n\n")
            for g in pack.gaps[:30]:
                fh.write(f"- `{g.get('doi','')}` {g.get('reason','')}\n")

    # 7. Knowledge ledger — record the question + the evidence it used. The
    # answer + reasoning are filled later at synthesis (led.update).
    entry_id = led.record(question, project=project_dir.name, run_id=run_id,
                          evidence=evidence_from_passages(passages),
                          coverage=pack.coverage, gaps=pack.gaps)

    summary = {
        "question": question,
        "run_id": run_id,
        "project": project_dir.name,
        **pack.coverage,
        "retrieved_passages": len(passages),
        "passage_store": store_stats,
        "ledger_entry": entry_id,
        "prior_similar": [{"id": e["id"], "sim": round(s, 2)} for e, s in prior],
        "evidence_pack": str(pack_path),
    }
    _write_json(run_dir / "summary.json", summary)

    if verbose:
        print(f"\n=== done ===")
        print(json.dumps(summary, indent=2))
        print(f"\nEvidence pack: {pack_path}")

    return AskResult(project_dir, run_dir, pack_path, summary, passages)
