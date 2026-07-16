"""Acquisition: turn triage survivors into full text in the local library.

Thin wrapper over `library.pull`, the provider cascade — dedup (already_have)
-> the legal open-access providers (Unpaywall, Europe PMC, local-file drop) ->
an optional, you-supply-it shadow tier. Every candidate PDF is verified against
its title before it is kept, then catalogued and text-extracted. We don't
reimplement any of that here; we feed it DOIs and record outcomes.

The compounding lives here: anything pulled stays in the local library, so the
next similar question finds it as `already_have`.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from . import config
from .discovery import Candidate
from .triage import TriageResult

# Hang-protection bounds (env-tunable). A slow open-access mirror must not be
# able to block a whole acquisition run indefinitely.
DEFAULT_PULL_BUDGET_S = 120.0   # per-pull timeout;   <=0 disables
DEFAULT_ACQUIRE_BUDGET_S = 0.0  # per-question budget; <=0 disables (unlimited)


class PullTimeout(Exception):
    """A single pull exceeded its per-pull time budget (LE_PULL_BUDGET_S)."""


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _pull_bounded(fn: Callable[[], object], *, timeout_s: float):
    """Run `fn()` with a wall-clock bound. Returns its result, re-raises its
    exception, or raises PullTimeout if it outlives `timeout_s`.

    `timeout_s <= 0` runs `fn` inline (no bound). The bound uses a daemon worker
    thread: on timeout the underlying network call is abandoned (it cannot be
    force-killed from here) and the run moves on; the orphaned thread dies with
    the process. This is the CLI hang-guard, not a hard cancellation — a pull
    that completes just after the deadline may still finish its own commit into
    the local library out of band (harmless: it is idempotent and slug-keyed, so
    the paper is simply found as `already_have` next run). For a truly killable
    pull, run each acquisition in its own process; that is deliberately out of
    scope for this self-contained public build.
    """
    if timeout_s is None or timeout_s <= 0:
        return fn()

    box: dict[str, object] = {}

    def _worker():
        try:
            box["result"] = fn()
        except BaseException as exc:  # propagate the real pull error to the caller
            box["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise PullTimeout(f"pull exceeded {timeout_s:g}s")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("result")


@dataclass
class AcquiredPaper:
    slug: str
    doi: str = ""
    pmid: str = ""
    title: str = ""
    authors: str = ""
    year: str = ""
    journal: str = ""
    text_path: str = ""
    tier: str = ""
    relevance: float = 0.0
    status: str = ""        # pulled / already_have / library
    source: str = ""        # which acquisition tier won (unpaywall/europepmc/localfile/...)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AcquireReport:
    papers: list[AcquiredPaper] = field(default_factory=list)   # have text, ready to index
    pulled: int = 0
    already_have: int = 0
    from_library: int = 0
    no_oa: int = 0
    not_found: int = 0
    wrong_paper_only: int = 0
    no_text: int = 0
    budget_skipped: int = 0     # candidates skipped because the acquisition budget was spent
    failures: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d["indexable"] = len(self.papers)
        d.pop("papers")
        d.pop("failures")
        return d


def _record_from_lib(rec: dict, cand: Optional[Candidate], status: str) -> Optional[AcquiredPaper]:
    """Build an AcquiredPaper from a library catalogue row."""
    text_path = rec.get("text_path") or ""
    if not text_path or not Path(text_path).exists():
        return None
    return AcquiredPaper(
        slug=rec.get("slug", ""),
        doi=rec.get("doi", ""),
        pmid=str(rec.get("pmid", "") or ""),
        title=rec.get("title") or (cand.title if cand else ""),
        authors=rec.get("authors") or (", ".join(cand.authors) if cand else ""),
        year=str(rec.get("year", "") or (cand.year if cand else "") or ""),
        journal=rec.get("journal") or (cand.journal if cand else ""),
        text_path=text_path,
        tier=cand.tier if cand else "",
        relevance=cand.relevance if cand else 0.0,
        status=status,
        source=rec.get("source", ""),
    )


def acquire(
    triage_result: TriageResult,
    *,
    oa_only: bool = False,
    pace_s: float = 1.0,
    log_path: Optional[Path] = None,
    verbose: bool = True,
) -> AcquireReport:
    """Pull every `to_acquire` candidate; fold in the `in_library` ones.

    Returns an AcquireReport whose `papers` are the text-bearing records to
    index. Idempotent: re-running skips papers already held.
    """
    from .library import pull, find  # self-contained local library

    report = AcquireReport()
    log_fh = open(log_path, "a") if log_path else None

    def _log(entry: dict) -> None:
        if log_fh:
            import json
            log_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log_fh.flush()

    # 1. Papers already in the library (free) — index them, no fetch.
    for c in triage_result.in_library:
        rec = find(doi=c.doi) or (find(pmid=c.pmid) if c.pmid else None)
        if rec:
            ap = _record_from_lib(rec, c, status="library")
            if ap:
                report.papers.append(ap)
                report.from_library += 1
            else:
                report.no_text += 1

    # 2. Fetch the triage survivors not already held.
    pull_budget_s = _env_seconds("LE_PULL_BUDGET_S", DEFAULT_PULL_BUDGET_S)
    acquire_budget_s = _env_seconds("LE_ACQUIRE_BUDGET_S", DEFAULT_ACQUIRE_BUDGET_S)
    started = time.monotonic()
    deadline = started + acquire_budget_s if acquire_budget_s > 0 else None
    total = len(triage_result.to_acquire)
    for i, c in enumerate(triage_result.to_acquire, 1):
        remaining = (deadline - time.monotonic()) if deadline is not None else None
        if remaining is not None and remaining <= 0:
            report.budget_skipped = total - (i - 1)
            _log({"status": "budget_exhausted", "skipped": report.budget_skipped,
                  "budget_s": acquire_budget_s})
            if verbose:
                print(f"  acquire: budget {acquire_budget_s:g}s spent; "
                      f"skipping {report.budget_skipped} remaining")
            break
        # Bound each pull by the smaller of the per-pull budget and the time left
        # in the acquisition budget, so no single pull can overrun the deadline.
        timeout_s = pull_budget_s
        if remaining is not None:
            timeout_s = remaining if timeout_s <= 0 else min(timeout_s, remaining)
        if verbose:
            print(f"  acquire [{i}/{total}] {c.tier:<17} {c.doi}  {c.title[:50]}")
        try:
            res = _pull_bounded(
                lambda c=c: pull(
                    c.doi,
                    title=c.title,
                    pmid=c.pmid,
                    authors="; ".join(c.authors[:8]),
                    year=str(c.year or ""),
                    journal=c.journal,
                    oa_only=oa_only,
                ),
                timeout_s=timeout_s,
            )
        except Exception as e:
            report.failures.append({"doi": c.doi, "error": f"{type(e).__name__}: {e}"})
            _log({"doi": c.doi, "status": "error", "error": str(e)})
            continue

        status = res.get("_status", "")
        _log({"doi": c.doi, "title": c.title, "status": status,
              "source": res.get("source", ""), "best_ratio": res.get("best_ratio")})

        if status in ("pulled", "already_have"):
            ap = _record_from_lib(res, c, status=status)
            if ap:
                report.papers.append(ap)
                if status == "pulled":
                    report.pulled += 1
                else:
                    report.already_have += 1
            else:
                report.no_text += 1
        elif status == "no_oa":
            report.no_oa += 1
        elif status == "wrong_paper_only":
            report.wrong_paper_only += 1
        else:
            report.not_found += 1

        if pace_s and i < total and (deadline is None or time.monotonic() + pace_s < deadline):
            time.sleep(pace_s)

    if log_fh:
        log_fh.close()
    if verbose:
        print(f"  acquire: {report.pulled} pulled, {report.already_have} already had, "
              f"{report.from_library} from library, "
              f"{report.no_oa} no-OA, {report.wrong_paper_only} wrong-paper, "
              f"{report.not_found} not-found -> {len(report.papers)} indexable")
    return report
