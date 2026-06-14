"""Ingest an existing library into the passage index.

The `ask` loop indexes papers as it pulls them — but you may already hold a large
library (your own durable store, a shared one, or a harvested pack). Point
``LOCALEVIDENCE_LIBRARY`` at it and `index-library` chunks + embeds every
full-text paper in its catalog into the passage index, so retrieval covers the
**whole corpus you already have**, not just what this tool fetched itself.

This is how LocalEvidence sits on top of an existing paper store rather than
duplicating it: the library is the durable corpus (catalog + PDFs + text); the
passage index is the retrieval layer over it. Incremental — papers already
indexed (by slug) are skipped, so re-running only adds what's new.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from . import config


def index_library(*, match: Optional[str] = None, source: Optional[str] = None,
                  limit: int = 0, batch: int = 150, index=None,
                  verbose: bool = True) -> dict:
    """Index the configured library's full-text papers into the passage index.

    match : only papers whose title matches this regex (for a topic subset).
    source: only papers with this catalog `source`.
    limit : index at most N papers.
    batch : papers per persisted chunk — each batch is saved before the next, so a
            large run is resumable and never loses work to an interruption.
    """
    from .library import connect
    from .acquire import AcquiredPaper
    from .index import PassageIndex

    rx = re.compile(match, re.I) if match else None
    con = connect()
    q = "SELECT slug,doi,pmid,title,year,journal,text_path,source FROM papers WHERE text_path!=''"
    params: tuple = ()
    if source:
        q += " AND source=?"
        params = (source,)
    rows = con.execute(q, params).fetchall()
    con.close()

    papers = []
    for slug, doi, pmid, title, year, journal, tp, src in rows:
        if rx and not (title and rx.search(title)):
            continue
        if not (tp and Path(tp).exists()):
            continue
        papers.append(AcquiredPaper(
            slug=slug, doi=doi or "", pmid=str(pmid or ""), title=title or "",
            year=str(year or ""), journal=journal or "",
            tier=config.classify_tier(title or "", ""), text_path=tp,
            source=src or "library"))
        if limit and len(papers) >= limit:
            break

    if verbose:
        bits = []
        if match:
            bits.append(f"matching /{match}/")
        if source:
            bits.append(f"source='{source}'")
        print(f"  index-library: {len(papers)} full-text papers "
              f"{' '.join(bits)} to consider".replace("  ", " "))

    idx = index if index is not None else PassageIndex()
    before = idx.stats()
    step = batch if batch and batch > 0 else max(len(papers), 1)
    n_batches = (len(papers) + step - 1) // step if papers else 0
    added = 0
    for bi, i in enumerate(range(0, len(papers), step), 1):
        added += idx.add_papers(papers[i:i + step], verbose=False)  # persists each batch
        if verbose and n_batches > 1:
            print(f"  index-library: batch {bi}/{n_batches} -> +{added} cumulative "
                  f"({idx.stats()['papers']} papers, {idx.stats()['passages']} passages)")
    after = idx.stats()
    if verbose:
        print(f"  index-library: +{added} passages "
              f"({before['papers']} -> {after['papers']} papers indexed)")
    return {"considered": len(papers), "passages_added": added,
            "store_before": before, "store_after": after}
