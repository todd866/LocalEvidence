"""Knowledge packs — the distributable layer of a compounding corpus.

You cannot legally redistribute a corpus of copyrighted PDFs. But three things
*about* the corpus are not the publisher's content and ARE shareable:

  1. the **paper list** — bibliographic metadata (DOI, title, authors, year,
     journal, evidence tier). Facts, not expression.
  2. **summaries** — *your own words* on what each paper provides (NOT the
     abstract — a paraphrase you generate). The exported slots start empty; you
     fill them in your own words (a Claude-in-the-loop step; see docs/PACK.md).
  3. the **map** — how the papers fit together: topic clusters + a
     nearest-neighbour similarity graph, derived from embeddings (structure, not
     text; no verbatim passages or raw vectors are shipped).

A recipient takes the pack and **harvests the papers themselves** (`pack
harvest`) under their own access — reconstructing the corpus from the map. The
knowledge compounds and travels; the copyrighted bytes stay home.

Shareable boundary (enforced here): a pack contains NO full text, NO PDF paths,
NO verbatim passages, NO raw embedding vectors — only bibliographic facts,
generated summaries, and derived cluster/neighbour structure.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from . import config

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_STOP = {"the", "and", "for", "with", "from", "study", "review", "analysis",
         "patients", "clinical", "using", "based", "effect", "effects", "among",
         "associated", "outcomes", "management", "treatment", "trial"}

# Bibliographic fields ONLY. Deliberately excludes `source`: how a paper was
# acquired (OA vs a shadow/ezproxy tier) is private provenance and must never ship
# in a public pack — it would leak the acquisition method the public repo says it
# doesn't include.
_BIBLIO = ("slug", "doi", "pmid", "title", "authors", "year", "journal", "tier")


def _paper_centroids(index) -> tuple[list[str], np.ndarray, dict]:
    """Mean passage vector per paper (slug), + a slug->biblio map from meta."""
    by_slug: dict[str, list[int]] = {}
    meta: dict[str, dict] = {}
    for i, m in enumerate(index.meta):
        s = m["slug"]
        by_slug.setdefault(s, []).append(i)
        meta.setdefault(s, m)
    slugs, rows = [], []
    for s, idxs in by_slug.items():
        if index.vectors.shape[0] == 0:
            continue
        v = index.vectors[idxs].mean(axis=0)
        n = np.linalg.norm(v) or 1.0
        slugs.append(s)
        rows.append(v / n)
    X = np.stack(rows) if rows else np.zeros((0, 384), dtype="float32")
    return slugs, X.astype("float32"), meta


def _kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Tiny cosine k-means (X rows are L2-normalised). Returns labels."""
    n = X.shape[0]
    if n == 0 or k <= 1:
        return np.zeros(n, dtype=int)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        sims = X @ cent.T                       # (n, k) cosine (normalised)
        new = sims.argmax(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = X[labels == c]
            if len(members):
                v = members.mean(axis=0)
                cent[c] = v / (np.linalg.norm(v) or 1.0)
    return labels


def _cluster_label(slugs: list[str], meta: dict, n: int = 4) -> str:
    """Top title terms across a cluster's papers."""
    freq: dict[str, int] = {}
    for s in slugs:
        for t in _TOKEN_RE.findall((meta.get(s, {}).get("title") or "").lower()):
            if len(t) > 3 and t not in _STOP:
                freq[t] = freq.get(t, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return " · ".join(t for t, _ in top) or "(unlabelled)"


def export_pack(out_dir: str | Path, *, index=None, match: Optional[str] = None,
                k_clusters: Optional[int] = None, neighbours: int = 5,
                verbose: bool = True) -> dict:
    """Write a shareable knowledge pack: papers.jsonl + map.json + summaries.jsonl.

    `match`: restrict the pack to papers whose title matches this regex (a topic
    subset, e.g. an MND pack from a mixed corpus).
    """
    from .index import PassageIndex
    from .library import connect
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. paper list (bibliographic only) — from the catalog, indexed papers --
    idx = index if index is not None else PassageIndex()
    slugs, X, meta = _paper_centroids(idx)
    if match:
        rx = re.compile(match, re.I)
        keep = [i for i, s in enumerate(slugs)
                if rx.search(meta.get(s, {}).get("title") or "")]
        slugs = [slugs[i] for i in keep]
        X = X[keep] if keep else X[:0]
    con = connect()
    rows = {r["slug"]: dict(r) for r in
            (dict(zip([d[0] for d in con.execute("SELECT * FROM papers LIMIT 0").description], row))
             for row in con.execute("SELECT * FROM papers").fetchall())}
    con.close()

    papers = []
    for s in slugs:
        r = rows.get(s, {})
        m = meta.get(s, {})
        papers.append({k: (r.get(k) if r.get(k) not in (None, "") else m.get(k, ""))
                       for k in _BIBLIO})
    with (out / "papers.jsonl").open("w") as fh:
        for p in papers:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    # --- 2. summaries (empty own-words slots — filled by hand; see docs/PACK.md) -
    with (out / "summaries.jsonl").open("w") as fh:
        for p in papers:
            fh.write(json.dumps({"slug": p["slug"], "title": p["title"],
                                 "tier": p["tier"], "provides": ""}) + "\n")

    # --- 3. map: clusters + nearest-neighbour graph (derived structure) --------
    k = k_clusters or max(1, int(round(len(slugs) ** 0.5)))
    labels = _kmeans(X, k) if len(slugs) else np.zeros(0, dtype=int)
    clusters = []
    for c in sorted(set(labels.tolist())):
        members = [slugs[i] for i in range(len(slugs)) if labels[i] == c]
        clusters.append({"id": int(c), "label": _cluster_label(members, meta),
                         "size": len(members), "members": members})
    links = []
    if len(slugs) > 1:
        sims = X @ X.T
        np.fill_diagonal(sims, -1.0)
        for i, s in enumerate(slugs):
            order = np.argsort(-sims[i])[:neighbours]
            links.append({"slug": s, "neighbours": [
                {"slug": slugs[j], "sim": round(float(sims[i][j]), 3)}
                for j in order if sims[i][j] > 0]})
    tiers: dict[str, int] = {}
    for p in papers:
        tiers[p["tier"] or "other"] = tiers.get(p["tier"] or "other", 0) + 1
    (out / "map.json").write_text(json.dumps(
        {"clusters": clusters, "links": links, "tiers": tiers,
         "n_papers": len(papers)}, indent=2))

    # --- pack README ----------------------------------------------------------
    (out / "README.md").write_text(_pack_readme(len(papers), len(clusters)))

    summary = {"papers": len(papers), "clusters": len(clusters),
               "with_neighbours": len(links), "out": str(out)}
    if verbose:
        print(f"  pack: {summary['papers']} papers, {summary['clusters']} clusters "
              f"-> {out}  (summaries.jsonl slots are empty — fill them in your own "
              f"words; see docs/PACK.md)")
    return summary


def _pack_readme(n_papers: int, n_clusters: int) -> str:
    return (
        f"# LocalEvidence knowledge pack\n\n"
        f"{n_papers} papers · {n_clusters} topic clusters · "
        f"built {dt.datetime.now().date()}\n\n"
        "A shareable map of a literature corpus — **not the corpus itself**.\n\n"
        "- `papers.jsonl` — the paper list (bibliographic metadata; DOIs to harvest).\n"
        "- `summaries.jsonl` — what each paper provides (generated summaries, own words).\n"
        "- `map.json` — topic clusters + a nearest-neighbour similarity graph.\n\n"
        "No full text, PDFs, verbatim passages, or raw vectors are included — those "
        "stay with whoever holds the corpus.\n\n"
        "## Rebuild the corpus from this pack\n\n"
        "```\nlocalevidence pack harvest <this-dir>\n```\n\n"
        "Harvests each paper by DOI under *your own* access and indexes it, "
        "reconstructing the corpus locally. What you can then ground in is yours.\n")


def harvest_pack(pack_dir: str | Path, *, index=None, oa_only: bool = False,
                 top: int = 0, verbose: bool = True) -> dict:
    """Reconstruct a corpus from a pack: acquire each listed paper, then index."""
    from .library import pull
    from .acquire import AcquiredPaper
    from .index import PassageIndex
    pack = Path(pack_dir)
    papers = [json.loads(l) for l in (pack / "papers.jsonl").read_text().splitlines() if l.strip()]
    if top:
        papers = papers[:top]
    acquired, pulled, had, failed = [], 0, 0, 0
    for i, p in enumerate(papers, 1):
        doi = p.get("doi")
        if not doi:
            failed += 1
            continue
        res = pull(doi, title=p.get("title", ""), pmid=str(p.get("pmid") or ""),
                   oa_only=oa_only)
        st = res.get("_status")
        if st in ("pulled", "already_have") and res.get("text_path"):
            acquired.append(AcquiredPaper(
                slug=res.get("slug", ""), doi=res.get("doi", ""),
                title=res.get("title") or p.get("title", ""), tier=p.get("tier", ""),
                text_path=res.get("text_path", ""), source=res.get("source", "")))
            pulled += st == "pulled"
            had += st == "already_have"
        else:
            failed += 1
        if verbose and i % 10 == 0:
            print(f"  harvest [{i}/{len(papers)}] {pulled} pulled, {had} had, {failed} missing")
    pidx = index if index is not None else PassageIndex()
    indexed = pidx.add_papers(acquired, verbose=verbose)
    summary = {"listed": len(papers), "pulled": pulled, "already_had": had,
               "missing": failed, "passages_indexed": indexed}
    if verbose:
        print(f"  harvest: {pulled} pulled, {had} already held, {failed} missing; "
              f"+{indexed} passages indexed")
    return summary
