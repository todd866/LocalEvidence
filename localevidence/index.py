"""Persistent passage index + hybrid retrieval — the structure that compounds.

One growing store (`data/passages/passages.db` FTS5 + `passages.npy` dense),
keyed by paper slug and extended incrementally: a paper is chunked and embedded
exactly once, ever. `search()` retrieves over the WHOLE accumulated corpus, so a
new question benefits from every paper any prior question pulled — and a
well-covered topic is answered with zero new chunking.

Retrieval fuses two complementary signals by Reciprocal Rank Fusion:
  - dense : MiniLM cosine (semantic; paraphrase, synonyms)
  - sparse: SQLite FTS5 BM25 (lexical; exact drug names, thresholds)
plus reference/table chunk hygiene and a guideline/SR tier-guarantee scoped to
the current question's own papers.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import config, embedding
from .acquire import AcquiredPaper
from .library import chunk_text


_PAGE_MARKER = re.compile(r"^-{2,}\s*Page\s+\d+\s*-{2,}\s*$", re.M)

# Drug-dosing signal: a number adjacent to a dose unit (mg/kg, mg, microgram,
# units, mL, IU, g). Dosing tables are numeric-dense by nature — the density
# filter below would otherwise discard them — but they are the single most
# clinically valuable content (the dose a clinician actually needs), so a chunk
# carrying doses is never treated as low-value.
_DOSING_RE = re.compile(
    r"\d\s*(?:mg|mcg|microgram|micrograms|units?|iu|ml|g)\b|\bmg/kg\b|\bmicrograms?/kg\b",
    re.I)


def _clean(text: str) -> str:
    return _PAGE_MARKER.sub(" ", text or "")


def _is_low_value_chunk(text: str) -> bool:
    """Drop reference lists, citation dumps, and dense numeric tables —
    grounding a clinical claim in a bibliography or a stats grid is worse than
    useless. Conservative, so real prose survives — and chunks carrying drug
    doses are always kept (they look numeric-dense but are the point)."""
    t = text or ""
    low = t.lower()
    ref_signals = (low.count("doi.org") + low.count("https://") +
                   low.count("et al") + low.count("accessed"))
    if ref_signals >= 4:
        return True
    # Protect clinical dosing tables from the numeric-density filter below.
    if _DOSING_RE.search(t):
        return False
    tokens = t.split()
    if len(tokens) >= 20:
        numeric = sum(1 for w in tokens if re.fullmatch(r"[-+(]?\d[\d.,;:%)/-]*", w))
        if numeric / len(tokens) > 0.32:
            return True
        alpha_words = sum(1 for w in tokens if re.search(r"[A-Za-z]{3,}", w))
        if alpha_words / len(tokens) < 0.45:
            return True
    return False


@dataclass
class Passage:
    passage_id: int
    slug: str
    text: str
    title: str = ""
    doi: str = ""
    pmid: str = ""
    year: str = ""
    journal: str = ""
    tier: str = ""
    chunk_idx: int = 0
    score: float = 0.0
    cosine: float = 0.0          # dense query·passage cosine (discriminative coverage signal)
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    guaranteed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class PassageIndex:
    """Persistent, incremental, slug-keyed hybrid passage store."""

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = Path(store_dir or config.PASSAGES_DIR)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.store_dir / "passages.db"
        self.vec_path = self.store_dir / "passages.npy"
        self._con = sqlite3.connect(self.db_path)
        self._con.execute("PRAGMA busy_timeout=30000")
        self._init_db()
        self.model_stamp = self.store_dir / "embedding_model.txt"
        self.meta: list[dict] = self._load_meta()           # aligned by passage_id (== rowid)
        self.indexed_slugs: set[str] = {m["slug"] for m in self.meta}
        self.vectors = (np.load(self.vec_path) if self.vec_path.exists()
                        else np.zeros((0, 384), dtype="float32"))
        # Rebuild the dense matrix if it falls out of alignment with the catalog,
        # OR if the embedding model changed under it (a different model means a
        # different vector space / dimension — mixing them silently corrupts
        # retrieval). The stamp records which model produced passages.npy.
        prev_model = (self.model_stamp.read_text().strip()
                      if self.model_stamp.exists() else config.EMBED_MODEL)
        if self.vectors.shape[0] != len(self.meta) or prev_model != config.EMBED_MODEL:
            self._rebuild_vectors()
        self.model_stamp.write_text(config.EMBED_MODEL)

    # -- store management ----------------------------------------------------

    def _init_db(self) -> None:
        self._con.execute("""CREATE TABLE IF NOT EXISTS passages (
            passage_id INTEGER PRIMARY KEY, slug TEXT, title TEXT, doi TEXT,
            pmid TEXT, year TEXT, journal TEXT, tier TEXT, chunk_idx INTEGER,
            text TEXT)""")
        self._con.execute("CREATE INDEX IF NOT EXISTS idx_pslug ON passages(slug)")
        self._con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, content='')")
        self._con.commit()

    def _load_meta(self) -> list[dict]:
        cur = self._con.execute(
            "SELECT passage_id,slug,title,doi,pmid,year,journal,tier,chunk_idx "
            "FROM passages ORDER BY passage_id")
        cols = ("passage_id", "slug", "title", "doi", "pmid", "year", "journal", "tier", "chunk_idx")
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _rebuild_vectors(self) -> None:
        """Safety net: re-embed all passage texts from the DB in id order if the
        dense matrix and the catalogue ever fall out of alignment."""
        rows = self._con.execute("SELECT text FROM passages ORDER BY passage_id").fetchall()
        texts = [r[0] for r in rows]
        self.vectors = (embedding.embed(texts) if texts
                        else np.zeros((0, 384), dtype="float32"))
        np.save(self.vec_path, self.vectors)

    def reload(self) -> None:
        """Re-read the on-disk store in place after another writer grew it (e.g. an
        acquire-on-miss pull). Keeps this object's identity so warm callers — the
        server's `_INDEX` — see the new passages without re-instantiating."""
        self.meta = self._load_meta()
        self.indexed_slugs = {m["slug"] for m in self.meta}
        self.vectors = (np.load(self.vec_path) if self.vec_path.exists()
                        else np.zeros((0, 384), dtype="float32"))
        if self.vectors.shape[0] != len(self.meta):
            self._rebuild_vectors()

    def stats(self) -> dict:
        return {"papers": len(self.indexed_slugs), "passages": len(self.meta)}

    # -- incremental build ---------------------------------------------------

    def add_papers(self, papers: Sequence[AcquiredPaper], *,
                   target_words: int = 220, overlap_words: int = 50,
                   verbose: bool = True) -> int:
        """Chunk + embed + append any papers not already indexed (by slug)."""
        new_rows: list[dict] = []
        new_texts: list[str] = []
        for p in papers:
            if not p.slug or p.slug in self.indexed_slugs:
                continue
            tp = Path(p.text_path)
            if not tp.exists():
                continue
            try:
                body = _clean(tp.read_text(errors="ignore"))
            except OSError:
                continue
            chunks = [c for c in chunk_text(body, target_words=target_words,
                                            overlap_words=overlap_words)
                      if not _is_low_value_chunk(c)]
            if not chunks:
                continue
            self.indexed_slugs.add(p.slug)
            for ci, ch in enumerate(chunks):
                new_rows.append({"slug": p.slug, "title": p.title, "doi": p.doi,
                                 "pmid": p.pmid, "year": str(p.year), "journal": p.journal,
                                 "tier": p.tier, "chunk_idx": ci})
                new_texts.append(ch)

        if not new_texts:
            return 0

        if verbose:
            print(f"  index: +{len(new_texts)} passages from "
                  f"{len({r['slug'] for r in new_rows})} new papers "
                  f"(store now {len(self.meta) + len(new_texts)})")
        vecs = embedding.embed(new_texts, show_progress=verbose and len(new_texts) > 400)

        next_id = len(self.meta)
        con = self._con
        for off, (row, txt) in enumerate(zip(new_rows, new_texts)):
            pid = next_id + off
            con.execute(
                "INSERT INTO passages "
                "(passage_id,slug,title,doi,pmid,year,journal,tier,chunk_idx,text) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pid, row["slug"], row["title"], row["doi"], row["pmid"],
                 row["year"], row["journal"], row["tier"], row["chunk_idx"], txt))
            con.execute("INSERT INTO fts(rowid, text) VALUES (?, ?)", (pid, txt))
            self.meta.append({k: row[k] for k in
                              ("slug", "title", "doi", "pmid", "year", "journal", "tier", "chunk_idx")}
                             | {"passage_id": pid})
        con.commit()
        self.vectors = (np.vstack([self.vectors, vecs]) if self.vectors.shape[0] else vecs)
        np.save(self.vec_path, self.vectors)
        return len(new_texts)

    # -- retrieve ------------------------------------------------------------

    def _sparse_ranked(self, query: str, k: int) -> list[int]:
        # FTS5 treats - : ( ) " * ^ as operators, so a hyphenated key term like
        # "beta-blocker" / "first-line" / "ST-elevation" raises
        # "no such column: <suffix>" and would silently kill the whole lexical
        # arm. Reduce every term to bare alphanumeric tokens and OR them.
        toks: list[str] = []
        for t in config.key_terms(query, limit=16):
            for tok in re.findall(r"[A-Za-z0-9]+", t):
                if len(tok) >= 2 and tok not in toks:
                    toks.append(tok)
        if not toks:
            return []
        try:
            cur = self._con.execute(
                "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
                (" OR ".join(toks), k))
            return [r[0] for r in cur.fetchall()]
        except sqlite3.OperationalError:
            return []

    def _passage(self, pid: int, sc: float, dense_rank, sparse_rank,
                 guaranteed: bool = False, cosine: float = 0.0) -> Optional[Passage]:
        row = self._con.execute(
            "SELECT slug,title,doi,pmid,year,journal,tier,chunk_idx,text "
            "FROM passages WHERE passage_id=?", (pid,)).fetchone()
        if not row:
            return None
        slug, title, doi, pmid, year, journal, tier, chunk_idx, text = row
        return Passage(passage_id=pid, slug=slug, text=text, title=title, doi=doi,
                       pmid=pmid, year=year, journal=journal, tier=tier,
                       chunk_idx=chunk_idx, score=sc, cosine=cosine, dense_rank=dense_rank,
                       sparse_rank=sparse_rank, guaranteed=guaranteed)

    def search(self, query: str, *, k: int = 12, k_dense: int = 50,
               k_sparse: int = 50, rrf_k: int = 60,
               focus_slugs: Optional[set[str]] = None,
               guideline_boost: bool = True, guideline_floor: float = 0.42,
               max_guidelines: int = 3) -> list[Passage]:
        """Hybrid retrieve over the whole store. The tier-guarantee is scoped to
        `focus_slugs` (this question's own papers) so a growing multi-topic store
        does not inject unrelated guidelines."""
        if self.vectors.shape[0] == 0:
            return []
        qv = embedding.embed([query])[0]
        sims = self.vectors @ qv
        kd = min(k_dense, sims.shape[0])
        didx = np.argpartition(-sims, kd - 1)[:kd]
        dense = didx[np.argsort(-sims[didx])].tolist()
        sparse = self._sparse_ranked(query, k_sparse)

        dense_rank = {pid: i for i, pid in enumerate(dense)}
        sparse_rank = {pid: i for i, pid in enumerate(sparse)}
        fused: dict[int, float] = {}
        for pid, i in dense_rank.items():
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (rrf_k + i)
        for pid, i in sparse_rank.items():
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (rrf_k + i)
        if not fused:
            return []

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = [p for p in (self._passage(pid, sc, dense_rank.get(pid), sparse_rank.get(pid),
                                         cosine=float(sims[pid]))
                           for pid, sc in top) if p]

        # Tier guarantee: each of THIS question's OWN guideline/SR papers
        # contributes its single best passage even if organic fusion buried it.
        # Only fires when the caller scoped the question's papers via focus_slugs.
        # A bare search (no focus — e.g. the server / PWA) must NOT fall back to
        # the whole corpus here: that injected the first-indexed guidelines
        # regardless of relevance (e.g. unrelated guidelines bleeding into every
        # query). Bare search relies on the relevance-gated guideline surfacing
        # below instead.
        if focus_slugs:
            covered = {p.slug for p in out}
            high_tier = []
            for m in self.meta:
                if (m.get("tier") in ("guideline", "systematic-review")
                        and m["slug"] in focus_slugs and m["slug"] not in covered
                        and m["slug"] not in high_tier):
                    high_tier.append(m["slug"])
            for slug in high_tier[:4]:
                pids = [i for i, m in enumerate(self.meta) if m["slug"] == slug]
                if pids:
                    best = max(pids, key=lambda i: sims[i])
                    p = self._passage(best, float(sims[best]), dense_rank.get(best),
                                      sparse_rank.get(best), guaranteed=True,
                                      cosine=float(sims[best]))
                    if p:
                        out.append(p)

        # Local-guideline surfacing. Authoritative guidelines (RCH etc.) are
        # written in terse bedside style and lose the dense race to academic
        # journal prose, so they rarely reach the organic top-k even when they
        # are the right source for a clinician. Guarantee the most query-relevant
        # guideline(s) a seat — but gated on a relevance floor so an off-topic
        # guideline is never injected into an unrelated question.
        if guideline_boost:
            covered = {p.slug for p in out}
            best_by_guideline: dict[str, tuple[float, int]] = {}
            for i, m in enumerate(self.meta):
                if m.get("tier") == "guideline" and m["slug"] not in covered:
                    s = float(sims[i])
                    if s > best_by_guideline.get(m["slug"], (-1.0, -1))[0]:
                        best_by_guideline[m["slug"]] = (s, i)
            for slug, (s, pid) in sorted(best_by_guideline.items(),
                                         key=lambda kv: kv[1][0], reverse=True)[:max_guidelines]:
                if s < guideline_floor:
                    break
                p = self._passage(pid, s, dense_rank.get(pid), sparse_rank.get(pid),
                                  guaranteed=True, cosine=float(s))
                if p:
                    out.append(p)
        return out
