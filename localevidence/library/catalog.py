"""The local paper library — one SQLite catalog + a PDF store + extracted text.

Deduped by DOI/MD5; every paper acquired anywhere lands here so it is never
re-fetched. This is the thing that *compounds*: the corpus you build by using
the tool is the moat. Ships empty — it grows as you ask questions.

Self-contained: this replaces a personal `PaperLibrary`-style stack. The
schema is intentionally the same shape so a personal library can be pointed at
(set ``LOCALEVIDENCE_LIBRARY``) or migrated.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .. import config

# Serializes store writes within a process. acquire()'s per-pull timeout can
# leave an orphan worker still writing a slug's files while a later same-slug
# pull writes them too (single-process multi-question runs); this keeps those
# non-atomic file writes from interleaving into a torn PDF/text. The catalog DB
# is separately guarded by PRAGMA busy_timeout.
_write_lock = threading.Lock()

LIBRARY_ROOT = config.LIBRARY_ROOT
PDFS = LIBRARY_ROOT / "pdfs"
TEXTS = LIBRARY_ROOT / "text"
INBOX = LIBRARY_ROOT / "inbox"      # drop PDFs here for the local-file provider
DB = LIBRARY_ROOT / "catalog.db"

for _d in (PDFS, TEXTS, INBOX):
    _d.mkdir(parents=True, exist_ok=True)


def norm_doi(doi: str | None) -> str:
    if not doi:
        return ""
    d = doi.strip().lower()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", d)


def slugify(doi: str = "", pmid: str = "", md5: str = "") -> str:
    if doi:
        return re.sub(r"[^a-z0-9]+", "-", norm_doi(doi)).strip("-")
    if pmid:
        return f"pmid-{pmid}"
    if md5:
        return f"md5-{md5}"
    raise ValueError("need doi, pmid, or md5 for a slug")


def connect() -> sqlite3.Connection:
    c = sqlite3.connect(DB)
    c.execute("PRAGMA busy_timeout=30000")   # wait, don't error, on concurrent writers
    c.execute("""CREATE TABLE IF NOT EXISTS papers (
        slug TEXT PRIMARY KEY,
        doi TEXT, md5 TEXT, pmid TEXT, cite_key TEXT,
        title TEXT, authors TEXT, year TEXT, journal TEXT,
        pdf_path TEXT, text_path TEXT, bytes INTEGER,
        source TEXT, added_ts TEXT
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_md5 ON papers(md5)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pmid ON papers(pmid)")
    return c


# Back-compat alias for callers that reached for the private name.
_conn = connect


def find(doi: str = "", pmid: str = "", md5: str = "") -> dict | None:
    """Return the catalog row for a paper if held (and its file still exists)."""
    c = connect()
    row = None
    if doi:
        row = c.execute("SELECT * FROM papers WHERE doi=?", (norm_doi(doi),)).fetchone()
    if not row and md5:
        row = c.execute("SELECT * FROM papers WHERE md5=?", (md5,)).fetchone()
    if not row and pmid:
        row = c.execute("SELECT * FROM papers WHERE pmid=?", (str(pmid),)).fetchone()
    cols = [d[0] for d in c.execute("SELECT * FROM papers LIMIT 0").description]
    c.close()
    if not row:
        return None
    rec = dict(zip(cols, row))
    # A row whose PDF was evicted but whose text remains is still usable; only
    # treat it as stale if it claims a PDF that has vanished AND has no text.
    if rec.get("pdf_path") and not Path(rec["pdf_path"]).exists() \
            and not (rec.get("text_path") and Path(rec["text_path"]).exists()):
        return None
    return rec


def upsert(rec: dict) -> dict:
    """Insert or replace a catalog row from a full dict of column values."""
    cols = ("slug", "doi", "md5", "pmid", "cite_key", "title", "authors", "year",
            "journal", "pdf_path", "text_path", "bytes", "source", "added_ts")
    row = {k: rec.get(k, "") for k in cols}
    c = connect()
    c.execute(
        "INSERT OR REPLACE INTO papers "
        "(slug,doi,md5,pmid,cite_key,title,authors,year,journal,pdf_path,text_path,bytes,source,added_ts) "
        "VALUES (:slug,:doi,:md5,:pmid,:cite_key,:title,:authors,:year,:journal,:pdf_path,:text_path,:bytes,:source,:added_ts)",
        row)
    c.commit()
    c.close()
    return row


def store_pdf(pdf_bytes: bytes, *, doi="", pmid="", md5="", title="", authors="",
              year="", journal="", cite_key="", source="") -> dict:
    """Write a PDF into the store, extract its text, and catalogue it."""
    from .extract import extract_text
    slug = slugify(doi=doi, pmid=pmid, md5=md5)
    pdf_path = PDFS / f"{slug}.pdf"
    text_path = TEXTS / f"{slug}.txt"
    with _write_lock:
        pdf_path.write_bytes(pdf_bytes)
        has_text = extract_text(pdf_path, text_path)
        return upsert(dict(
            slug=slug, doi=norm_doi(doi), md5=md5, pmid=str(pmid or ""),
            cite_key=cite_key, title=title, authors=authors, year=str(year or ""),
            journal=journal, pdf_path=str(pdf_path),
            text_path=str(text_path) if has_text else "",
            bytes=len(pdf_bytes), source=source,
            added_ts=datetime.now(timezone.utc).isoformat(timespec="seconds")))


def store_text(slug: str, title: str, text: str, *, source: str,
               journal: str = "", url: str = "", year: str = "") -> dict:
    """Catalogue a text-only document (e.g. a web-published guideline).

    Used by the guideline harvester: these have no DOI/PDF, just body text we
    want the passage index to retrieve over.
    """
    text_path = TEXTS / f"{slug}.txt"
    with _write_lock:
        text_path.write_text(text)
        return upsert(dict(
            slug=slug, doi="", md5="", pmid="", cite_key="", title=title, authors="",
            year=year or str(datetime.now().year), journal=journal, pdf_path=url,
            text_path=str(text_path), bytes=len(text.encode("utf-8")), source=source,
            added_ts=datetime.now(timezone.utc).isoformat(timespec="seconds")))


def import_pdf(path: str | Path, **meta) -> dict:
    """File a PDF you already hold into the library (no network)."""
    path = Path(path)
    existing = find(doi=meta.get("doi", ""), pmid=meta.get("pmid", ""))
    if existing:
        return {**existing, "_status": "already_have"}
    rec = store_pdf(path.read_bytes(), source=meta.pop("source", "imported"), **meta)
    return {**rec, "_status": "imported"}


def stats() -> dict:
    c = connect()
    n = c.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    nt = c.execute("SELECT COUNT(*) FROM papers WHERE text_path!=''").fetchone()[0]
    by = dict(c.execute("SELECT source, COUNT(*) FROM papers GROUP BY source").fetchall())
    c.close()
    return {"papers": n, "with_text": nt, "by_source": by}
