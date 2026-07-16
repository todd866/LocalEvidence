"""Guideline harvester — pull web-published clinical guidelines into the library.

Guidelines (e.g. RCH Melbourne CPGs, ASCIA, state pathways) are web pages without
DOIs, so the DOI-centric `ask` pipeline can't reach them — yet they are often the
convention-setters a clinician actually follows. This crawls a guideline source's
index, fetches each guideline, extracts the body text, and catalogues it into the
local library with a synthetic slug (source 'guideline:rch') so the persistent
passage index picks it up and every future question can ground in them.

Shipped sources: RCH (Melbourne CPGs), AIH (the Australian Immunisation
Handbook), ASCIA (allergy/immunology), and RANZCOG (obstetrics/gynaecology).
Adding more — NICE, state pathways, etc. — is a matter of writing another
crawler that ends in `library.store_text`.

Idempotent (skips guidelines already held unless --refresh) and politely paced.
Copyright stays the publisher's: harvested text lands in your personal corpus,
never shared.
"""

from __future__ import annotations

import html as _html
import re
import time
from typing import Optional

import requests

from . import config
from . import library  # self-contained local library (subpackage)


_UA = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-AU,en;q=0.9",
}

RCH_INDEX = "https://www.rch.org.au/clinicalguide/guideline_index/"
RCH_BASE = "https://www.rch.org.au"

# Index entries that are not clinical guidelines.
_RCH_SKIP = {
    "Parent_resources", "Retrieval_services", "CPG_Committee_Calendar",
    "CPG_feedback", "Nursing_Guidelines", "Paediatric_Improvement_Collaborative",
    "Paediatric_palliative_care_guidelines", "About_the_clinical_practice_guidelines",
    "guideline_index", "Search", "Resources",
}


def _clean_html(raw: str) -> str:
    """Crude but serviceable HTML -> readable text.

    Drops script/style/nav/header/footer, unescapes entities, collapses
    whitespace. Repeated chrome that survives is dropped downstream by the
    passage index's low-value-chunk filter; the goal here is to keep the
    guideline body intact, not to be a perfect reader.
    """
    h = re.sub(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", " ", raw, flags=re.S | re.I)
    h = re.sub(r"<(nav|header|footer|aside)\b.*?</\1>", " ", h, flags=re.S | re.I)
    # Prefer the main content region if present.
    m = re.search(r"<main\b[^>]*>(.*?)</main>", h, flags=re.S | re.I)
    if m and len(m.group(1)) > 800:
        h = m.group(1)
    h = re.sub(r"<[^>]+>", " ", h)
    h = _html.unescape(h)
    h = re.sub(r"[ \t]+", " ", h)
    h = re.sub(r"\s*\n\s*", "\n", h)
    return re.sub(r"\n{2,}", "\n", h).strip()


# Guideline sources. Adding one is a config entry: index URL, base, a regex that
# captures (path, name[, title]) per guideline link, an optional skip set, and a
# slug prefix. The crawl/harvest/index machinery below is source-agnostic, so the
# library grows toward the local convention-setters one small config entry at a time.
SOURCES: dict[str, dict] = {
    "rch": {
        "source": "guideline:rch",
        "journal": "RCH Clinical Practice Guidelines",
        "index_url": RCH_INDEX, "base": RCH_BASE, "slug_prefix": "rch-",
        "link_re": r"href=['\"](/clinicalguide/guideline_index/([A-Za-z][^/'\"#?]+)/?)['\"][^>]*>([^<]+)</a>",
        "skip": _RCH_SKIP,
    },
    "aih": {  # Australian Immunisation Handbook — NIP schedule + per-disease vaccine specifics
        "source": "guideline:aih",
        "journal": "Australian Immunisation Handbook",
        "index_url": "https://immunisationhandbook.health.gov.au/contents/vaccine-preventable-diseases",
        "base": "https://immunisationhandbook.health.gov.au", "slug_prefix": "aih-",
        "link_re": r"href=['\"](/contents/vaccine-preventable-diseases/([a-z0-9][a-z0-9-]+))['\"]",
        "skip": set(),
        # High-value childhood-schedule diseases the AIH index/sitemap does NOT
        # link (orphan pages, at inconsistent paths — verified 200). `seeds` are
        # fetched directly alongside the crawled index (name, full-path-from-base).
        "seeds": [
            ("measles", "/contents/vaccine-preventable-diseases/measles"),
            ("diphtheria", "/contents/vaccine-preventable-diseases/diphtheria"),
            ("pertussis-whooping-cough", "/pertussis-whooping-cough"),
            ("meningococcal-disease", "/meningococcal-disease"),
            ("pneumococcal-disease", "/pneumococcal-disease"),
            ("influenza-flu", "/influenza-flu"),
            ("covid-19", "/covid-19"),
        ],
    },
    "ascia": {  # ASCIA — allergy/immunology guidelines & position papers (anaphylaxis, food/drug allergy)
        "source": "guideline:ascia",
        "journal": "ASCIA Guidelines & Position Papers",
        "index_url": "https://www.allergy.org.au/hp/papers",
        "base": "https://www.allergy.org.au", "slug_prefix": "ascia-",
        "link_re": r"href=['\"](/hp/papers/([a-z0-9][a-z0-9-]+))['\"]",
        "skip": set(),
        "skip_prefixes": ("references-", "id-register"),  # bibliography/registration pages, not guidelines
    },
    "ranzcog": {  # RANZCOG Statements & Guidelines — obstetric/gynae (PDF documents)
        "source": "guideline:ranzcog",
        "journal": "RANZCOG Statements & Guidelines",
        "index_url": "https://ranzcog.edu.au/womens-health/statements-guidelines/",
        "base": "https://ranzcog.edu.au", "slug_prefix": "ranzcog-",
        # statements are PDFs (full URLs on the index); capture (url, filename).
        "link_re": r"href=['\"](https?://ranzcog\.edu\.au/[^'\"]*?/([^/'\"]+)\.pdf)['\"]",
        "skip": set(),
    },
}


def crawl_index(session: requests.Session, cfg: dict) -> list[tuple[str, str, str]]:
    """Return (name, title, url) for every guideline link on a source's index.
    The link regex captures (path, name) and optionally (title); when the title
    is not in the href, it is derived from the name."""
    r = session.get(cfg["index_url"], timeout=30)
    r.raise_for_status()
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(cfg["link_re"], r.text):
        g = m.groups()
        path, name = g[0], g[1]
        title = (_html.unescape(g[2]).strip() if len(g) > 2 and g[2]
                 else name.replace("-", " ").title())
        if (name in cfg.get("skip", set()) or name.startswith(cfg.get("skip_prefixes", ()))
                or name in seen):
            continue
        seen.add(name)
        url = path if path.startswith(("http://", "https://")) else cfg["base"] + path
        out.append((name, title, url))
    return out


def crawl_rch_index(session: requests.Session) -> list[tuple[str, str, str]]:
    """Back-compat alias — RCH via the generic crawler."""
    return crawl_index(session, SOURCES["rch"])


def _extract_pdf(data: bytes) -> str:
    """Extract readable text from a PDF (many guideline statements — RANZCOG etc.
    — are PDFs, not HTML). Lazy pypdf import so HTML-only sources don't need it."""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def _fetch_text(session: requests.Session, url: str) -> str:
    """Fetch a guideline URL → readable body text, HTML or PDF (by content-type,
    .pdf extension, or the %PDF- magic). Returns '' on non-200 or empty extract."""
    r = session.get(url, timeout=45)
    if r.status_code != 200:
        return ""
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf") or r.content[:5] == b"%PDF-":
        return _extract_pdf(r.content)
    return _clean_html(r.text)


def _make_session(cfg: dict):
    """Fetch session for a source. Sources behind TLS-fingerprint bot protection
    (a plain-requests 403 — e.g. some state-health sites) set impersonate='chrome'
    etc.; we use curl_cffi (curl-impersonate) when installed, else fall back to
    requests (which will likely still 403 — install curl_cffi to reach them)."""
    imp = cfg.get("impersonate")
    if imp:
        try:
            from curl_cffi import requests as _creq
            return _creq.Session(impersonate=imp, headers=_UA)
        except ImportError:
            pass  # curl_cffi not installed — degrade to plain requests
    s = requests.Session()
    s.headers.update(_UA)
    return s


def _store_text(slug: str, title: str, text: str, *, source: str,
                journal: str, url: str = "") -> None:
    """Catalogue a web guideline into the library (text file + catalog row)."""
    library.store_text(slug, title, text, source=source, journal=journal, url=url)


def rch_slug(name: str) -> str:
    return "rch-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def harvest(source: str = "rch", *, limit: int = 0, pace_s: float = 0.7,
            refresh: bool = False, min_chars: int = 500, verbose: bool = True) -> dict:
    """Crawl + fetch + catalogue a guideline source (see SOURCES). Idempotent
    (skips slugs already held for that source unless refresh). Returns a summary."""
    cfg = SOURCES[source]
    s = _make_session(cfg)  # curl-impersonate for bot-protected sources, else requests
    guidelines = crawl_index(s, cfg)
    # Curated seed pages the index/sitemap misses (orphan pages at odd paths).
    for name, path in cfg.get("seeds", []):
        guidelines.append((name, name.replace("-", " ").title(), cfg["base"] + path))
    _seen: set[str] = set()  # dedupe by name (index and seeds can overlap)
    guidelines = [g for g in guidelines if not (g[0] in _seen or _seen.add(g[0]))]
    if verbose:
        print(f"  {source} index: {len(guidelines)} guidelines")
    if limit:
        guidelines = guidelines[:limit]

    con = library.connect()
    existing = {row[0] for row in
                con.execute("SELECT slug FROM papers WHERE source=?", (cfg["source"],)).fetchall()}
    con.close()

    def _slug(name: str) -> str:
        return cfg["slug_prefix"] + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    added = skipped = failed = thin = 0
    new_slugs: list[str] = []
    for i, (name, title, url) in enumerate(guidelines, 1):
        slug = _slug(name)
        if slug in existing and not refresh:
            skipped += 1
            continue
        try:
            text = _fetch_text(s, url)  # HTML or PDF
            if not text:
                failed += 1
                continue
            if len(text) < min_chars:
                thin += 1
                continue
            _store_text(slug, title, text, source=cfg["source"],
                        journal=cfg["journal"], url=url)
            added += 1
            new_slugs.append(slug)
            if verbose:
                print(f"  [{i}/{len(guidelines)}] + {slug}  ({len(text)} chars)  {title[:48]}")
        except requests.RequestException:
            failed += 1
        time.sleep(pace_s)

    summary = {"total": len(guidelines), "added": added, "skipped_existing": skipped,
               "failed": failed, "thin": thin, "new_slugs": new_slugs}
    if verbose:
        print(f"  harvest: +{added} added, {skipped} already held, "
              f"{failed} failed, {thin} too-thin")
    return summary


def harvest_rch(**kw) -> dict:
    """Back-compat alias — RCH via the generic harvester."""
    return harvest("rch", **kw)


def index_guidelines(source: str = "guideline:rch", verbose: bool = True) -> int:
    """Index any not-yet-indexed guidelines into the persistent passage store, so
    every future question retrieves over them (the store searches the whole
    corpus). Returns passages added."""
    from .acquire import AcquiredPaper
    from .index import PassageIndex

    con = library.connect()
    rows = con.execute(
        "SELECT slug,title,year,journal,text_path FROM papers "
        "WHERE source=? AND text_path!=''", (source,)).fetchall()
    con.close()

    papers = [AcquiredPaper(slug=s, title=t, year=str(y or ""), journal=j,
                            text_path=tp, tier="guideline", relevance=0.0,
                            status="guideline", source=source)
              for (s, t, y, j, tp) in rows]
    pidx = PassageIndex()
    added = pidx.add_papers(papers, verbose=verbose)
    if verbose:
        print(f"  indexed {added} guideline passages "
              f"(store now {pidx.stats()['passages']} passages / {pidx.stats()['papers']} papers)")
    return added
