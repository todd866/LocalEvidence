"""Guideline harvester — pull web-published clinical guidelines into the library.

Guidelines (e.g. RCH Melbourne CPGs, ASCIA, state pathways) are web pages without
DOIs, so the DOI-centric `ask` pipeline can't reach them — yet they are often the
convention-setters a clinician actually follows. This crawls a guideline source's
index, fetches each guideline, extracts the body text, and catalogues it into the
local library with a synthetic slug (source 'guideline:rch') so the persistent
passage index picks it up and every future question can ground in them.

The shipped source is RCH (a worked example of the pattern); adding ASCIA, NICE,
state pathways, etc. is a matter of writing another crawler that ends in
`library.store_text`.

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
        if name in cfg.get("skip", set()) or name in seen:
            continue
        seen.add(name)
        out.append((name, title, cfg["base"] + path))
    return out


def crawl_rch_index(session: requests.Session) -> list[tuple[str, str, str]]:
    """Back-compat alias — RCH via the generic crawler."""
    return crawl_index(session, SOURCES["rch"])


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
    s = requests.Session()
    s.headers.update(_UA)
    guidelines = crawl_index(s, cfg)
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
            r = s.get(url, timeout=30)
            if r.status_code != 200:
                failed += 1
                continue
            text = _clean_html(r.text)
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
