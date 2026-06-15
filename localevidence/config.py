"""Shared paths, constants, and lightweight helpers for the pipeline.

Local-first and self-contained: everything resolves to on-disk locations under
the repository (or paths you set via environment variables). No service and no
API key are required to run the pipeline. The polite-pool literature APIs
(OpenAlex, Unpaywall, Europe PMC) ask for a contact email — set one with
``LOCALEVIDENCE_EMAIL`` so you are a good citizen of those free services.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# --- Roots -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]               # repo root
PROJECTS = ROOT / "projects"


def load_dotenv(path: Path) -> None:
    """Load KEY=value lines from a gitignored .env into the environment, so an
    operator can keep keys (LOCALEVIDENCE_EMAIL, *_API_KEY, …) in one local file.
    `setdefault` means a real exported env var always wins; missing file is a no-op.
    Mirrors how paperscope/shadow.py loads PaperLibrary/.env."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# Load the gitignored .env BEFORE the os.environ reads below. Override its
# location with LOCALEVIDENCE_ENV.
load_dotenv(Path(os.environ.get("LOCALEVIDENCE_ENV", ROOT / ".env")))

# The durable local paper store (catalog + PDFs + extracted text + the passage
# index). Keep it out of the repo by default; override with LOCALEVIDENCE_LIBRARY.
LIBRARY_ROOT = Path(
    os.environ.get("LOCALEVIDENCE_LIBRARY", ROOT / "data" / "library")
)

# The passage index (retrieval layer). Default lives in the repo; override with
# LOCALEVIDENCE_PASSAGES to serve/verify against a corpus you keep elsewhere (e.g.
# a private full-stack instance) without copying code into it.
PASSAGES_DIR = Path(
    os.environ.get("LOCALEVIDENCE_PASSAGES", ROOT / "data" / "passages")
)

# Polite-pool contact for OpenAlex / Unpaywall / Europe PMC. These free APIs
# rate-limit anonymous traffic and prioritise requests that carry a real email.
# Set your OWN: Unpaywall rejects this placeholder outright (HTTP 422) and the
# others throttle it, so the OA cascade is degraded until you export a real one.
CONTACT_EMAIL = os.environ.get("LOCALEVIDENCE_EMAIL", "you@example.com")

# Embedding model. The same model embeds queries, corpus passages, and the
# ledger's questions, so everything lives in one vector space.
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Evidence tiers ----------------------------------------------------------
# Higher tier_rank = stronger evidence for a clinical answer. Detected from
# title/abstract/type; deliberately coarse (this only biases triage ordering).

TIER_RANK = {
    "guideline": 6,
    "systematic-review": 5,
    "rct": 4,
    "cohort": 3,
    "review": 2,
    "article": 1,
    "other": 0,
}

_TIER_PATTERNS = [
    ("guideline", re.compile(
        r"\b(guideline|guidance|recommendation|consensus statement|position statement|"
        r"practice parameter)\b", re.I)),
    ("systematic-review", re.compile(
        r"\b(systematic review|meta-analysis|meta analysis|cochrane|network meta)\b", re.I)),
    ("rct", re.compile(
        r"\b(randomi[sz]ed controlled trial|randomi[sz]ed[, ]|double-blind|placebo-controlled|"
        r"\brct\b)\b", re.I)),
    ("cohort", re.compile(
        r"\b(cohort study|prospective cohort|retrospective cohort|case-control|"
        r"observational study|registry)\b", re.I)),
    ("review", re.compile(
        r"\b(review|narrative review|scoping review|seminar|state of the art)\b", re.I)),
]


def classify_tier(title: str, abstract: str, openalex_type: str = "") -> str:
    """Coarse evidence-tier label from text + OpenAlex work type."""
    hay = f"{title or ''}  {abstract or ''}"
    for label, pat in _TIER_PATTERNS:
        if pat.search(hay):
            return label
    t = (openalex_type or "").lower()
    if "review" in t:
        return "review"
    if t in ("article", "journal-article", "proceedings-article"):
        return "article"
    return "other"


# --- DOI / id helpers --------------------------------------------------------

def norm_doi(doi: str | None) -> str:
    if not doi:
        return ""
    d = doi.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d


def slugify_doi(doi: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", norm_doi(doi)).strip("-")


_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "with", "without", "in", "on",
    "to", "is", "are", "be", "this", "that", "these", "those", "should", "would",
    "could", "what", "when", "which", "how", "does", "do", "i", "my", "patient",
    "patients", "clinical", "vs", "versus", "use", "using", "give", "given",
}


def key_terms(text: str, limit: int = 12) -> list[str]:
    """Content terms from a clinical question, order-preserving, deduped."""
    raw = re.findall(r"[A-Za-z][A-Za-z0-9+\-]{2,}", (text or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        t = t.strip("-+")
        if len(t) < 3 or t in _STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:limit]
