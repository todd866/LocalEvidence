# LocalEvidence

**Scaffolding that makes Claude Code a better clinical-evidence engine than it is
on its own. Claude Code is the interface; this is the substrate — a local-first,
*compounding* corpus with grounded retrieval and a memory of your past questions.**

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

The bet is simple: a capable agent already rivals closed tools like OpenEvidence
on clinical questions, but on its own it has no persistent corpus, no memory of
local conditions, and a habit of confident hallucination. LocalEvidence fills
exactly those three gaps — a paper library you own, a ledger of worked answers,
and passage-grounded retrieval — so the agent answers better, on your topics,
and auditably. You drive it from Claude Code; everything stays on your machine.

Concretely: it answers a clinical question by discovering the relevant literature,
acquiring the full text it needs, indexing it at passage level, and assembling a
grounded evidence pack that the agent (in the loop — no paid API) synthesises into
a cited answer.

The point is not a clever one-shot retriever. It is that **the corpus compounds**:
every paper a question pulls is kept, so the next related question is faster and
better-grounded. You come home with one or two burning questions, set it to work,
and over weeks it grows a library shaped to the medicine you actually practise.
A closed product serves everyone the same way and cannot know you; this gets
better the more *you* use it.

> **Scope.** A personal reference aid for a clinician's own use — not a validated
> medical device, not autonomous clinical decision-making. Every claim is grounded
> to a retrievable passage; provenance and reasoning are recorded, not hidden.

## How it works

```
question ─► discover ─► triage ─► acquire ─► index ─► evidence pack ─► (Claude) answer ─► ledger
            OpenAlex    relevance  provider   hybrid    ranked, cited     grounded        reused
                        × tier     cascade    dense+BM25  passages         synthesis       next time
```

Two stores grow and are never thrown away: the **library** (papers) and the
**passage index + ledger** (structure + worked answers). See
[`PHILOSOPHY.md`](PHILOSOPHY.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Acquisition: legal by default, extensible by you

Acquisition is a **provider cascade**. Each provider tries to return a PDF for a
DOI; the cascade verifies every candidate against the paper's title before it is
stored, so a wrong file is rejected rather than catalogued. The providers that
**ship and work out of the box are all legal** — open access, or files you
already hold and are entitled to use:

- a **local-file drop** (`inbox/` — any PDF you legitimately have),
- **Unpaywall** (the open-access copy when one exists),
- **Europe PMC** (green-OA / PMC author manuscripts).

These cover a large fraction of the literature on their own.

**The author runs a further, last-resort tier** that resolves a DOI on a *shadow
library* (Anna's Archive / LibGen / Sci-Hub) when no open-access copy exists.
**That machinery is deliberately NOT included in this repository.** What ships is
the *seam*: a documented `ShadowProvider` interface that raises
`NotImplementedError`, so the default cascade is open-access only. If you want to
use it that way, you rebuild that tier yourself — the interface, the reasons each
safety guard exists, and the conceptual steps are spelled out in
[`docs/ACQUISITION.md`](docs/ACQUISITION.md). A provider is one small class with a
`fetch(doi, ...) -> bytes | None` method; `localevidence/library/providers/localfile.py`
is the minimal worked example. Whether to add such a tier, and how, is your
decision, for your own use, in your own jurisdiction.

This split is deliberate: the **architecture** is the shared public good; the
**corpus** you grow and the **acquisition tier** you add are yours.

## Built to run with Claude Code

This is designed to be operated by a capable coding agent. The synthesis step
(turning an evidence pack into a cited answer) is Claude *in the loop* — you read
the grounded pack and write the answer back into the ledger — not a metered API
call. Reconstructing the optional acquisition tier, adding a new guideline
crawler, wiring a citation graph: these are exactly the tasks you hand to Claude
Code inside this repo. The repository is the kernel; the agent does the rest.

## Quickstart

```bash
git clone https://github.com/<you>/localevidence.git
cd localevidence
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export LOCALEVIDENCE_EMAIL="your.name@example.org"   # YOUR real address — Unpaywall 422s the placeholder

# Ask a question (open-access acquisition by default):
python3 -m localevidence ask \
  "In anorexia nervosa with bradycardia, what HR threshold warrants admission?" \
  -q "eating disorder inpatient admission cardiovascular" --top-n 15
# -> projects/<slug>/runs/<run_id>/evidence-pack.md   (then synthesise + `answer`)

# Warm the system on a bank of simulated questions (this is also the eval harness):
python3 -m localevidence load --limit 8

# Harvest web-published guidelines (RCH paediatrics shipped as the example):
python3 -m localevidence guidelines --source rch

# Serve the phone face (installable offline PWA) over your LAN / tailnet:
python3 -m localevidence serve         # http://127.0.0.1:8765

# Audit a worked answer: emit its full provenance trail + grounding check:
python3 -m localevidence audit -e 24   # or --project <slug>, --json, --resolve
```

Commands: `ask` (the engine), `answer` (write a synthesised answer into the
ledger), `audit` (provenance + grounding trail), `pack` (export/harvest a
shareable knowledge pack), `queue` (drain the phone queue), `load` (self-play a
question bank), `guidelines` (harvest CPGs), `serve` (backend + PWA). `--help`
on each.

## Pack: distribute the map, not the corpus

You can't legally share a corpus of copyrighted PDFs — but you can share the
knowledge *about* it. `localevidence pack export` writes a **knowledge pack**: the
**paper list** (bibliographic metadata), **summaries** of what each paper provides
(your own words), and a **map** (topic clusters + a nearest-neighbour graph). No
full text, no PDFs, no verbatim passages. Commit it to a public repo; anyone runs
`localevidence pack harvest` to acquire the papers under their own access and
**rebuild the corpus locally**. The compounding knowledge travels as a public good;
the copyright stays home. See [`docs/PACK.md`](docs/PACK.md).

## Audit: every answer ships its trail

Auditability is the whole point — and a safety property. `localevidence audit`
reconstructs how an answer was produced: the discovery → triage → acquisition →
retrieval chain, the **citation provenance check** (were the answer's cited
sources — by DOI, or by name matched to a retrieved paper's title — actually
retrieved this session, or introduced from the model's memory? a citation that
isn't in the retrieval set is flagged; note this verifies retrieval *presence*,
not whether the source supports the specific claim — that is the manual step),
which retrieved sources were used vs available, the logged gaps, and a
**verification ceiling** (how far back an independent party can reconstruct it —
up to "end-to-end re-runnable"). A closed product can show you its answer and a
reference list; an open one can show you the whole chain, and let you re-run it.
That difference is the case for open clinical-evidence tooling.

## Requirements

Python 3.10+, and the packages in [`requirements.txt`](requirements.txt)
(`sentence-transformers`, `numpy`, `requests`, `pyyaml`; `PyMuPDF` recommended
for robust PDF text). `pdftotext` (poppler) is used if present. SQLite FTS5 ships
with CPython. First run downloads the MiniLM model (~90 MB) once.

## Dogfooding: run this repo, improve this repo

The repo is meant to be the author's daily driver. The code is tracked; the
private parts never are:

- `data/library/`, `data/passages/`, `ledger/`, `projects/` — your corpus and
  state — are **git-ignored**. Point `LOCALEVIDENCE_LIBRARY` at an existing
  corpus or let it grow from empty.
- Your acquisition provider lives in a **git-ignored**
  `localevidence/library/providers/private.py` (exporting `Provider`), or behind
  the `LOCALEVIDENCE_SHADOW` env — so the implementation is never committed while
  the code that calls it is.

Use it at the bedside, fix what annoys you, commit, push. Improvements flow back;
nothing private enters git history.

## Provenance

The `localevidence/library/` package is a clean, self-contained reimplementation
of the subset of the author's personal paper-management stack that this tool
needs (catalog, text extraction, title verification, OA acquisition, chunking).
The author's own instance points at a larger private library and the unshipped
acquisition tier above; this repository runs standalone with neither.

## License

MIT — see [`LICENSE`](LICENSE). The code is MIT; the medical literature it
retrieves is not — respect each publisher's copyright and your own institution's
access terms.
