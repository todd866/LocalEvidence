# LocalEvidence

**Answer your own clinical questions from a library of papers you own, with every
claim traceable to the source it came from.** LocalEvidence finds the relevant
literature, acquires the full text, and assembles a grounded, cited evidence pack
on your machine — and the library compounds as you use it.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Under the hood it's **scaffolding around a capable AI agent** (you drive it from
Claude Code; the agent does the synthesis, no paid API). The bet: an agent already
rivals closed tools like OpenEvidence on clinical questions, but on its own it has
no persistent corpus, no memory of local conditions, and a habit of confident
hallucination — which LocalEvidence counters by grounding every claim to a
retrievable passage and recording an auditable provenance trail. It fills exactly
those three gaps: a paper library you own, a ledger of worked answers, and
passage-grounded retrieval. Everything stays on your machine.

The point is not a clever one-shot retriever. It is that **the corpus compounds**:
every paper a question pulls is kept, so the next related question is faster and
better-grounded. You come home with one or two burning questions, set it to work,
and over weeks it grows a library shaped to the medicine you actually practise.
A closed product serves everyone the same way and cannot know you; this gets
better the more *you* use it.

> **Scope.** A personal reference aid for a clinician's own use — not a validated
> medical device, not autonomous clinical decision-making. Every claim is grounded
> to a retrievable passage; provenance and reasoning are recorded, not hidden.

## Three tools, three jobs

LocalEvidence is one of three linked but distinct tools:

| Tool | Does | Input → Output |
|---|---|---|
| [PaperScope](https://github.com/todd866/PaperScope) | **Analyzes** the literature — bibliography / DOI / retraction QA, forensic metascience, systematic reviews, embeddings. | papers → checked analysis |
| **LocalEvidence** (this repo) | **Answers** a clinical question from a library you own, grounded and cited. | a question + your library → a cited evidence pack |
| [EvidenceViewer](https://github.com/todd866/EvidenceViewer-public) | **Presents** any source-backed artifact through one contract + viewer, every claim traceable to its source. | an EvidenceArtifact → a source-linked reading UI |

The pipeline: **PaperScope analyzes → LocalEvidence answers** (using PaperScope's fact-checking) **→ EvidenceViewer presents** either one's output.

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

**Beyond open access, the cascade's last-resort tier** resolves a DOI on a *shadow
library* (Anna's Archive / LibGen / Sci-Hub) when no open-access copy exists.
**That tier is deliberately NOT included in this repository** — it's the one part
you supply yourself. What ships is the *seam*: a documented `ShadowProvider`
interface that raises
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

## Run it fully local (free, on-device)

The synthesis step doesn't have to be Claude-in-the-loop — it can be a **free,
local open-weight model**. Point `synthesize` at an [Ollama](https://ollama.com)
server and a 14B model on a laptop returns a grounded, cited answer with no paid
API and nothing leaving the machine:

```bash
ollama pull qwen2.5:14b
LOCALEVIDENCE_PASSAGES=/path/to/corpus \
  python3 -m localevidence synthesize \
    "Empiric antibiotic for a child with bacterial meningitis?" \
    --model ollama:qwen2.5:14b          # or set LOCALEVIDENCE_MODEL
```

This is deliberate: the safety lives in the **corpus and grounding**, not the
model. The prompt constrains the model to answer *only* from the retrieved
passages and cite them — so a small local model declines to guess when the
evidence is thin rather than confabulating. The whole stack — open corpus + open
model + local inference — then runs on-prem, private, and inspectable, which is
the strongest substrate for treating the *deployment*, not the model's
capability, as the safety variable.

**Comparing models / sweeping model grades (optional, eval only).** The model is a
swappable backend, so the same grounded question can be put to *any* AI to compare
how each handles it: `ollama:<name>` (local, free), or — opt-in, with your own key —
`anthropic:<model>` / `openai:<model>`, or `openrouter:<provider/model>`
(`OPENROUTER_API_KEY`). **OpenRouter is the efficient one**: one key reaches ~every
model, so you can sweep *grades* (`openrouter:qwen/qwen-2.5-72b-instruct`,
`…/llama-3.3-70b-instruct`, frontier APIs) and **simulate what a local model of that
size would do without the hardware to run it** — e.g. answering "does a 72B close the
reasoning gap?" without a 64 GB box. These paid backends are purely for the
cross-model evaluation arm; the default stack stays free-local / Claude-in-the-loop.

## Quickstart

```bash
git clone https://github.com/<you>/localevidence.git
cd localevidence
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                                     # optional: enables the bare `localevidence` command
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

# Audit a worked answer: emit its full provenance trail + citation-provenance check:
python3 -m localevidence audit -e 24   # or --project <slug>, --json, --resolve
```

Commands: `ask` (the engine), `answer` (write a synthesised answer into the
ledger), `verify` (retrieve evidence + citation-provenance for one claim, headless),
`audit` (provenance + citation-provenance check), `synthesize` (grounded answer via
a free local model; `--harness` for the multi-stage loop), `eval-local` (run the
harness over many questions on-device and score grounding/reasoning — see
[`evals/`](evals/)), `index-library` (index a corpus you already hold), `pack`
(export/harvest a shareable knowledge pack), `queue` (drain the phone queue),
`load` (self-play a question bank), `guidelines` (harvest CPGs), `operating-point`
(show the deterministic local decision for a probability — the settable dial a
closed tool won't expose; see [`examples/operating_points/`](examples/operating_points/)),
`serve` (backend + PWA). `--help` on each.

### Bulk-index a library you've built up

`ask` indexes papers as it pulls them. But if you've accumulated full-text papers
another way — dropped a batch of PDFs into the library's `inbox/`, harvested a
pack, or grown the library over many sessions — `index-library` chunks + embeds
them all into retrieval in one pass (incremental; already-indexed papers skipped):

```bash
python3 -m localevidence index-library
python3 -m localevidence index-library --match "amyotrophic|motor neuron"   # a topic subset
```

The library is the durable store; the passage index is the retrieval layer over
it. *Power-user note:* the `library` backend is pluggable via `LOCALEVIDENCE_LIBRARY`,
so if you keep a separate paper store with a compatible catalog, LocalEvidence can
sit on top of it instead of duplicating it — which is how it pairs with a fuller
acquisition/storage stack.

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

**Clone-and-run, not a wheel.** State (corpus, passage index, ledger, projects)
lives under the repo by default, so run from a clone (`pip install -e .` just adds
the `localevidence` command). To keep state elsewhere, point the env vars at it:
`LOCALEVIDENCE_PASSAGES` (passage index), `LOCALEVIDENCE_LIBRARY` (paper store),
`LOCALEVIDENCE_MODEL` (local model), `LOCALEVIDENCE_EMAIL` (OA contact). Acquisition
is hang-guarded: `LE_PULL_BUDGET_S` bounds each pull (default 120s; a slow mirror is
abandoned and recorded as a failure) and `LE_ACQUIRE_BUDGET_S` caps a question's
total acquisition time (default 0 = unlimited; once spent, remaining candidates are
skipped). Easiest:
`cp .env.example .env` and fill it in — the gitignored `.env` is auto-loaded at
startup (a real exported env var always overrides it), and it's where the optional
`OPENROUTER_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` live too.

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
of the subset of a personal paper-management stack that this tool needs (catalog,
text extraction, title verification, OA acquisition, chunking). A fuller private
deployment can point at a larger private library and add the last-resort tier
above; this repository runs standalone with neither.

## License

MIT — see [`LICENSE`](LICENSE). The code is MIT; the medical literature it
retrieves is not — respect each publisher's copyright and your own institution's
access terms.
