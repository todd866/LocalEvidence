# LocalEvidence — Architecture

Top-to-bottom map of the system. See [`PHILOSOPHY.md`](PHILOSOPHY.md) for *why*.
Status flags: ✅ built · 🔶 partial · ⬜ planned.

## The compounding loop

```
                          ┌──────────────────────────────────────────┐
                          │              SIMULATED QUESTIONS          │  ✅ loader
                          │        (self-play to load the topic)      │
                          └───────────────────┬──────────────────────┘
                                              │ drives
   real question ───────────────────────────►▼
        │                          ┌──────────────────┐
        │   reuse / refresh ◄──────┤  KNOWLEDGE LEDGER │  ✅  Q + A + reasoning, grounded
        │                          └──────────────────┘
        ▼
  ┌───────────┐   ┌──────────┐   ┌──────────────────────────────────────┐   ┌────────────┐
  │ DISCOVER  │──►│  TRIAGE   │──►│        ACQUIRE (local library)        │──►│  STRUCTURE │
  │ OpenAlex  │✅ │ embed ×   │✅ │ have → pluggable provider cascade     │✅ │ embeddings │✅
  │ +graph ⬜ │   │ tier ✅   │   │ DOI/title-verified · filed · text     │   │ passages   │✅
  └───────────┘   └──────────┘   └──────────────────────────────────────┘   │ graphs ⬜  │
                                                                             │ condensed⬜│
                                                                             └─────┬──────┘
                                                                                   ▼
                                                          ┌───────────────────────────────┐
                                                          │  RETRIEVE (hybrid + rerank)   │  ✅ hybrid · ⬜ MedCPT
                                                          └──────────────┬────────────────┘
                                                                         ▼
                                                          ┌───────────────────────────────┐
                                                          │  SYNTHESISE (Claude) + ground │  🔶 pack ✅ · faithfulness ⬜
                                                          └──────────────┬────────────────┘
                                                                         ▼
                                                            answer  +  ledger entry  +  gaps
```

Every loop iteration deepens three stores that never get thrown away: the
**paper** store, the **structure** store, and the **knowledge** store.

## Layers, bottom to top

### 1. Sources & acquisition — ✅ (legal tiers) · 🔌 (pluggable last tier)

`localevidence.library.pull(doi, title=, pmid=)` runs a **provider cascade**.
Each provider returns candidate PDF bytes; the cascade title-verifies every
candidate (`pdf_matches_title`) before cataloguing it, so a wrong file is
rejected, not stored. Filing is deduped by DOI/MD5 and text-extracted.

Shipped providers (all legal, all open-access):
- **local-file drop** — a PDF you place in `inbox/` named by DOI slug;
- **Unpaywall** — the OA copy of a DOI when one exists;
- **Europe PMC** — green-OA / PMC author manuscripts.

The **last-resort tier is pluggable and not shipped** — `ShadowProvider` is a
documented stub that raises `NotImplementedError`, so by default the cascade is
open-access only. You supply your own implementation if you want more (env
`LOCALEVIDENCE_SHADOW`, or a git-ignored `providers/private.py`). See
[`docs/ACQUISITION.md`](docs/ACQUISITION.md). Adding a new provider is one class
with a `fetch(doi, ...) -> bytes | None` method; `LocalFileProvider` is the
minimal worked example.

The library ships **empty**: the corpus is what you grow by using the tool.

### 2. Structure — how the library is mapped

**Structure is corpus-gated: these maps are only as good as the corpus under
them.** A citation graph over fifteen papers is noise; topic clusters over a thin
corpus just reflect the seed bias back. So for a cold start, *grow the foundation
first* — the loader, acquisition breadth, and dedup rank ahead of the graph/cluster
maps, and the planned maps should switch on only past a corpus-size threshold (and
no-op gracefully below it). The exception is **citation-based snowball discovery**
(§roadmap): it follows live citation edges *outward* to grow the corpus from a
seed, so it earns its keep even when the library is small. The asymmetry to
remember: a fresh clone ships empty, so its maps are premature, but an operator
pointing `LOCALEVIDENCE_LIBRARY` at an existing large corpus gets value from them
immediately.

What actually runs today, and what is designed but not yet built:

- **Dense embeddings** ✅ — MiniLM (`all-MiniLM-L6-v2`, 384-d, L2-normalized)
  over every passage. One model, one vector space for passages, queries, and the
  ledger's questions.
- **Lexical index** ✅ — SQLite FTS5 BM25 over the same passages (exact drug
  names, thresholds, dose strings the dense model blurs).
- **Citation graph** ⬜ — *not built yet.* Design: OpenAlex
  `referenced_works`/`cited_by` to build a directed citation graph for snowball
  discovery and authority ranking. (Today, discovery is OpenAlex relevance
  search only — no graph traversal.)
- **Claim / contradiction graph** ⬜ — atomic claims (entity · relation · value ·
  provenance) linked to source passages; powers contradiction detection
  (guideline A: admit HR<40; B: <50).
- **Condensed cluster objects (CKO)** ⬜ — the "learned representation" (below).

Other library-mapping methods worth adding (none shipped): **co-citation /
bibliographic coupling** (cluster papers by shared references), **topic
clustering** (k-means / HDBSCAN over embeddings → the CKO basins), **concept
tagging** (MeSH / UMLS terms per passage for a controlled-vocabulary index),
**temporal / recency mapping** (so the newest RCT isn't buried), and an
**author / venue graph**. The honest current state: two maps run (dense + BM25);
the graph maps are designed, not implemented.

### 3. Retrieval — ✅ hybrid · ⬜ rerank

Hybrid: FTS5 BM25 (lexical) + MiniLM dense (semantic), fused by Reciprocal Rank
Fusion, with reference/table chunk hygiene, a guideline/SR tier-guarantee scoped
to the current question's own papers, and a relevance-gated **local-guideline
guarantee** (terse bedside guidelines lose the dense race to journal prose, so
the most query-relevant one is guaranteed a seat above a floor). Next: a
**MedCPT** cross-encoder reranker (open, PubMed-trained) over the fused top-k.

### 4. Synthesis — 🔶

The pipeline emits a grounded **evidence pack** (ranked passages + provenance +
coverage + gaps). Claude (in the loop — no paid API) synthesises the answer per
[`docs/answer-style.md`](docs/answer-style.md) and writes it back into the ledger
(`localevidence answer`). Next: a **faithfulness gate** — every
management-changing claim must be entailed by a retrieved passage, else hedge or
refuse.

### 5. Knowledge ledger — ✅

`ledger/answers.jsonl` + a question-embedding index. Each entry:
`{id, ts, question, evidence:[{slug, doi, passage_ids}], answer, reasoning,
grounding, confidence, gaps, coverage}`; the question embedding is stored
row-aligned in a `questions.npy` side-car (not inline in the JSONL). On a new
question the pipeline checks `find_similar(question)` and reports prior worked
answers for reuse/refresh. The answer + reasoning are filled at synthesis time (a worked
result, not just a cache of papers). This is the layer a closed product cannot
replicate.

### 6. Loader — ✅

`localevidence load` runs `ask` over a **question bank**
(`data/question_bank.yaml`) in your domains. Idempotent, polite-paced, resumable.
It is simultaneously the warm-up (fills papers/structure/ledger) and the **eval
harness** (coverage per question). Ships with a generic clinical/paediatric bank;
replace it with your own.

### 7. Guidelines — ✅

`localevidence guidelines` harvests web-published clinical guidelines (RCH
Melbourne CPGs shipped as the worked example) into the library as text docs, so
the convention-setters a clinician actually follows are retrievable even though
they have no DOI. Adding ASCIA / NICE / state pathways is another small crawler
ending in `library.store_text`.

### 8. Serve — ✅

`localevidence serve` is a zero-dependency stdlib server that loads the index +
ledger + model once and stays warm, plus an installable offline PWA
(`localevidence/webapp/`). Reach it from your phone over your **LAN or tailnet —
never the open internet**. Worked answers cache offline; a novel question returns
live evidence and queues for the next home deep-run.

### 9. Audit — ✅

`localevidence audit` (module `audit.py`) reconstructs how an answer was produced
and makes it independently re-checkable — auditability as a safety property, and
the concrete case for open clinical tooling. For a ledger entry it assembles the
discovery → triage → acquisition → retrieval chain from the run checkpoints, runs
the **citation provenance check** (were the answer's cited sources — by DOI or by
name matched to a retrieved title — actually retrieved this session? a citation
not in the retrieval set is flagged as possibly introduced from memory; `--resolve`
adds a live doi.org existence check. It verifies retrieval *presence*, not whether
the source supports the specific claim — that is the manual claim-support step),
marks which retrieved sources were used vs available, and reports a **verification
ceiling** (rung 0–9: how far back a third party can reconstruct it, up to
"end-to-end re-runnable"). The tool reports its own ceiling honestly — a
ledger-only answer tops out lower than one with full run checkpoints.

## Condensed cluster objects (CKO) — the condensation layer ⬜

The "coded compressed thing" so we don't re-read whole papers, built over the
passage index. Cluster passages into topic basins, then condense each into a CKO:

```
{ cluster_id, label, centroid_vec,
  canonical_claims: [ { text, value/threshold, support:[passage_ids], tier,
     agreement, determination, range } ],
  contradictions: [ {claim_a, claim_b, sources} ],
  representative_passages: [...], papers: [...], year_span }
```

Use the CKO instead of the cluster's full text; only the residual
(contradictions, gaps, novelty) sends you back to full passages.

### Do not over-condense the granular layer

A topic has a **stable core** (consensus claims condensation collapses
losslessly) and an **underdetermined granular layer** (exact thresholds, doses,
cut-points). The granular number is often a *free parameter the published
evidence does not identify* — only a dedicated A/B trial would settle it, and it
usually hasn't run, so different guidelines pick different conventions. Every
granular claim carries a `determination` status:

- **stable-core** → state the value plainly.
- **underdetermined-granular** → give the *defensible range*, flag it as
  convention not hard evidence, name the local guideline's choice, and state what
  a trial would need to settle.

This is also where **targeted acquisition** matters: society/local guidelines are
decisive at the granular layer and add nothing at the core, so wider acquisition
is surgical.

## Locations

```
localevidence-public/
  localevidence/          # the engine
    library/              # the self-contained local library
      catalog.py          #   SQLite catalog + PDF/text store
      extract.py          #   PDF text extraction + title-match content guard
      chunk.py            #   passage chunking
      providers/          #   pluggable acquisition (localfile, unpaywall,
                          #   europepmc shipped; shadow = documented stub;
                          #   private.py = your git-ignored implementation)
    discovery/ triage/ acquire/ index/ evidence/ ledger/ loader/ guidelines/
    pipeline.py server.py webapp/
  data/
    question_bank.yaml    # the loader's simulated questions (ships generic)
    library/              # the corpus: catalog.db + pdfs + text  (git-ignored)
    passages/             # persistent passage index               (git-ignored)
  ledger/                 # the knowledge ledger                    (git-ignored)
  projects/<slug>/        # per-question case files                 (git-ignored)
  docs/  PHILOSOPHY.md  ARCHITECTURE.md  README.md
```

The code is the repo; the corpus, the index, the ledger, your case files, and any
private acquisition provider are git-ignored — see "Dogfooding" below.

## Dogfooding — running the public repo as a daily driver

This repository is meant to be *used*, by its author first, for a long time. The
design keeps the shareable code and the private operation cleanly separated so
that using it improves it:

- **Code** (the `localevidence/` package, docs, the question-bank schema) is
  tracked and improved as you use it — fix a retrieval bug at the bedside, commit
  it, push it.
- **Corpus + state** (`data/library/`, `data/passages/`, `ledger/`, `projects/`)
  are git-ignored. Point `LOCALEVIDENCE_LIBRARY` at an existing corpus, or let it
  grow from empty.
- **Your acquisition provider** lives in a git-ignored
  `localevidence/library/providers/private.py` (or behind `LOCALEVIDENCE_SHADOW`),
  so the implementation is never committed while the code that *calls* it is.

The result: you run this exact public repo, every improvement you make to the
engine is a commit, and nothing private — corpus, patient context, or
acquisition method — ever enters git history.

## Build order (roadmap)

1. ✅ ask loop (discover→triage→acquire→index→pack), resumable.
2. ✅ self-contained library + pluggable provider cascade (legal OA shipped).
3. ✅ persistent passage index (hybrid dense + BM25, tier/guideline guarantees).
4. ✅ knowledge ledger (store Q + A + reasoning; reuse lookup; writeback).
5. ✅ simulated-question loader + seed bank.
6. ✅ guideline harvester (RCH worked example).
7. ✅ serve + offline PWA.
8. ✅ audit layer (provenance trail + citation grounding check + verification ceiling).
8b. ✅ knowledge packs (`pack export`/`harvest`): the shareable list + summaries + map of a corpus, minus the copyrighted PDFs — corpus distributable as a public good (see `docs/PACK.md`).
9. ⬜ citation + claim graphs (the graph maps — designed, not built).
10. ⬜ CKO condensation layer (cluster → distil → contradiction set).
11. ⬜ MedCPT reranker + faithfulness gate.
12. ⬜ single-writer lock for safe concurrent `ask`/`load`/`serve`.
