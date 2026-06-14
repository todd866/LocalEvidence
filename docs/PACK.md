# Knowledge packs — distributing a corpus you can't distribute

A compounding corpus is LocalEvidence's moat — but it's copyrighted PDFs, so you
can't hand it to anyone. A **knowledge pack** is the part you *can* share: the map
of the corpus, minus the corpus.

## What's in a pack (and why it's shareable)

| File | What | Why it's not the publisher's content |
|------|------|--------------------------------------|
| `papers.jsonl` | the **paper list** — DOI, title, authors, year, journal, evidence tier | bibliographic *facts*, not expression |
| `summaries.jsonl` | **what each paper provides** — your own-words summary | a paraphrase you generate (NOT the abstract) |
| `map.json` | the **map** — topic clusters + a nearest-neighbour similarity graph | derived *structure*, not text |
| `README.md` | what the pack is + how to rebuild from it | — |

A pack contains **no full text, no PDFs, no verbatim passages, and no raw embedding
vectors** — only facts, your summaries, and derived structure. That boundary is
enforced in `pack.py` and checked in the tests.

## Build and share a pack

```bash
localevidence pack export ./mnd-pack        # writes papers.jsonl + summaries.jsonl + map.json
# (summaries start empty — fill them with your own-words summaries, the
#  Claude-in-the-loop step, the same pattern as ask -> answer)
```

Commit `./mnd-pack` to a public repo. It tells anyone *what literature grounds this
topic and how it fits together* — without shipping a byte of copyrighted text.

## Rebuild the corpus from a pack

```bash
localevidence pack harvest ./mnd-pack       # acquires each DOI under YOUR access, then indexes
```

The recipient harvests the papers themselves, through whatever acquisition they
choose — the same pluggable cascade as everywhere else (`docs/ACQUISITION.md`).
Open access lands free; anything beyond it is their call, in their jurisdiction.
**Whether and how to harvest is the user's decision, not the pack's.**

## Why this matters

It makes the compounding knowledge **travel as a public good** while the copyright
stays home. A clinician grows a corpus on a topic, ships the *map* of it, and the
next clinician rebuilds and extends it. The map is the durable, shareable artifact;
the bytes are re-fetchable. Lists of papers, summaries of what each provides, and a
map of how they fit together — that's the distributable shape of a literature
corpus.

## The map, today

The map is an **embedding-derived** structure: topic clusters (cosine k-means over
per-paper centroids) + a per-paper nearest-neighbour graph, labelled by title
terms. A **citation graph** (OpenAlex references) would be a richer map and is on
the roadmap — it would add directed "builds-on / cited-by" edges to the
similarity edges shipped now.
