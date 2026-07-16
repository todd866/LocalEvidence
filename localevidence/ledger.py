"""The knowledge ledger — queries + evidence + answers + the reasoning behind them.

This is the layer OpenEvidence cannot have: a record of what *you* asked, what
evidence answered it, and the worked reasoning — so a similar future question
reuses or refreshes the result instead of re-deriving it. Mirrors the research
program's `don't re-derive` ledgers (exptheory/, expmath/, MEMORY.md).

Storage (all under `ledger/`):
  answers.jsonl       one JSON object per worked question
  questions.npy       L2-normalized MiniLM vectors, row-aligned to question_ids
  question_ids.json   the entry ids, in vector-row order

The pipeline records the question + evidence + coverage at `ask` time; the
answer + reasoning + grounding are filled at synthesis time via `update`.
`find_similar` powers reuse/refresh.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import config, embedding


class Ledger:
    def __init__(self, store_dir: Optional[Path] = None):
        self.dir = Path(store_dir or (config.ROOT / "ledger"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.answers_path = self.dir / "answers.jsonl"
        self.vec_path = self.dir / "questions.npy"
        self.ids_path = self.dir / "question_ids.json"
        self.entries: list[dict] = self._load_entries()
        self.q_ids: list[int] = (json.loads(self.ids_path.read_text())
                                 if self.ids_path.exists() else [])
        self.q_vecs = (np.load(self.vec_path) if self.vec_path.exists()
                       else np.zeros((0, 384), dtype="float32"))

    def _load_entries(self) -> list[dict]:
        if not self.answers_path.exists():
            return []
        out = []
        for line in self.answers_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def _rewrite(self) -> None:
        with self.answers_path.open("w") as fh:
            for e in self.entries:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    def _next_id(self) -> int:
        return (max((e["id"] for e in self.entries), default=0) + 1)

    def get(self, entry_id: int) -> Optional[dict]:
        return next((e for e in self.entries if e["id"] == entry_id), None)

    def record(self, question: str, *, project: str = "", run_id: str = "",
               evidence: Optional[list[dict]] = None, coverage: Optional[dict] = None,
               gaps: Optional[list] = None, supersedes: Optional[int] = None) -> int:
        """Append the question + evidence half of an entry; returns its id.

        `evidence` is a list of
        {slug, doi, title, tier, passage_ids, snippet, score} (see
        `pipeline.evidence_from_passages`); `snippet`/`score` are what the
        EvidenceViewer loader renders per passage.
        Answer/reasoning/grounding are filled later via `update`.
        """
        entry_id = self._next_id()
        entry = {
            "id": entry_id,
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "question": question.strip(),
            "project": project,
            "run_id": run_id,
            "evidence": evidence or [],
            "coverage": coverage or {},
            "gaps": gaps or [],
            "answer": None,
            "reasoning": None,
            "grounding": None,
            "confidence": None,
            "supersedes": supersedes,
        }
        self.entries.append(entry)
        with self.answers_path.open("a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # question-embedding index
        qv = embedding.embed([question])[0].astype("float32")
        self.q_vecs = (np.vstack([self.q_vecs, qv[None, :]])
                       if self.q_vecs.shape[0] else qv[None, :])
        self.q_ids.append(entry_id)
        np.save(self.vec_path, self.q_vecs)
        self.ids_path.write_text(json.dumps(self.q_ids))
        return entry_id

    def update(self, entry_id: int, *, answer: Optional[str] = None,
               reasoning: Optional[str] = None, grounding: Optional[dict] = None,
               confidence: Optional[str] = None) -> bool:
        """Fill the answer/reasoning half of an entry (at synthesis time)."""
        e = self.get(entry_id)
        if not e:
            return False
        if answer is not None:
            e["answer"] = answer
        if reasoning is not None:
            e["reasoning"] = reasoning
        if grounding is not None:
            e["grounding"] = grounding
        if confidence is not None:
            e["confidence"] = confidence
        e["updated_ts"] = dt.datetime.now().isoformat(timespec="seconds")
        self._rewrite()
        return True

    def find_similar(self, question: str, *, threshold: float = 0.80,
                     top: int = 3) -> list[tuple[dict, float]]:
        """Prior worked questions semantically close to this one (reuse/refresh)."""
        if self.q_vecs.shape[0] == 0:
            return []
        qv = embedding.embed([question])[0]
        sims = self.q_vecs @ qv
        order = np.argsort(-sims)[:top]
        out = []
        for i in order:
            s = float(sims[i])
            if s < threshold:
                break
            e = self.get(self.q_ids[i])
            if e and e["question"].strip().lower() != question.strip().lower():
                out.append((e, s))
        return out

    def stats(self) -> dict:
        answered = sum(1 for e in self.entries if e.get("answer"))
        return {"questions": len(self.entries), "answered": answered}
