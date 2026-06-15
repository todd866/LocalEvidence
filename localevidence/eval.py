"""Long-form local eval: run the grounded harness over many clinical questions
on-device and measure how much the harness lifts a small model.

The metric that matters here is GROUNDING, computed deterministically (no judge):
- coverage: fraction of answer sentences tied to a real retrieved passage,
- hallucinated_citations: [ids] the model cited that were never retrieved.

A frontier model isn't needed to score this — it's arithmetic over the citation
ids. Run with `--baseline` to also produce the one-shot answer (no expand /
critique / revise) on the same passages, so the harness's contribution is
isolated. Because the model is local and free, run it over hundreds of questions.
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Sequence, Union

from . import harness, inference


def baseline_answer(question: str, passages: Sequence[dict], *,
                    model: Optional[str] = None) -> dict:
    """One-shot answer on the SAME passages — the no-harness control."""
    ans = inference.synthesize_answer(question, passages, model=model)["answer"]
    return {"answer": ans,
            "grounding": harness.verify_citations(ans, [p["id"] for p in passages])}


def score_rubric(answer: str, rubric: Sequence[str], *,
                 model: Optional[str] = None) -> Optional[dict]:
    """Completeness check: did the answer address each rubric point a good answer
    MUST cover? One batched yes/no pass by the (free, local) model. This is the
    metric that matters for reasoning questions, where grounding-coverage is not
    enough. NB: grading the model with the same model is a known limitation — it
    measures whether the considerations are present, not whether they are correct;
    the full answers are kept for human inspection alongside."""
    if not rubric:
        return None
    crit = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(rubric))
    prompt = (f"A clinical answer is given, then numbered criteria. For EACH "
              f"criterion, decide whether the answer adequately addresses it. Reply "
              f"with exactly one line per criterion as 'N: YES' or 'N: NO', nothing "
              f"else.\n\nAnswer:\n{answer}\n\nCriteria:\n{crit}")
    raw = inference.generate(prompt, model=model)
    verdicts: dict[int, bool] = {}
    for line in raw.splitlines():
        m = re.match(r"\s*(\d+)\s*[:.\)]\s*(YES|NO)", line, re.I)
        if m:
            verdicts[int(m.group(1))] = m.group(2).upper().startswith("Y")
    covered = [verdicts.get(i + 1, False) for i in range(len(rubric))]
    return {"rubric_n": len(rubric), "rubric_covered": sum(covered),
            "rubric_coverage": round(sum(covered) / len(rubric), 3),
            "missed": [rubric[i] for i in range(len(rubric)) if not covered[i]]}


def _g(r: dict) -> dict:
    return r["grounding"] if "grounding" in r else r


def summarize(results: Sequence[dict]) -> dict:
    """Aggregate grounding metrics over a set of per-question grounding reports."""
    gs = [_g(r) for r in results]
    n = len(gs) or 1
    return {
        "n": len(gs),
        "mean_coverage": round(sum(g["coverage"] for g in gs) / n, 3),
        # absolute grounded claims — guards against "100% grounded because it said
        # almost nothing": coverage rewards terseness, this rewards substance.
        "mean_grounded_claims": round(sum(g.get("n_grounded", 0) for g in gs) / n, 2),
        "mean_claims": round(sum(g.get("n_sentences", 0) for g in gs) / n, 2),
        "mean_hallucinated": round(sum(g["hallucinated_citations"] for g in gs) / n, 3),
        "fully_grounded": sum(1 for g in gs if g["coverage"] >= 0.99),
        "any_hallucination": sum(1 for g in gs if g["hallucinated_citations"] > 0),
    }


def lift(harness_results: Sequence[dict], baseline_results: Sequence[dict]) -> dict:
    """The harness's contribution: harness summary vs baseline summary."""
    h, b = summarize(harness_results), summarize(baseline_results)
    return {"baseline": b, "harness": h,
            "coverage_gain": round(h["mean_coverage"] - b["mean_coverage"], 3),
            "hallucination_reduction": round(b["mean_hallucinated"] - h["mean_hallucinated"], 3)}


def run_eval(items: Sequence[Union[str, dict]], *,
             retrieve: Callable[[str, int], list[dict]],
             model: Optional[str] = None, k: int = 8, baseline: bool = False,
             rubric: bool = False, mode: str = "grounded", profile=None,
             on_result: Optional[Callable[[int, dict], None]] = None) -> dict:
    """Run the harness over `items` (plain question strings, or vignette dicts with
    `question`/`rubric`/`type`/`id`). mode='grounded' (citation-optimised, for
    retrieval) or 'reasoning' (scaffolded, for management/judgment/epi). `profile`
    selects the reasoning discipline (reasoning/safe modes; e.g. 'clinical-decision'
    to evaluate the grounded+decision-profile arm). Optionally also the one-shot
    baseline and rubric-completeness scoring."""
    if mode == "safe":
        from . import safety
        def answer_fn(q, *, retrieve, model=None, k=8):
            def _reason(qq, **kw):
                return harness.reasoning_answer(qq, profile=profile, **kw)
            return safety.guarded_answer(q, retrieve=retrieve,
                                         answer_fn=_reason, model=model, k=k)
    elif mode == "reasoning":
        def answer_fn(q, *, retrieve, model=None, k=8):
            return harness.reasoning_answer(q, retrieve=retrieve, model=model, k=k,
                                            profile=profile)
    else:
        answer_fn = harness.grounded_answer
    h_results, b_results, h_paired, rows = [], [], [], []
    _ZERO = {"coverage": 0.0, "hallucinated_citations": 0, "n_grounded": 0, "n_sentences": 0}
    for i, item in enumerate(items):
        q = item["question"] if isinstance(item, dict) else item
        rub = item.get("rubric") if isinstance(item, dict) else None
        iid = item.get("id") if isinstance(item, dict) else None
        itype = item.get("type") if isinstance(item, dict) else None
        try:
            # Per-item isolation: a single model timeout/error must not kill a long
            # unattended local run — record it and carry on.
            h = answer_fn(q, retrieve=retrieve, model=model, k=k)
            row = {"id": iid, "type": itype, "question": q, "answer": h["answer"],
                   "stages": h["stages"], "harness": h["grounding"]}
            if "disposition" in h:  # safe mode: record the guardrail outcome
                row["disposition"] = h["disposition"]
                row["rules_applied"] = h.get("rules_applied")
                row["violations"] = h.get("violations")
            h_results.append(h)
            if baseline and h["passages"]:
                b = baseline_answer(q, h["passages"], model=model)
                row["baseline"] = b["grounding"]
                b_results.append(b)
                h_paired.append(h)  # pair: lift compares the SAME items, not all-N vs M
            if rubric and rub:
                row["rubric"] = score_rubric(h["answer"], rub, model=model)
        except Exception as e:  # noqa: BLE001 — resilience is the point
            row = {"id": iid, "type": itype, "question": q, "error": str(e),
                   "harness": dict(_ZERO)}
        rows.append(row)
        if on_result:
            on_result(i, row)
    out = {"rows": rows, "model": model or inference.DEFAULT_MODEL,
           "summary": summarize(h_results)}
    if baseline and b_results:
        out["lift"] = lift(h_paired, b_results)  # paired subset only — apples to apples
    rub_rows = [r["rubric"] for r in rows if r.get("rubric")]
    if rub_rows:
        n = len(rub_rows)
        out["rubric_summary"] = {
            "n": n,
            "mean_rubric_coverage": round(sum(x["rubric_coverage"] for x in rub_rows) / n, 3),
            "fully_covered": sum(1 for x in rub_rows if x["rubric_coverage"] >= 0.99),
        }
    return out
