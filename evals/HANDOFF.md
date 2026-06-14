# Clinical-reasoning eval — handoff for a larger run

This is the reproducible spec for the **bigger model / bigger hardware** run. The
laptop work below establishes the method and the 14B result; the open question is
whether a larger *local* model closes the clinical-reasoning gap (the
"capability commoditises onto cheap hardware" hypothesis).

## The question
Open grounding makes a small local model safe against **hallucination** on
retrieval questions (grounding ~100%, no fabricated citations). It does **not**
fix **reasoning**. On hard vignettes (multi-comorbidity management, test-ordering
judgment, epidemiological/base-rate reasoning) the 14B model is unsafe. The
question for a bigger box: does model size (or the reasoning lane) close it?

## What we found on the laptop (qwen2.5:14b, M-series, 32 GB)
- Retrieval questions: strong (high grounding, 0 hallucinated citations).
- Hard reasoning vignettes (`evals/clinical-reasoning.json`, n=9): **mean rubric
  coverage 24%** with **14 distinct safety errors** by an independent frontier
  grader — e.g. missed midgut volvulus in a bilious-vomiting neonate, inverted the
  base rate on a positive newborn screen, "aggressive fluids" + mis-staged pH 7.05
  as "mild" in cerebral-oedema-risk DKA.
- The **grounded** harness gave *negative* lift on these (its citation-tuned
  critique/revise strips reasoning the corpus doesn't contain). A **reasoning**
  lane (`--mode reasoning`) was added to test whether scaffolding lifts it.
- **qwen2.5:72b is NOT viable on 32 GB** — it loads at ~61 GB (CPU spill) and a
  single one-sentence call timed out at 600 s. Hence this handoff.

## Run this on a box with ≥64 GB RAM (or a GPU; 72b-q4 ≈ 45-48 GB)
```bash
ollama pull qwen2.5:72b          # and optionally qwen2.5:32b
export LOCALEVIDENCE_PASSAGES=/path/to/passage/store   # see README "Run it fully local"

# The 2x2: {grounded, reasoning} x {one-shot baseline, harness}, per model.
for M in qwen2.5:14b qwen2.5:32b qwen2.5:72b; do
  for MODE in grounded reasoning; do
    python3 -m localevidence eval-local \
      --vignettes evals/clinical-reasoning.json \
      --model ollama:$M --mode $MODE --baseline \
      --out eval-$M-$MODE.json
  done
done
```
`--baseline` runs the one-shot control on the same passages so the harness lift is
isolated. The deterministic grounding metric (coverage, hallucinated citations) is
in each output; the per-question full answers are kept for grading.

## Grade properly — do NOT trust the built-in `--rubric`
The built-in rubric scorer uses the *same local model*, which we found unreliable
(it scored decent answers 0% and over-credited a flawed one). For a real number,
grade each answer against its rubric with an **independent, stronger** grader and
have it also list **safety errors** actually present. The pattern we used: one
grader per (answer, rubric), demanding examiner, returns per-point covered/missed
+ safety_errors + a one-line quality note; aggregate mean coverage and total
safety errors. (A frontier API model, or a panel, is appropriate here.)

## What would be decisive
- **Model-size lift:** 14b vs 32b vs 72b, same vignettes/grader. Does coverage rise
  and do safety errors fall with size? Where does the safety-error count reach ~0?
- **Harness lift:** grounded vs reasoning mode at each size.
- **Expand the set:** 9 vignettes is a probe. Scale to ~50-100 across more
  specialties before drawing strong conclusions; the per-type breakdown
  (management / test-decision / epidemiological) is where the interesting structure is.
- **The safety-relevant claim:** find the smallest local model at which the
  safety-error count on these reasoning cases reaches zero — that is the
  deploy-on-cheap-hardware threshold the whole argument hinges on.
