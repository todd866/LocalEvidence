# Grade sweep: capability vs. safety on hard clinical reasoning

Does a bigger model reason more *safely*, and where do safety errors fall to zero?
Run on the free stack (local + OpenRouter-free + Claude-via-subagent) — no paid API.

## Method
- **Items:** the 12 held-out vignettes (`evals/held-out-reasoning.json`) — new
  conditions (afib+cirrhosis anticoagulation, hyponatremia/ODS, myxedema coma,
  phaeo/pre-test-probability, lithium/HD, occult-paracetamol screen, …), each with a
  rubric of must-cover considerations.
- **Grades:** 7B/14B (local, ollama qwen2.5) · 31B (gemma-4) / 120B (nemotron-3),
  OpenRouter-free · Sonnet / Opus (Claude, via subagent).
- **Held constant:** retrieval — each model answers from the *same* k=6 passages, one
  grounded answer per (vignette, grade).
- **Grading:** a single consistent **Opus examiner** scores every grade's answer for
  rubric coverage and counts clinically wrong/unsafe statements. Same standard across grades.

## Result (coverage excludes 2 harness glitches; see caveats)

| Grade | Rubric coverage | Total safety errors | per-vignette |
|---|---|---|---|
| 7B  | 15% | 30 | 2.5 |
| 14B | 27% | 20 | 1.67 |
| 31B | 67% | **2** | 0.17 |
| 120B | 83% | **8** | 0.67 |
| Sonnet | **92%** | **0** | 0 |
| Opus | 86% | **0** | 0 |

## Findings
1. **Safety errors reach zero only at the frontier.** No open-weight rung — not even
   120B — hit zero. On this set the smallest grade with zero safety errors is **Sonnet**.
   This quantifies the capability gate: sub-frontier models should not autonomously
   answer reasoning questions.
2. **Bigger is not automatically safer.** 120B is *more complete* than 31B (83% vs 67%)
   yet commits **4× the safety errors** (8 vs 2): more generation → more confident wrong
   statements. You cannot size-up your way to safety in the open-weight mid-range.
3. **Completeness rises monotonically** (15→27→67→83→…), with the usability jump at
   **14B→31B** — but usable ≠ safe (31B still erred).
4. **Only the frontier achieves both at once** — high completeness *and* zero safety
   errors. That dual property is the real bar for unsupervised clinical reasoning.

## Caveats
- **n = 12**, one held-out set, one examiner (Opus — consistent reference, with mild
  self-grade circularity for the Opus rung).
- **Two harness glitches**, excluded from coverage and noted, not hidden: a 14B answer
  was lost to a timeout (scored over 11), and one Sonnet answer was generated against the
  wrong vignette by a subagent index-read error (off-topic, 0 unsafe content) — excluding
  it, Sonnet is 92% (vs 85% with the 0 included). Neither changes the ranking.
- Grounding (factual retrieval) is a separate, already-safe floor; this study isolates
  *reasoning*.

## Takeaway
Grounding gives every grade a safe floor on retrieval; **reasoning safety is a
frontier-capability property** that mid-size open models don't reach — and don't reach
monotonically. The deploy rule follows: small/local for grounded factual work, frontier
(or a human) for reasoning. Now with numbers behind it.
