"""Evidence records written to the ledger must carry the snippet + score the
EvidenceViewer loader reads, so the LocalEvidence -> EvidenceViewer pipeline
renders real passage text rather than empty passages.

Regression guard for the seam bug where the ledger stored only
{slug, doi, title, tier, passage_ids} and every rendered EvidencePassage.text
came out empty.
"""
from __future__ import annotations

import unittest


class EvidenceRecordsTest(unittest.TestCase):
    def _passages(self):
        from localevidence.index import Passage
        return [
            Passage(passage_id=1, slug="paper-a", text="A weaker supporting chunk.",
                    title="Paper A", doi="10.1/a", tier="rct", score=0.2),
            Passage(passage_id=2, slug="paper-a", text="The pivotal finding sentence.",
                    title="Paper A", doi="10.1/a", tier="rct", score=0.9),
            Passage(passage_id=3, slug="paper-b", text="Second paper evidence.",
                    title="Paper B", doi="10.1/b", tier="observational", score=0.5),
        ]

    def test_records_preserve_the_existing_contract(self):
        from localevidence import pipeline
        by_slug = {r["slug"]: r for r in pipeline.evidence_from_passages(self._passages())}
        self.assertEqual(sorted(by_slug), ["paper-a", "paper-b"])
        self.assertEqual(by_slug["paper-a"]["passage_ids"], [1, 2])
        self.assertEqual(by_slug["paper-a"]["doi"], "10.1/a")
        self.assertEqual(by_slug["paper-a"]["tier"], "rct")

    def test_records_carry_a_nonempty_snippet_from_the_best_passage(self):
        from localevidence import pipeline
        by_slug = {r["slug"]: r for r in pipeline.evidence_from_passages(self._passages())}
        a = by_slug["paper-a"]
        self.assertTrue(a.get("snippet"), "evidence record must carry a non-empty snippet")
        # highest-scoring passage for the slug, not merely the first seen
        self.assertEqual(a["snippet"], "The pivotal finding sentence.")

    def test_records_carry_the_max_passage_score(self):
        from localevidence import pipeline
        by_slug = {r["slug"]: r for r in pipeline.evidence_from_passages(self._passages())}
        self.assertAlmostEqual(by_slug["paper-a"]["score"], 0.9)
        self.assertAlmostEqual(by_slug["paper-b"]["score"], 0.5)

    def test_long_snippet_is_whitespace_normalized_and_truncated(self):
        from localevidence.index import Passage
        from localevidence import pipeline
        # Real whitespace runs (double spaces, tab, newline) so the assertions
        # below actually exercise the ' '.join(text.split()) normalization.
        long_text = "alpha  beta\t gamma\n   delta " * 100  # >600 chars, with runs
        rec = pipeline.evidence_from_passages(
            [Passage(passage_id=9, slug="p", text=long_text, score=1.0)]
        )[0]
        self.assertLessEqual(len(rec["snippet"]), 602)
        self.assertNotIn("  ", rec["snippet"])   # double spaces collapsed
        self.assertNotIn("\t", rec["snippet"])   # tabs collapsed
        self.assertNotIn("\n", rec["snippet"])   # newlines collapsed
        self.assertTrue(rec["snippet"].endswith("…"))

    def test_tie_on_score_keeps_the_first_seen_passage(self):
        from localevidence.index import Passage
        from localevidence import pipeline
        recs = pipeline.evidence_from_passages([
            Passage(passage_id=1, slug="p", text="first", score=0.5),
            Passage(passage_id=2, slug="p", text="second", score=0.5),
        ])
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["snippet"], "first")  # strict '>' keeps first-seen

    def test_empty_passages_returns_empty_list(self):
        from localevidence import pipeline
        self.assertEqual(pipeline.evidence_from_passages([]), [])

    def test_single_zero_score_passage_still_gets_a_snippet(self):
        from localevidence.index import Passage
        from localevidence import pipeline
        rec = pipeline.evidence_from_passages(
            [Passage(passage_id=1, slug="p", text="only chunk", score=0.0)]
        )[0]
        self.assertEqual(rec["snippet"], "only chunk")
        self.assertEqual(rec["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
