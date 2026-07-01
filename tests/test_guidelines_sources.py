"""The guideline-source registry: adding a source is a config entry, and the
generic crawler parses each source's index links (deriving a title when the
link regex doesn't capture one). Network mocked; no fetch, no store."""
from __future__ import annotations

import unittest
from unittest import mock


def _session(html: str):
    resp = mock.Mock()
    resp.text = html
    resp.raise_for_status = lambda: None
    sess = mock.Mock()
    sess.get = mock.Mock(return_value=resp)
    return sess


class GuidelineSourceRegistryTest(unittest.TestCase):
    def test_rch_source_parses_and_skips_non_guidelines(self):
        from localevidence.guidelines import crawl_index, SOURCES
        html = ('<a href="/clinicalguide/guideline_index/Asthma_acute/">Acute asthma</a>'
                '<a href="/clinicalguide/guideline_index/Croup">Croup</a>'
                '<a href="/clinicalguide/guideline_index/Search">Search</a>')
        out = crawl_index(_session(html), SOURCES["rch"])
        names = {n for n, _, _ in out}
        self.assertIn("Asthma_acute", names)
        self.assertIn("Croup", names)
        self.assertNotIn("Search", names)  # in _RCH_SKIP

    def test_aih_source_derives_title_from_name(self):
        from localevidence.guidelines import crawl_index, SOURCES
        html = ('<a href="/contents/vaccine-preventable-diseases/measles">Measles</a>'
                '<a href="/contents/vaccine-preventable-diseases/mumps">whatever</a>')
        out = crawl_index(_session(html), SOURCES["aih"])
        by_name = {n: (t, u) for n, t, u in out}
        self.assertIn("measles", by_name)
        self.assertEqual(by_name["measles"][0], "Measles")  # title derived from name
        self.assertTrue(by_name["measles"][1].startswith(
            "https://immunisationhandbook.health.gov.au/contents/"))

    def test_aih_is_registered_with_its_own_source_tag(self):
        from localevidence.guidelines import SOURCES
        self.assertEqual(SOURCES["aih"]["source"], "guideline:aih")
        self.assertEqual(SOURCES["aih"]["slug_prefix"], "aih-")


if __name__ == "__main__":
    unittest.main()
