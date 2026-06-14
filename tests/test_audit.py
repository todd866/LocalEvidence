from localevidence.audit import audit_entry, render


def _entry(answer, evidence, **kw):
    return {"id": 1, "question": "q", "answer": answer, "evidence": evidence,
            "gaps": [], **kw}


def test_doi_provenance_flags_unretrieved():
    a = audit_entry(_entry(
        "Dexamethasone 0.15 mg/kg [10.1002/14651858.CD001955]. "
        "Adrenaline IM [10.9999/not-retrieved].",
        [{"slug": "x", "doi": "10.1002/14651858.CD001955", "title": "Croup SR", "tier": "systematic-review"}]))
    p = a["provenance"]
    assert "10.1002/14651858.cd001955" in p["doi_matched"]      # cited + retrieved
    assert "10.9999/not-retrieved" in p["doi_unmatched"]         # cited, NOT retrieved -> flag
    assert p["match_rate"] == 0.5


def test_embedded_paren_doi_is_not_falsely_flagged():
    # the HIGH bug: Lancet/Elsevier S-series DOIs contain parens and were truncated
    doi = "10.1016/S0140-6736(16)31678-6"
    ev = [{"slug": "a", "doi": doi, "title": "A Lancet paper", "tier": "rct"}]
    for form in (f"stewardship [{doi}].", f"see ({doi}).", f"per {doi} and others"):
        p = audit_entry(_entry(form, ev))["provenance"]
        assert p["doi_matched"] == ["10.1016/s0140-6736(16)31678-6"], form
        assert p["doi_unmatched"] == [], form


def test_named_citation_matched_to_retrieved_title():
    # the other HIGH bug: name-style citations (the documented house style) must
    # be checkable, not silently no-op
    ev = [{"slug": "e", "doi": "10.1111/all.15032",
           "title": "EAACI guidelines: Anaphylaxis (2021 update)", "tier": "guideline"}]
    p = audit_entry(_entry("IM adrenaline 0.01 mg/kg [EAACI 2021].", ev))["provenance"]
    assert any("EAACI" in n for n in p["name_matched"])
    assert p["name_unmatched"] == []


def test_named_citation_not_in_retrieval_is_flagged():
    ev = [{"slug": "e", "doi": "10.1/x", "title": "A paper on bronchiolitis", "tier": "rct"}]
    p = audit_entry(_entry("Give antibiotics [Smith et al 2019].", ev))["provenance"]
    assert any("Smith" in n for n in p["name_unmatched"])


def test_no_citations_is_reported_honestly():
    a = audit_entry(_entry("A plain answer with no citations at all.", []))
    assert a["provenance"]["checkable"] == 0
    out = render(a).lower()
    assert "no parseable citations" in out


def test_ceiling_ledger_only_caps_below_reproducible():
    a = audit_entry(_entry("x [10.1/a]", [{"slug": "a", "doi": "10.1/a"}]))
    assert a["reproducible"] is False
    assert a["verification_ceiling"]["rung"] == 5


def test_ceiling_empty_answer_is_zero():
    a = audit_entry(_entry("", []))
    assert a["verification_ceiling"]["rung"] == 0


def test_render_smoke():
    out = render(audit_entry(_entry(
        "x [10.1/a]", [{"slug": "a", "doi": "10.1/a", "title": "T", "tier": "rct"}])))
    assert "Audit — ledger #1" in out
    assert "provenance" in out.lower()
    assert "verification ceiling" in out.lower()
