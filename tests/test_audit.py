from localevidence.audit import audit_entry, render


def test_grounding_flags_citation_not_in_evidence():
    entry = {
        "id": 5, "question": "Croup steroid dose?",
        "answer": ("Dexamethasone 0.15 mg/kg [10.1002/14651858.CD001955]. "
                   "Also adrenaline IM [10.9999/not-retrieved]."),
        "evidence": [
            {"slug": "a", "doi": "10.1002/14651858.CD001955",
             "title": "Glucocorticoids for croup", "tier": "systematic-review"},
            {"slug": "b", "doi": "10.1111/all.15032", "title": "EAACI 2021", "tier": "guideline"},
        ],
        "confidence": "moderate", "gaps": [],
    }
    a = audit_entry(entry)
    c = a["citations"]
    assert "10.1002/14651858.cd001955" in c["grounded"]     # cited AND retrieved
    assert "10.9999/not-retrieved" in c["ungrounded"]        # cited but NOT retrieved -> flag
    assert c["grounding_rate"] == 0.5
    ev = {e["slug"]: e["cited"] for e in a["evidence"]}
    assert ev["a"] is True and ev["b"] is False              # which retrieved sources were used


def test_ceiling_ledger_only_caps_below_reproducible():
    entry = {"id": 1, "question": "q", "answer": "x [10.1/a]",
             "evidence": [{"slug": "a", "doi": "10.1/a"}]}
    a = audit_entry(entry)
    assert a["reproducible"] is False
    assert a["verification_ceiling"]["rung"] == 5            # evidence visible, but not re-runnable


def test_ceiling_no_evidence_no_citations():
    a = audit_entry({"id": 9, "question": "q", "answer": "a plain answer", "evidence": []})
    assert a["verification_ceiling"]["rung"] == 1
    assert a["citations"]["cited"] == []


def test_render_smoke():
    entry = {"id": 2, "question": "q", "answer": "x [10.1/a]",
             "evidence": [{"slug": "a", "doi": "10.1/a", "title": "T", "tier": "rct"}], "gaps": []}
    out = render(audit_entry(entry))
    assert "Audit — ledger #2" in out
    assert "grounding" in out.lower()
    assert "verification ceiling" in out.lower()
