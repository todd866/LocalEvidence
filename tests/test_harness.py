from localevidence import harness, inference


def _passages():
    return [{"id": "s1#0", "paper": "Paper One", "doi": "10.1/x",
             "text": "ceftriaxone 50 mg/kg is appropriate empiric therapy"}]


def test_verify_citations_valid_invalid_and_coverage():
    ans = "Ceftriaxone is recommended [s1#0]. It must be given early [s9#9]. No cite here."
    r = harness.verify_citations(ans, ["s1#0", "s2#3"])
    assert r["valid"] == ["s1#0"]
    assert r["invalid"] == ["s9#9"]          # cited but not retrieved -> flagged
    # only sentence 1 has a VALID citation; sentence 2's [s9#9] is hallucinated
    assert r["n_sentences"] == 3 and r["n_grounded"] == 1
    assert 0.0 < r["coverage"] < 1.0
    assert r["hallucinated_citations"] == 1


def test_expand_queries_parses_lines(monkeypatch):
    monkeypatch.setattr(inference, "generate",
                        lambda *a, **k: "1. ceftriaxone dose\n- meningitis timing\n\n")
    qs = harness.expand_queries("Q?", model="ollama:x", n=3)
    assert qs == ["ceftriaxone dose", "meningitis timing"]


def test_expand_queries_degrades_when_model_unavailable(monkeypatch):
    def boom(*a, **k):
        raise inference.InferenceError("no model")
    monkeypatch.setattr(inference, "generate", boom)
    assert harness.expand_queries("Q?", model="") == []


def _fake_generate_factory():
    def fake(prompt, *, system=None, model=None, **kw):
        if "search queries" in prompt:
            return "ceftriaxone dose\nmeningitis timing"
        if "Critique the draft" in prompt:
            return "- missing the timing caveat the passages support"
        if "Rewrite the answer" in prompt:
            return "Ceftriaxone 50 mg/kg, given early, is recommended [s1#0]."
        return "Ceftriaxone 50 mg/kg is recommended [s1#0]."   # draft (synthesize)
    return fake


def test_grounded_answer_runs_full_loop(monkeypatch):
    monkeypatch.setattr(inference, "generate", _fake_generate_factory())
    out = harness.grounded_answer(
        "Empiric therapy for paediatric bacterial meningitis?",
        retrieve=lambda q, k: _passages(), model="ollama:qwen2.5:14b")
    assert "given early" in out["answer"]                       # revised, not the draft
    assert out["stages"] == ["retrieve", "expand", "draft", "critique", "revise", "verify"]
    assert out["grounding"]["valid"] == ["s1#0"]
    assert out["grounding"]["coverage"] > 0 and out["grounded"] is True
    assert out["draft"] != out["answer"]


def test_grounded_answer_skips_revise_when_critique_ok(monkeypatch):
    def fake(prompt, *, system=None, model=None, **kw):
        if "search queries" in prompt:
            return ""
        if "Critique the draft" in prompt:
            return "OK"
        return "Grounded draft [s1#0]."
    monkeypatch.setattr(inference, "generate", fake)
    out = harness.grounded_answer("Q?", retrieve=lambda q, k: _passages(),
                                  model="ollama:x", expand=False)
    assert "revise" not in out["stages"]
    assert out["answer"] == out["draft"]


def test_reasoning_answer_runs_scaffold(monkeypatch):
    def fake(prompt, *, system=None, model=None, **kw):
        if "can't-miss" in prompt or "can’t-miss" in prompt:   # frame OR safety-check
            if "audit" in prompt.lower():
                return "- did not weigh pre-test probability"        # safety-check: not OK
            return "Frame: worst-case is volvulus; weigh base rates."  # frame
        if "Rewrite the answer" in prompt:
            return "Exclude volvulus first; positive is likely false at low prevalence [s1#0]."
        return "Initial reasoned draft [s1#0]."                       # reasoned draft
    monkeypatch.setattr(inference, "generate", fake)
    out = harness.reasoning_answer("Pros and cons of test X?",
                                   retrieve=lambda q, k: _passages(), model="ollama:x")
    assert out["stages"] == ["frame", "retrieve", "draft", "safety-check", "revise", "verify"]
    assert "likely false" in out["answer"]                            # the revised, reasoned answer
    assert out["grounding"]["valid"] == ["s1#0"]                      # factual claim still grounded
    assert out["frame"]


def test_frame_uses_decision_profile_dimensions(monkeypatch):
    cap = {}
    def fake(prompt, **k):
        cap["p"] = prompt
        return "framed"
    monkeypatch.setattr(inference, "generate", fake)
    harness.frame("Q?", model="ollama:x", profile="clinical-decision")
    low = cap["p"].lower()
    # the decision profile forces the dimensions core lacked
    assert "mimic" in low and "escalate" in low and "discriminating" in low


def test_reasoning_answer_threads_decision_profile(monkeypatch):
    seen = []
    def fake(prompt, *, system=None, model=None, **kw):
        seen.append((prompt, system))
        if "audit" in prompt.lower():
            return "OK"                      # safety-check passes -> no revise
        return "Reasoned answer [s1#0]."
    monkeypatch.setattr(inference, "generate", fake)
    out = harness.reasoning_answer("Pros and cons of test X?",
                                   retrieve=lambda q, k: _passages(),
                                   model="ollama:x", profile="clinical-decision")
    # the decision profile's system prompt reached the reasoned draft
    systems = [s for _, s in seen if s]
    assert any("mimic" in s.lower() for s in systems)
    # default profile still works (regression): no profile -> original behaviour
    assert out["stages"] == ["frame", "retrieve", "draft", "safety-check", "verify"]


def test_grounded_answer_no_passages_is_honest(monkeypatch):
    monkeypatch.setattr(inference, "generate", lambda *a, **k: "should not be called")
    out = harness.grounded_answer("Q?", retrieve=lambda q, k: [], model="ollama:x")
    assert out["grounded"] is False and out["passages"] == []
    # regression: the no-passages branch MUST carry the same keys as the normal
    # return, or callers (run_eval, synthesize --harness) KeyError on an empty corpus.
    for key in ("question", "grounding", "n_passages", "stages", "model"):
        assert key in out, f"no-passages return missing {key!r}"
    assert out["grounding"]["coverage"] == 0.0 and out["n_passages"] == 0
