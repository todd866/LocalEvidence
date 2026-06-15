from localevidence import eval as ev


def _grd(coverage, hall):
    return {"grounding": {"coverage": coverage, "hallucinated_citations": hall}}


def test_summarize():
    s = ev.summarize([_grd(1.0, 0), _grd(0.5, 2), _grd(0.0, 1)])
    assert s["n"] == 3
    assert s["mean_coverage"] == 0.5
    assert s["mean_hallucinated"] == 1.0
    assert s["fully_grounded"] == 1
    assert s["any_hallucination"] == 2


def test_lift_isolates_harness_contribution():
    harness_runs = [_grd(0.9, 0), _grd(1.0, 0)]   # well grounded, no hallucinations
    baseline_runs = [_grd(0.5, 1), _grd(0.6, 2)]  # weaker, hallucinating
    out = ev.lift(harness_runs, baseline_runs)
    assert out["coverage_gain"] == round(0.95 - 0.55, 3)
    assert out["hallucination_reduction"] == round(1.5 - 0.0, 3)


def test_score_rubric_parses_yes_no(monkeypatch):
    from localevidence import inference
    monkeypatch.setattr(inference, "generate",
                        lambda *a, **k: "1: YES\n2: NO\n3: YES")
    out = ev.score_rubric("an answer", ["base rates", "PPV", "confirm"], model="ollama:x")
    assert out["rubric_n"] == 3 and out["rubric_covered"] == 2
    assert out["rubric_coverage"] == round(2 / 3, 3)
    assert out["missed"] == ["PPV"]


def test_score_rubric_empty_is_none():
    assert ev.score_rubric("a", [], model="ollama:x") is None


def test_run_eval_with_vignette_and_rubric(monkeypatch):
    from localevidence import inference
    def fake(prompt, *, system=None, model=None, **kw):
        if "criterion" in prompt.lower():
            return "1: YES\n2: YES"
        return "Plan [s1#0]."
    monkeypatch.setattr(inference, "generate", fake)
    passages = [{"id": "s1#0", "paper": "P", "doi": "10/x", "text": "t"}]
    out = ev.run_eval([{"id": "v1", "type": "management", "question": "Manage?",
                        "rubric": ["point a", "point b"]}],
                      retrieve=lambda q, kk: passages, model="ollama:x", rubric=True)
    assert out["rows"][0]["rubric"]["rubric_coverage"] == 1.0
    assert out["rubric_summary"]["mean_rubric_coverage"] == 1.0


def test_run_eval_empty_retrieval_does_not_crash(monkeypatch):
    # regression for the no-passages KeyError blocker: run_eval over a corpus that
    # retrieves nothing must produce a row, not raise.
    from localevidence import inference
    monkeypatch.setattr(inference, "generate", lambda *a, **k: "unused")
    out = ev.run_eval(["Q?"], retrieve=lambda q, kk: [], model="ollama:x")
    assert out["summary"]["n"] == 1
    assert out["rows"][0]["harness"]["coverage"] == 0.0


def test_run_eval_survives_a_model_failure(monkeypatch):
    # one item's model call blows up (e.g. a timeout); the run must continue and
    # record the failure, not die — critical for long unattended local runs.
    from localevidence import inference, harness
    calls = {"n": 0}
    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise inference.InferenceError("timed out")
        return "Recovered answer [s1#0]."
    monkeypatch.setattr(inference, "generate", flaky)
    monkeypatch.setattr(harness, "expand_queries", lambda q, **k: [])  # keep call count simple
    passages = [{"id": "s1#0", "paper": "P", "doi": "10/x", "text": "t"}]
    out = ev.run_eval(["Q1?", "Q2?"], retrieve=lambda q, kk: passages, model="ollama:x")
    assert len(out["rows"]) == 2
    assert "error" in out["rows"][0] and "timed out" in out["rows"][0]["error"]
    assert out["rows"][1].get("answer", "").endswith("[s1#0].")
    assert out["summary"]["n"] == 1   # only the successful item counts in the summary


def test_run_eval_threads_profile_to_reasoning(monkeypatch):
    # mode='reasoning' must carry the chosen profile into the reasoning lane, so the
    # grounded+decision-profile arm can actually be evaluated.
    from localevidence import harness
    seen = {}
    def fake_reason(q, *, retrieve, model=None, k=8, profile=None, constraints=""):
        seen["profile"] = profile
        return {"answer": "reasoned [s1#0]", "passages": list(retrieve(q, k)),
                "n_passages": 1, "stages": ["frame", "retrieve", "draft", "verify"],
                "grounding": {"coverage": 1.0, "hallucinated_citations": 0,
                              "n_grounded": 1, "n_sentences": 1}}
    monkeypatch.setattr(harness, "reasoning_answer", fake_reason)
    p = [{"id": "s1#0", "paper": "P", "doi": "10/x", "text": "t"}]
    out = ev.run_eval([{"id": "v1", "question": "Interpret this for my patient?"}],
                      retrieve=lambda q, kk: p, model="ollama:x",
                      mode="reasoning", profile="clinical-decision")
    assert seen["profile"] == "clinical-decision"
    assert out["rows"][0]["answer"] == "reasoned [s1#0]"


def test_run_eval_with_mock(monkeypatch):
    from localevidence import harness, inference
    monkeypatch.setattr(inference, "generate",
                        lambda *a, **k: "Grounded claim [s1#0].")
    passages = [{"id": "s1#0", "paper": "P", "doi": "10/x", "text": "t"}]
    out = ev.run_eval(["Q1?", "Q2?"], retrieve=lambda q, kk: passages,
                      model="ollama:x", baseline=True)
    assert len(out["rows"]) == 2
    assert out["summary"]["n"] == 2
    assert "lift" in out
    assert out["rows"][0]["harness"]["valid"] == ["s1#0"]
