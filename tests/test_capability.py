from localevidence import capability, inference


def test_tier_for_model(monkeypatch):
    monkeypatch.delenv("LOCALEVIDENCE_MODEL_TIER", raising=False)
    assert capability.tier_for_model("ollama:qwen2.5:14b") == "small"   # unknown -> safe
    assert capability.tier_for_model("ollama:qwen2.5:72b") == "large"
    assert capability.tier_for_model(None) == "small"
    monkeypatch.setenv("LOCALEVIDENCE_MODEL_TIER", "large")
    assert capability.tier_for_model("ollama:qwen2.5:14b") == "large"   # explicit override


def test_classify_prefers_reasoning_on_doubt(monkeypatch):
    monkeypatch.setattr(inference, "generate", lambda *a, **k: "management")
    assert capability.classify_task("plan this", model="ollama:x") == "management"
    monkeypatch.setattr(inference, "generate", lambda *a, **k: "factual-lookup")
    assert capability.classify_task("what dose?", model="ollama:x") == "factual-lookup"
    # unrecognised reply -> fail safe to a reasoning class
    monkeypatch.setattr(inference, "generate", lambda *a, **k: "dunno")
    assert capability.classify_task("?", model="ollama:x") == "management"


def test_classify_fails_safe_when_model_down(monkeypatch):
    def boom(*a, **k):
        raise inference.InferenceError("down")
    monkeypatch.setattr(inference, "generate", boom)
    assert capability.classify_task("what dose?", model="ollama:x") == "management"


def test_gated_answer_allows_factual_for_small(monkeypatch):
    monkeypatch.setattr(inference, "generate", lambda *a, **k: "factual-lookup")
    p = [{"id": "s1#0", "title": "P", "doi": "10/x", "text": "dose is 50 mg/kg"}]
    out = capability.gated_answer("what is the dose?", retrieve=lambda q, k: p,
                                  model="ollama:qwen2.5:14b",
                                  synth_fn=lambda q, ps, model=None: {"answer": "50 mg/kg [s1#0]"})
    assert out["disposition"] == "answered" and out["answer"].startswith("50")
    assert out["task_class"] == "factual-lookup" and out["tier"] == "small"


def test_gated_answer_refuses_reasoning_for_small_but_returns_evidence(monkeypatch):
    monkeypatch.setattr(inference, "generate", lambda *a, **k: "management")
    p = [{"id": "s1#0", "title": "P", "doi": "10/x", "text": "evidence body"}]
    out = capability.gated_answer("manage this DKA with comorbidities", retrieve=lambda q, k: p,
                                  model="ollama:qwen2.5:14b")
    assert out["disposition"] == "refused" and out["answer"] is None
    assert out["passages"] == p and "requires a clinician" in out["refusal"]


def test_gated_answer_routes_allowed_reasoning_through_lane(monkeypatch):
    # large tier + a reasoning class -> must go through the reasoning lane (not the
    # factual one-shot synth), carrying the chosen profile.
    monkeypatch.setattr(inference, "generate", lambda *a, **k: "epidemiological")
    p = [{"id": "s1#0", "title": "P", "doi": "10/x", "text": "body"}]
    seen = {}
    def fake_reason(q, *, retrieve, model=None, k=8, profile=None):
        seen["profile"] = profile
        return {"answer": "reasoned [s1#0]", "passages": list(retrieve(q, k)),
                "n_passages": 1, "grounding": {"coverage": 1.0}}
    out = capability.gated_answer("interpret this result for my patient",
                                  retrieve=lambda q, k: p, model="ollama:qwen2.5:72b",
                                  reason_fn=fake_reason, profile="clinical-decision")
    assert out["disposition"] == "answered" and out["answer"] == "reasoned [s1#0]"
    assert out["task_class"] == "epidemiological" and out["tier"] == "large"
    assert seen["profile"] == "clinical-decision"      # profile threaded to the lane
    assert out["grounding"] == {"coverage": 1.0}        # lane's grounding report passed up


def test_large_tier_may_reason(monkeypatch):
    monkeypatch.setattr(inference, "generate", lambda *a, **k: "management")
    out = capability.gate("manage this", model="ollama:qwen2.5:72b")
    assert out["tier"] == "large" and out["allowed"] is True
