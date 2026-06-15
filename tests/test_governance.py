import pytest

from localevidence import governance as gov
from localevidence import inference
from localevidence import operating_point as op


def _op(**kw):
    base = dict(node="n", base_rate=0.05, cost_fn=10.0, cost_fp=1.0, escalate_threshold=0.9)
    base.update(kw)
    return op.OperatingPoint(**base)


def test_decision_log_records_and_is_attributable(tmp_path):
    log = gov.DecisionLog(tmp_path / "g.jsonl")
    o = _op(node="rural-gp", cost_fn=50.0, cost_fp=1.0)
    rec = gov.govern(o, 0.10, log=log, question="MRI for low-risk headache?",
                     task_class="test-ordering", tier="large", at="2026-06-15T00:00:00")
    assert rec["action"] == "investigate"
    rows = log.records()
    assert len(rows) == 1
    r = rows[0]
    assert r["node"] == "rural-gp" and r["task_class"] == "test-ordering"
    assert r["operating_point"]["cost_fp"] == 1.0 and r["at"] == "2026-06-15T00:00:00"


def test_investigate_rate_per_node(tmp_path):
    log = gov.DecisionLog(tmp_path / "g.jsonl")
    aggressive = _op(node="A", cost_fn=50.0, cost_fp=1.0)    # low threshold -> investigates
    for p in (0.10, 0.20, 0.30):
        gov.govern(aggressive, p, log=log, task_class="test-ordering", tier="large")
    cautious = _op(node="B", cost_fn=1.0, cost_fp=50.0)      # high threshold -> watches
    for p in (0.10, 0.20, 0.30):
        gov.govern(cautious, p, log=log, task_class="test-ordering", tier="large")
    # "wants to MRI everyone" is now a measurable, per-node rate
    assert log.investigate_rate("A") == 1.0
    assert log.investigate_rate("B") == 0.0
    s = log.summary("A")
    assert s["n"] == 3 and s["counts"]["investigate"] == 3


def test_governed_answer_gate_refuses_small_tier(tmp_path):
    log = gov.DecisionLog(tmp_path / "g.jsonl")
    gate_fn = lambda q, model=None, tier=None: {"task_class": "test-ordering", "tier": "small", "allowed": False}
    out = gov.governed_answer("Should I MRI?", _op(), retrieve=lambda q, k: [], model="ollama:14b",
                              log=log, gate_fn=gate_fn, estimate_fn=lambda *a, **k: 0.1)
    assert out["disposition"] == "refused"
    assert log.records() == []          # nothing decided or logged when the gate refuses


def test_governed_answer_decides_and_dial_governs_the_action(tmp_path):
    log = gov.DecisionLog(tmp_path / "g.jsonl")
    gate_fn = lambda q, model=None, tier=None: {"task_class": "test-ordering", "tier": "large", "allowed": True}
    aggressive = _op(node="tert", cost_fn=50.0, cost_fp=1.0)
    out = gov.governed_answer("Should I MRI?", aggressive, retrieve=lambda q, k: [], model="opus",
                              log=log, gate_fn=gate_fn, estimate_fn=lambda *a, **k: 0.10)
    assert out["disposition"] == "decided" and out["decision"]["action"] == "investigate"
    assert len(log.records()) == 1
    # SAME question + SAME estimate, only the dial changes -> the action moves
    cautious = op.OperatingPoint(node="tert", base_rate=0.05, cost_fn=50.0, cost_fp=50.0,
                                 escalate_threshold=0.9)
    out2 = gov.governed_answer("Should I MRI?", cautious, retrieve=lambda q, k: [], model="opus",
                               log=log, gate_fn=gate_fn, estimate_fn=lambda *a, **k: 0.10)
    assert out2["decision"]["action"] == "watch"


def test_estimate_probability_parses_and_falls_back(monkeypatch):
    monkeypatch.setattr(inference, "generate", lambda *a, **k: "The pre-test probability is 0.12.")
    assert gov.estimate_probability("q", retrieve=lambda q, k: [], model="x") == 0.12
    monkeypatch.setattr(inference, "generate", lambda *a, **k: "About 12%")
    assert gov.estimate_probability("q", retrieve=lambda q, k: [], model="x") == 0.12
    def boom(*a, **k):
        raise inference.InferenceError("down")
    monkeypatch.setattr(inference, "generate", boom)
    assert gov.estimate_probability("q", retrieve=lambda q, k: [], model="x") is None


def test_probability_parser_not_hijacked_by_bare_tokens():
    # the must-fix: a leading step number / grade / count must NOT win over the real
    # estimate and clamp to the catastrophic 0.0/1.0.
    cases = {
        "In the top 1 differential, my estimate is 30%": 0.30,
        "Step 1: estimate. Probability = 45%": 0.45,
        "Grade 1 risk; about 25% chance": 0.25,
        "On a scale of 0 to 1, I say 0.7": 0.7,
        "There is a 1 in 5 chance": 0.2,
    }
    for text, expected in cases.items():
        assert gov.parse_probability(text) == pytest.approx(expected), text
    # an uninterpretable bare integer is NOT a probability -> None (caller -> base rate)
    assert gov.parse_probability("1") is None
    assert gov.parse_probability("the top differential") is None


def test_decision_log_does_not_store_raw_question_by_default(tmp_path):
    log = gov.DecisionLog(tmp_path / "g.jsonl")
    q = "62F worst headache of life ?SAH"
    gov.govern(_op(), 0.1, log=log, question=q, task_class="test-ordering")
    r = log.records()[0]
    assert "question" not in r                       # PHI is not persisted by default
    assert "question_sha" in r and r["question_len"] == len(q)
    # opt-in raw storage for a trusted, gitignored local audit
    log2 = gov.DecisionLog(tmp_path / "g2.jsonl")
    gov.govern(_op(), 0.1, log=log2, question=q, keep_question=True)
    assert log2.records()[0]["question"] == q


def test_governed_answer_falls_back_to_base_rate_when_estimate_none(tmp_path):
    log = gov.DecisionLog(tmp_path / "g.jsonl")
    gate_fn = lambda q, model=None, tier=None: {"task_class": "test-ordering", "tier": "large", "allowed": True}
    o = _op(node="x", base_rate=0.20, cost_fn=10.0, cost_fp=1.0)   # base_rate above t* -> investigate
    out = gov.governed_answer("Should I MRI?", o, retrieve=lambda q, k: [], model="opus",
                              log=log, gate_fn=gate_fn, estimate_fn=lambda *a, **k: None)
    assert out["estimated_prob"] is None
    assert out["decision"]["prob"] == 0.20          # fell back to the node base rate
    assert len(log.records()) == 1                  # and still logged
