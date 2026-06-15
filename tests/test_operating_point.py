import json

import pytest

from localevidence import operating_point as op


def _node(**kw):
    base = dict(node="test", base_rate=0.05, cost_fn=10.0, cost_fp=1.0)
    base.update(kw)
    return op.OperatingPoint(**base)


def test_action_threshold_moves_monotonically_with_cost_asymmetry():
    # t* = cost_fp / (cost_fp + cost_fn): raising the cost of a MISS lowers the bar
    # to investigate; raising the cost of over-investigation raises it.
    cheap_miss = op.action_threshold(_node(cost_fn=1.0, cost_fp=10.0))
    dear_miss = op.action_threshold(_node(cost_fn=100.0, cost_fp=1.0))
    assert dear_miss < cheap_miss                       # higher cost_fn -> lower threshold
    assert 0.0 < dear_miss < cheap_miss < 1.0
    # resource friction (e.g. scarce MRI) raises the bar to investigate
    scarce = op.action_threshold(_node(cost_fn=10.0, cost_fp=1.0, resource_friction=5.0))
    ready = op.action_threshold(_node(cost_fn=10.0, cost_fp=1.0, resource_friction=1.0))
    assert scarce > ready


def test_decide_flips_with_the_dial_holding_probability_fixed():
    # THE FALSIFIABLE DEMONSTRATION a black box cannot give: same patient
    # probability, change ONLY the local operating point's cost asymmetry, and the
    # recommendation moves investigate <-> watch, deterministically.
    prob = 0.10
    aggressive = _node(cost_fn=50.0, cost_fp=1.0, escalate_threshold=0.9)   # t* ~= 0.02
    conservative = _node(cost_fn=50.0, cost_fp=50.0, escalate_threshold=0.9)  # t* = 0.5
    assert op.decide(aggressive, prob)["action"] == "investigate"
    assert op.decide(conservative, prob)["action"] == "watch"


def test_decide_flips_with_base_rate_holding_costs_fixed():
    low = _node(base_rate=0.02, cost_fn=10.0, cost_fp=1.0, escalate_threshold=0.9)   # t* ~= 0.09
    high = _node(base_rate=0.20, cost_fn=10.0, cost_fp=1.0, escalate_threshold=0.9)
    # with no patient-specific estimate, the node's base rate is the probability
    assert op.decide(low)["action"] == "watch"
    assert op.decide(high)["action"] == "investigate"


def test_decide_escalates_at_high_probability():
    o = _node(cost_fn=10.0, cost_fp=1.0, escalate_threshold=0.6)
    assert op.decide(o, 0.95)["action"] == "escalate"


def test_decide_record_is_attributable():
    o = _node(node="rural-gp", base_rate=0.03, cost_fn=20.0, cost_fp=2.0)
    d = op.decide(o, 0.10)
    # the decision carries the inputs + the dial, so a failure is traceable to it
    assert d["node"] == "rural-gp" and d["prob"] == 0.10
    assert d["threshold"] == pytest.approx(2.0 / 22.0)
    assert d["operating_point"]["cost_fn"] == 20.0 and d["operating_point"]["cost_fp"] == 2.0


def test_from_dict_and_load_roundtrip(tmp_path):
    spec = {"node": "tertiary", "base_rate": 0.15, "cost_fn": 8.0, "cost_fp": 3.0,
            "resource_friction": 1.0, "escalate_threshold": 0.7, "notes": "ready MRI"}
    o = op.OperatingPoint.from_dict(spec)
    assert o.node == "tertiary" and o.notes == "ready MRI"
    p = tmp_path / "tertiary.json"
    p.write_text(json.dumps(spec))
    loaded = op.load_operating_point(p)
    assert loaded == o


def test_invalid_operating_point_rejected():
    with pytest.raises(ValueError):
        op.OperatingPoint(node="x", base_rate=1.5, cost_fn=1.0, cost_fp=1.0)   # prob > 1
    with pytest.raises(ValueError):
        op.OperatingPoint(node="x", base_rate=0.1, cost_fn=0.0, cost_fp=1.0)   # zero cost
