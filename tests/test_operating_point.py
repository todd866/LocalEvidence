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


def test_non_finite_fields_rejected():
    # NaN slips a bare `> 0` check and would make the threshold NaN -> silent 'watch'.
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            op.OperatingPoint(node="x", base_rate=0.1, cost_fn=bad, cost_fp=1.0)
        with pytest.raises(ValueError):
            op.OperatingPoint(node="x", base_rate=0.1, cost_fn=1.0, cost_fp=bad)
        with pytest.raises(ValueError):
            op.OperatingPoint(node="x", base_rate=0.1, cost_fn=1.0, cost_fp=1.0, resource_friction=bad)


def test_escalate_never_fires_below_the_investigate_bar():
    # the must-fix: a node may set escalate_threshold < t*, but it must NEVER recommend
    # escalate (treat outright) at a probability where it wouldn't even investigate.
    o = _node(cost_fn=1.0, cost_fp=50.0, escalate_threshold=0.1)   # t* ~= 0.98, esc=0.1
    assert op.action_threshold(o) > 0.9
    assert op.decide(o, 0.20)["action"] == "watch"        # NOT escalate (the old bug)
    assert op.decide(o, 0.99)["action"] == "escalate"
    # normal ordering still holds when escalate_threshold >= t*
    o2 = _node(cost_fn=50.0, cost_fp=1.0, escalate_threshold=0.6)  # t* ~= 0.02
    assert op.decide(o2, 0.10)["action"] == "investigate"
    assert op.decide(o2, 0.70)["action"] == "escalate"


def test_decide_rejects_out_of_range_probability():
    o = _node()
    for bad in (1.5, -0.1, float("nan")):
        with pytest.raises(ValueError):
            op.decide(o, bad)


def test_decide_boundary_inclusivity():
    o = _node(cost_fn=1.0, cost_fp=1.0, escalate_threshold=0.6)    # t* = 0.5
    assert op.decide(o, 0.5)["action"] == "investigate"           # p == t* -> investigate
    assert op.decide(o, 0.4999)["action"] == "watch"
    assert op.decide(o, 0.6)["action"] == "escalate"              # p == escalate -> escalate


def test_action_threshold_strictly_monotone_under_sweeps():
    # back the headline "monotonically" claim with an actual sweep, not 2 spot checks
    prev = None
    for cfn in range(1, 60):
        t = op.action_threshold(_node(cost_fn=float(cfn), cost_fp=1.0))
        if prev is not None:
            assert t < prev          # higher cost_fn -> strictly lower threshold
        prev = t
    prev = None
    for cfp in range(1, 60):
        t = op.action_threshold(_node(cost_fn=10.0, cost_fp=float(cfp)))
        if prev is not None:
            assert t > prev          # higher cost_fp -> strictly higher threshold
        prev = t
    prev = None
    for fr in range(1, 60):
        t = op.action_threshold(_node(cost_fn=10.0, cost_fp=1.0, resource_friction=float(fr)))
        if prev is not None:
            assert t > prev          # higher friction -> strictly higher threshold
        prev = t
