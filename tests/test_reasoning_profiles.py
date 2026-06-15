import pytest

from localevidence import reasoning_profiles as rp


def test_default_profile_is_behaviour_preserving():
    p = rp.get_profile(None)
    assert p.name == "clinical-default"
    # The harness tests couple to these tokens (frame vs safety-check disambiguation);
    # the default profile MUST keep them so existing behaviour is unchanged.
    assert "can't-miss" in p.frame_steps[0]
    assert "can't-miss" in p.safety_checks[0]
    # base-rate / pre-test reasoning is the spine of the existing reasoning lane
    assert any("pre-test" in s.lower() or "base rate" in s.lower() for s in p.frame_steps)


def test_decision_profile_adds_the_missing_discipline():
    p = rp.get_profile("clinical-decision")
    assert p.name == "clinical-decision"
    blob = (p.system + " " + " ".join(p.frame_steps) + " " + " ".join(p.safety_checks)).lower()
    # the four pieces core lacked, lifted disease-agnostically from the MND prototype
    assert "mimic" in blob                 # treatable-mimic exclusion
    assert "exclu" in blob                 # defensible exclusion threshold
    assert "escalat" in blob               # escalation trigger
    assert "premise" in blob or "pushback" in blob  # stability under pushback
    # and it still keeps the epi spine
    assert "base rate" in blob or "pre-test" in blob


def test_get_profile_passthrough_and_unknown():
    custom = rp.ReasoningProfile(name="x", system="s", frame_steps=("a",), safety_checks=("b",))
    assert rp.get_profile(custom) is custom            # a profile object passes through
    with pytest.raises(ValueError) as e:
        rp.get_profile("nope")
    assert "clinical-decision" in str(e.value)         # error lists valid names


def test_registry_lists_both_builtins():
    assert set(rp.PROFILES) >= {"clinical-default", "clinical-decision"}
