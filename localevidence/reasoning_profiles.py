"""Named reasoning profiles for the reasoning lane (harness.reasoning_answer).

A profile is the *content* of clinical reasoning — the system instruction, the
framing dimensions the model is forced to address, and the dimensions the safety
self-check audits. The harness loop (frame -> draft -> safety-check -> revise) is
the *mechanism*; the profile is what fills it. Separating them lets the same loop
run different reasoning disciplines without touching the loop.

`clinical-default` reproduces the lane's original inline reasoning exactly, so
nothing changes unless a profile is chosen. `clinical-decision` is the
disease-agnostic decision discipline (base-rate-first, grade-the-action, treatable-
mimic exclusion, a defensible exclusion + escalation threshold, and stability under
a leading/incorrect premise). It was prototyped on one disease (MND) but the
structure is generic — disease-specific priors are supplied per-query through the
lane's `constraints` channel, not baked into the profile.

IMPORTANT: this layer AMPLIFIES capability, it does not supply it. A richer
reasoning profile helps a capable model and makes a weak one confidently wrong, so
it must only be reached by a tier the capability gate permits to reason — never as
a way to push a sub-frontier model into autonomous reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class ReasoningProfile:
    name: str
    system: str                       # system prompt for the reasoned draft / revise
    frame_steps: tuple[str, ...]      # dimensions frame() forces the model to state
    safety_checks: tuple[str, ...]    # dimensions the safety self-check audits


# ── clinical-default: the lane's original behaviour, verbatim ────────────────
# Keep these strings identical to the harness's pre-profile inline text. The
# harness unit tests couple to the "can't-miss" token appearing in BOTH the frame
# and the safety-check prompt — do not drop it.

_DEFAULT = ReasoningProfile(
    name="clinical-default",
    system=(
        "You are a careful clinical reasoner. Answer with explicit reasoning: address the "
        "most dangerous / can't-miss possibility FIRST; when a test is involved, reason "
        "about pre-test probability and what a result actually changes (a positive in a "
        "low-probability setting is often a FALSE positive); adjust for EACH comorbidity "
        "named. Ground factual claims in the provided passages and cite them [slug#n], but "
        "DO include sound clinical reasoning even where the passages are silent — state it "
        "plainly as reasoning. Be specific, safe, and decisive."),
    frame_steps=(
        "The single most dangerous / can't-miss diagnosis or outcome to address first.",
        "The key decision and its main trade-off.",
        "If a test is involved: the rough pre-test probability and what a positive vs "
        "negative result would actually change (weigh false positives when probability is low).",
        "Any comorbidities and how each changes the standard approach.",
    ),
    safety_checks=(
        "most-dangerous / can't-miss diagnosis not addressed",
        "test reasoning that ignores pre-test probability / base rates",
        "comorbidity not accounted for",
        "clinically unsafe or incorrect statement",
    ),
)


# ── clinical-decision: the disease-agnostic decision discipline ──────────────

_DECISION = ReasoningProfile(
    name="clinical-decision",
    system=(
        "You are a careful clinical reasoner. Reason from the BASE RATE first: state the "
        "rough pre-test probability before interpreting any test, and remember a positive "
        "in a low-probability setting is often a FALSE positive. Grade the ACTION, not the "
        "label — decide what to DO (test, treat, watch, escalate) and why, not merely what "
        "to call it. Name the single most discriminating test or examination and how to "
        "interpret its result. Enumerate the treatable MIMICS and, for each, the test that "
        "settles it. State a defensible threshold for EXCLUDING the dangerous possibility "
        "and the point at which you would ESCALATE. Adjust for EACH comorbidity named. Hold "
        "the line under pushback: if the question asserts an incorrect or leading premise, "
        "correct it and reason from the evidence rather than capitulating. Ground factual "
        "claims in the provided passages and cite them [slug#n], but include sound clinical "
        "reasoning even where the passages are silent — state it plainly. Be specific, safe, "
        "and decisive."),
    frame_steps=(
        "The single most dangerous / can't-miss diagnosis or outcome to address first.",
        "The base rate / pre-test probability, and what a positive vs negative result would "
        "actually change (weigh false positives when the prior is low).",
        "The single most discriminating test or examination, and how to interpret its result.",
        "The treatable mimics, and for each the test that settles it.",
        "A defensible threshold for excluding the dangerous possibility, and the trigger to escalate.",
        "Each comorbidity and how it changes the standard approach.",
    ),
    safety_checks=(
        "most-dangerous / can't-miss possibility not addressed",
        "test or result interpretation that ignores the base rate / pre-test probability",
        "a treatable mimic left unexcluded",
        "no defensible exclusion threshold or escalation trigger stated",
        "a comorbidity not accounted for",
        "capitulation to an incorrect or leading premise in the question",
        "clinically unsafe or incorrect statement",
    ),
)


PROFILES: dict[str, ReasoningProfile] = {p.name: p for p in (_DEFAULT, _DECISION)}
DEFAULT_PROFILE = _DEFAULT

# What callers may pass for `profile`: a name, a profile object, or None (default).
ProfileArg = Optional[Union[str, ReasoningProfile]]


def get_profile(profile: Optional[Union[str, ReasoningProfile]] = None) -> ReasoningProfile:
    """Resolve a profile: None -> the default; a ReasoningProfile -> itself; a name
    -> the registered profile. An unknown name raises (a typo must not silently run
    the wrong clinical reasoning)."""
    if profile is None:
        return DEFAULT_PROFILE
    if isinstance(profile, ReasoningProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"unknown reasoning profile {profile!r}; choose one of {sorted(PROFILES)}")
