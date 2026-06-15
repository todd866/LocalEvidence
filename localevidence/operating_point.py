"""The local operating point — the settable, inspectable dial a closed product won't expose.

The companion study's load-bearing result is that safety is a property of the
deployment CONFIG, not the model, and that the config that could be safe is
necessarily LOCAL: the base rate is a property of the node (a rural GP vs a
tertiary clinic), and what's warranted also depends on local costs and resources.
So the dial must be local and inspectable; whether a given setting is safe is on
whoever sets it. This module makes that dial
real — and, crucially, ENFORCED BY CODE rather than coaxed from a model.

The separation of concerns is the whole point:
  - the MODEL (gated to a capable enough tier) estimates the pre-test probability
    of the dangerous condition — that's reasoning, and the capability gate governs it;
  - the ACTION given that probability (investigate / watch / escalate) is a
    DETERMINISTIC function of the local operating point — no model in the loop.

So a small model can't be coaxed past the dial, the dial moves the recommendation
monotonically and reproducibly (unlike free-text priors, which only *steer*), and
every decision is attributable to a specific, version-controlled setting.

The rule is the standard decision threshold (Pauker & Kassirer, NEJM 1975): act
when the probability of the dangerous condition exceeds
    t* = (cost_fp * resource_friction) / (cost_fp * resource_friction + cost_fn),
i.e. raising the cost of a missed diagnosis (cost_fn) lowers the bar to investigate;
raising the cost/scarcity of over-investigation (cost_fp, resource_friction) raises it.

IMPORTANT: the dial decides WHAT TO DO given a probability; it does not make the
probability correct. A badly-set local dial confidently mis-targets — which is why
the value is that the setting is explicit, bounded, and auditable, not that local
is automatically better. See [[localevidence-capability-gate-safety]].
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class OperatingPoint:
    """A per-node deployment dial. Plain data; version it in a JSON file per site."""
    node: str
    base_rate: float            # prior probability of the dangerous condition at this node
    cost_fn: float              # relative harm of a MISS (false negative)
    cost_fp: float              # relative harm/cost of over-investigation (false positive)
    resource_friction: float = 1.0   # >1 = the investigation is scarce/costly here (raises the bar)
    escalate_threshold: float = 0.6   # probability at/above which to escalate/treat outright
    notes: str = ""

    def __post_init__(self):
        if not (math.isfinite(self.base_rate) and 0.0 <= self.base_rate <= 1.0):
            raise ValueError(f"base_rate must be a finite number in [0,1], got {self.base_rate}")
        # cost/friction must be finite AND positive — a NaN slips a bare `> 0` check
        # (every NaN comparison is False) and would yield a NaN threshold that silently
        # routes every case to 'watch'.
        for name in ("cost_fn", "cost_fp", "resource_friction"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a finite positive number, got {v}")
        if not (math.isfinite(self.escalate_threshold) and 0.0 < self.escalate_threshold <= 1.0):
            raise ValueError(
                f"escalate_threshold must be a finite number in (0,1], got {self.escalate_threshold}")

    @classmethod
    def from_dict(cls, d: dict) -> "OperatingPoint":
        fields = ("node", "base_rate", "cost_fn", "cost_fp",
                  "resource_friction", "escalate_threshold", "notes")
        return cls(**{k: d[k] for k in fields if k in d})

    def to_dict(self) -> dict:
        return asdict(self)


def action_threshold(op: OperatingPoint) -> float:
    """The probability threshold above which investigating beats watching, given the
    local costs. t* = (cost_fp * friction) / (cost_fp * friction + cost_fn)."""
    fp = op.cost_fp * op.resource_friction
    return fp / (fp + op.cost_fn)


def decide(op: OperatingPoint, prob: Optional[float] = None) -> dict:
    """Deterministic action for a probability of the dangerous condition, under the
    local operating point. `prob` is the (model-estimated) pre-test probability; if
    omitted, the node's base rate is used. Returns the action plus every input and
    the dial, so the recommendation is fully attributable to a specific setting."""
    p = op.base_rate if prob is None else prob
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"prob must be in [0,1], got {p}")
    t = action_threshold(op)
    # Escalate only AT OR ABOVE the investigate bar. A node may set escalate_threshold
    # below t*, but escalating where you wouldn't even investigate is incoherent — so the
    # effective escalate point is max(t*, escalate_threshold). The ladder is then monotone
    # in p for every configuration: watch -> investigate -> escalate.
    if p >= max(t, op.escalate_threshold):
        action = "escalate"
    elif p >= t:
        action = "investigate"
    else:
        action = "watch"
    return {"node": op.node, "action": action, "prob": p, "threshold": t,
            "escalate_threshold": op.escalate_threshold,
            "operating_point": op.to_dict()}


def load_operating_point(path: Union[str, Path]) -> OperatingPoint:
    """Load a per-node operating point from a JSON file (plain text, inspectable,
    version-controllable — the opposite of a hidden global calibration)."""
    return OperatingPoint.from_dict(json.loads(Path(path).read_text()))
