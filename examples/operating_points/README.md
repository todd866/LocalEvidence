# Local operating points

A closed clinical-AI product bakes in one calibration and applies it everywhere;
its clearest symptom is over-investigation (recommending imaging far more widely
than a given setting's prevalence and resources warrant), with no way to dial it
back. But the operating point that *could* be safe is necessarily **local**: it
depends on the base rate and the costs of the specific deployment node, so the
dial must be local and inspectable. Whether any given local setting is actually
safe is on whoever sets it.

Each file here is one node's operating point — plain text, version-controllable,
inspectable. It sets:

| field | meaning |
|---|---|
| `base_rate` | prior probability of the dangerous condition in this node's population |
| `cost_fn` | relative harm of a **miss** (false negative) |
| `cost_fp` | relative harm/cost of **over-investigation** (false positive) |
| `resource_friction` | >1 if the investigation is scarce/costly here (raises the bar) |
| `escalate_threshold` | probability at/above which to escalate/treat outright |

The action is decided **by code**, not coaxed from a model — the standard decision
threshold (Pauker & Kassirer, NEJM 1975):

```
investigate when  prob >= cost_fp * friction / (cost_fp * friction + cost_fn)
```

So the dial moves the recommendation **monotonically and reproducibly**. See it:

```bash
# same patient probability, two nodes -> different action (the dial steers it;
# whether that action is RIGHT depends on whether the dial is set right)
python3 -m localevidence operating-point \
    --config examples/operating_points/rural-gp.json \
    --config examples/operating_points/tertiary-ed.json \
    --prob 0.08
# rural-gp -> WATCH (high local bar); tertiary-ed -> INVESTIGATE (low local bar)
```

The model's job — estimating the probability — stays behind the capability gate.
The dial only governs the action *given* a probability. A badly-set local dial
confidently mis-targets; the value is that the setting is **explicit, bounded, and
auditable**, not that local is automatically better.
