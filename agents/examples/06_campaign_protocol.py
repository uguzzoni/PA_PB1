"""The tiered-objective protocol, end to end, on synthetic numbers.

No Julia, no model, no LLM: this simulates a campaign so you can watch the
machinery behave before trusting it with real experiments.

    PYTHONPATH=. python agents/examples/06_campaign_protocol.py

What it demonstrates, in order:

1. Seed replication fixes the noise floor. Nothing smaller counts.
2. Affinity improves for a while, so the campaign keeps optimising it.
3. Affinity saturates -- the leaders' intervals all overlap and the best stops
   moving -- and the campaign switches to enrichment. The gate freezes.
4. The gate then blocks a candidate with a large enrichment gain but
   demonstrably damaged affinity.
5. A merely noisily-worse affinity is NOT blocked: the gate rejects evidence
   of harm, not the absence of evidence of benefit.
6. The hypothesis scorecard shows whether the proposer's predictions beat
   chance.
"""

from __future__ import annotations

import numpy as np

from agents.core import objective as obj
from agents.core.evaluation import EvaluationReport, MetricWithCI

AFF = "affinity.KD.spearman"
ENR = "enrichment.spearman"
CNT = "counts.heldout_loglik"

#: Affinity rests on ~35 measured peptides, so its interval is wide.
#: Enrichment rests on tens of thousands of sequences, so its interval is tight.
#: That asymmetry is the entire reason for the tier structure.
AFF_HALFWIDTH = 0.30
ENR_HALFWIDTH = 0.02


def m(name: str, point: float, half: float) -> MetricWithCI:
    return MetricWithCI(
        name=name, point=point, ci_low=point - half, ci_high=point + half,
        n=35 if name == AFF else 20000, raw_point=point, n_bootstrap=2000,
    )


def rep(
    name: str, aff: float, enr: float, cnt: float = 0.0,
    aff_half: float = AFF_HALFWIDTH, enr_half: float = ENR_HALFWIDTH,
) -> EvaluationReport:
    return EvaluationReport(
        model_name=name,
        metrics={
            AFF: m(AFF, aff, aff_half),
            ENR: m(ENR, enr, enr_half),
            CNT: m(CNT, cnt, 0.01),
        },
    )


def rule(text: str) -> None:
    print(f"\n{'─' * 72}\n{text}\n{'─' * 72}")


def main() -> int:
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    rule("1. Noise floor from seed replication")

    # Five runs of the SAME specification, differing only in seed.
    seeds = [rep(f"baseline-s{i}", aff, enr) for i, (aff, enr) in enumerate(
        zip(0.42 + rng.normal(scale=0.06, size=5), 0.50 + rng.normal(scale=0.008, size=5))
    )]
    replication = obj.summarise_replication("baseline", seeds)
    print(replication.summary())

    floors = obj.noise_floor_from_replication(replication, k=2.0)
    print("\nminimum detectable effect (2 sd):")
    for k, v in floors.items():
        print(f"  {k:36s} {v:.4f}")
    print(
        f"\n-> an affinity gain below {floors[AFF]:.3f} is a rerun, not a result.\n"
        f"   Enrichment resolves {floors[ENR]:.4f}, roughly "
        f"{floors[AFF] / floors[ENR]:.0f}x finer. Hence the tier order."
    )

    # ------------------------------------------------------------------
    rule("2. Declare the objective, once, and lock it")

    objective = obj.ObjectiveSpec(
        name="fitness-improvement",
        tiers=[
            # Affinity is optimised FIRST, while it can still tell models
            # apart, and is demoted to a frozen gate once it saturates.
            obj.Tier(metric=AFF, mode="optimize"),
            obj.Tier(metric=ENR, mode="optimize"),
            obj.Tier(metric=CNT, mode="optimize"),
        ],
        noise_floor=floors,
        eval_budget=40,
        saturation=obj.SaturationRule(top_k=3, patience=3, min_experiments=5),
    )
    campaign = obj.Campaign.declare(objective, overwrite=True)
    print(f"tiers      : {[f'{t.metric} ({t.mode})' for t in objective.tiers]}")
    print(f"active tier: {campaign.active_tier().metric}")
    print(f"budget     : {campaign.budget_remaining()}")

    # ------------------------------------------------------------------
    rule("3. Experiments: affinity improves, then saturates")

    # Affinity climbs out of the noise, then plateaus. Enrichment keeps
    # drifting -- which is exactly the situation the switch exists for.
    trajectory = [
        ("widen-hidden",     0.30, 0.46, "up"),
        ("deeper",           0.38, 0.48, "up"),
        ("lr-decay",         0.44, 0.49, "up"),
        ("longer-training",  0.46, 0.50, "up"),
        ("l2-tune",          0.45, 0.51, "none"),
        ("init-scheme",      0.46, 0.52, "up"),
        ("batch-size",       0.44, 0.53, "none"),
    ]
    for name, aff, enr, direction in trajectory:
        verdict = campaign.record(
            name, rep(name, aff, enr),
            hypothesis=f"{name} should help", predicted_direction=direction,
        )
        switched = campaign.maybe_switch_tier()
        flag = "   <<< TIER SWITCH" if switched else ""
        print(f"  {name:18s} aff={aff:+.2f} enr={enr:+.2f}  -> {verdict.outcome:9s}{flag}")
        if switched:
            print(f"      {campaign.status()['switch_log'][-1]}")

    saturated, reason = campaign.check_saturation(AFF)
    print(f"\naffinity saturated: {saturated}\n  reason: {reason}")

    # ------------------------------------------------------------------
    rule("4. The gate blocks a damaged candidate, however good tier 2 looks")

    frozen = campaign.status()["frozen_gate"].get(AFF)
    print(f"active tier is now  : {campaign.active_tier().metric}")
    print(f"frozen gate (demoted): {AFF} = {frozen:+.3f}  (tolerance {floors[AFF]:.3f})")
    print(
        "\nnote the role reversal: before demotion affinity could only HELP a\n"
        "candidate (as a tie-break); after demotion it can only BLOCK one."
    )

    damaged = campaign.compare(rep("great-enrichment-broken-affinity",
                                   aff=-0.10, enr=0.95, aff_half=0.05))
    print(f"\n  candidate: aff=-0.10 (tight CI), enr=+0.95  -- a huge tier-2 gain")
    print(f"  verdict  : {damaged.outcome}")
    print(f"  reason   : {damaged.reason}")

    # ------------------------------------------------------------------
    rule("5. ...but noise is not evidence of harm")

    noisy = campaign.compare(rep("noisily-lower-affinity", aff=0.30, enr=0.70))
    print(f"  candidate: aff=+0.30 but CI ±{AFF_HALFWIDTH}, enr=+0.70")
    print(f"  verdict  : {noisy.outcome}")
    print(f"  reason   : {noisy.reason}")
    print(
        "\n-> the gate rejects demonstrated damage, not the mere absence of\n"
        "   improvement. With 35 points, rejecting on point estimates would\n"
        "   discard good models at random."
    )

    # ------------------------------------------------------------------
    rule("6. Were the proposer's predictions worth anything?")

    card = campaign.hypothesis_scorecard()
    print(f"  predictions scored: {card['n']}")
    print(f"  hit rate          : {card['hit_rate']}")
    print(
        "\n-> a hit rate near 0.5 over a few dozen experiments means the LLM's\n"
        "   proposals carry no information on this axis, and a deterministic\n"
        "   sweep would do the same job for less. That is worth knowing early."
    )

    rule("Campaign status")
    for k, v in campaign.status().items():
        print(f"  {k:24s} {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
