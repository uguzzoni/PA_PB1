"""The tiered objective, the saturation rule, and the verdict machinery.

This module is the protocol. It exists so that "is this model better?" is a
*computed* answer rather than a narrated one, and so that the rules cannot be
renegotiated halfway through a campaign by whatever is proposing experiments.

The structure
-------------
A campaign declares an ordered list of tiers, once, before any experiment::

    tier 1  GATE      affinity.KD.spearman          must not be provably worse
    tier 2  OPTIMIZE  enrichment.weighted_pearson   discriminates the tier-1 ties
    tier 3  OPTIMIZE  counts.heldout_loglik         discriminates the tier-2 ties

Why a gate rather than an objective for affinity: it is the externally valid
measure but rests on a few dozen points, so its interval is wide and many
models tie within it. Optimising directly against it means selecting
configurations on noise. Demoting it to a gate moves the selection pressure
onto a measure with real statistics while still refusing anything that
demonstrably damages binding prediction.

Three safeguards
----------------
1. **The switch is a rule, not a judgement.** Saturation is detected from
   measured interval overlap and a patience counter, never asserted by an
   agent. An agent free to declare saturation will declare it whenever it is
   losing.
2. **The switch is irreversible.** Once the campaign moves to tier 2 it does
   not move back when tier 2 stops cooperating. Reversibility is cherry
   picking with extra steps, so the state is persisted.
3. **The gate threshold freezes at the moment of the switch.** Otherwise it
   drifts upward as sampling continues, which is its own selection effect.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from . import workspace as ws
from .evaluation import EvaluationReport, MetricWithCI

__all__ = [
    "Tier",
    "SaturationRule",
    "ObjectiveSpec",
    "Verdict",
    "SeedSpread",
    "ReplicationReport",
    "summarise_replication",
    "noise_floor_from_replication",
    "Campaign",
]


# ==========================================================================
# Objective
# ==========================================================================
class Tier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(..., description="Metric key as it appears in EvaluationReport.metrics.")
    mode: Literal["gate", "optimize"] = Field(
        "optimize",
        description=(
            "The tier's INITIAL role. 'optimize' tiers are used in order: each "
            "is optimised until it saturates, then DEMOTED to a frozen gate "
            "and the next one takes over. 'gate' tiers are gates from the "
            "start -- use that only for a guard you never intend to optimise."
        ),
    )
    tolerance: float | None = Field(
        None,
        description=(
            "How far below the frozen reference a candidate may fall before it "
            "is rejected, once this tier is acting as a gate. Defaults to the "
            "metric's noise floor."
        ),
    )


class SaturationRule(BaseModel):
    """When the campaign is allowed to stop optimising a tier and move down."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(5, ge=2, description="How many leaders must all tie.")
    patience: int = Field(
        5, ge=1, description="Experiments without a real improvement before declaring saturation."
    )
    min_experiments: int = Field(
        8, ge=2, description="Never declare saturation before this many experiments."
    )


class ObjectiveSpec(BaseModel):
    """Declared once, at the start of a campaign, and then locked."""

    model_config = ConfigDict(extra="forbid")

    name: str
    tiers: list[Tier] = Field(..., min_length=1)
    noise_floor: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-metric minimum detectable effect, from seed replication. "
            "Differences smaller than this are not differences."
        ),
    )
    eval_budget: int = Field(
        50, ge=1, description="Maximum experiments scored against this objective."
    )
    saturation: SaturationRule = Field(default_factory=SaturationRule)

    def floor(self, metric: str) -> float:
        """Minimum detectable effect for a metric; 0 when not yet measured."""
        return float(self.noise_floor.get(metric, 0.0))

    def declared_gates(self) -> list[Tier]:
        """Tiers that are gates from the start, never optimised."""
        return [t for t in self.tiers if t.mode == "gate"]

    def optimize_tiers(self) -> list[Tier]:
        """Tiers in cascade order. Each is optimised, then demoted to a gate."""
        return [t for t in self.tiers if t.mode == "optimize"]


class Verdict(BaseModel):
    """The computed answer to 'is this better?'."""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["improved", "tied", "worse", "gate_failed", "budget_exhausted"]
    deciding_metric: str | None = None
    reason: str
    details: dict[str, str] = Field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.outcome == "improved"


# ==========================================================================
# Replication and the noise floor
# ==========================================================================
class SeedSpread(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    mean: float
    sd: float
    min: float
    max: float
    n_seeds: int

    @property
    def spread(self) -> float:
        return self.max - self.min


class ReplicationReport(BaseModel):
    """The same specification trained several times, differing only in seed.

    This is the first thing to run in a campaign and the thing that makes
    every later comparison interpretable. Until the seed-to-seed spread is
    known, an apparent improvement cannot be distinguished from a rerun.
    """

    model_config = ConfigDict(extra="forbid")

    spec_name: str
    n_seeds: int
    per_metric: dict[str, SeedSpread] = Field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{self.spec_name}: {self.n_seeds} seeds"]
        for s in self.per_metric.values():
            lines.append(
                f"  {s.metric:36s} mean={s.mean:+.4f} sd={s.sd:.4f} spread={s.spread:.4f}"
            )
        return "\n".join(lines)


def summarise_replication(spec_name: str, reports: list[EvaluationReport]) -> ReplicationReport:
    """Collapse several same-spec runs into per-metric spreads."""
    if len(reports) < 2:
        raise ValueError(f"replication needs at least 2 runs, got {len(reports)}")

    names = set(reports[0].metrics)
    for r in reports[1:]:
        names &= set(r.metrics)
    if not names:
        raise ValueError("the runs share no metrics")

    per_metric: dict[str, SeedSpread] = {}
    for name in sorted(names):
        vals = np.array([r.metrics[name].point for r in reports], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue
        per_metric[name] = SeedSpread(
            metric=name,
            mean=float(vals.mean()),
            sd=float(vals.std(ddof=1)),
            min=float(vals.min()),
            max=float(vals.max()),
            n_seeds=int(vals.size),
        )
    return ReplicationReport(
        spec_name=spec_name, n_seeds=len(reports), per_metric=per_metric
    )


def noise_floor_from_replication(rep: ReplicationReport, *, k: float = 2.0) -> dict[str, float]:
    """Minimum detectable effect per metric, as ``k`` standard deviations.

    ``k=2`` is the default: an improvement smaller than two seed-to-seed
    standard deviations is not evidence of anything. Raise ``k`` to be
    stricter; lowering it below 1 defeats the purpose.
    """
    return {name: k * s.sd for name, s in rep.per_metric.items()}


# ==========================================================================
# Campaign: the state that makes the switch irreversible
# ==========================================================================
class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str
    report: EvaluationReport
    hypothesis: str = ""
    predicted_direction: Literal["up", "down", "none", ""] = ""
    verdict: Verdict | None = None
    recorded: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class _State(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: ObjectiveSpec
    history: list[_Entry] = Field(default_factory=list)
    active_tier_index: int = 0
    frozen_gate: dict[str, float] = Field(default_factory=dict)
    switch_log: list[str] = Field(default_factory=list)


class Campaign:
    """One search, with its objective, history and tier state persisted.

    Persistence is not convenience: it is what makes the tier switch
    irreversible across sessions. A campaign reloaded tomorrow cannot quietly
    go back to optimising the metric that was giving better numbers.
    """

    def __init__(self, state: _State, path) -> None:
        self._state = state
        self._path = path

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def declare(cls, objective: ObjectiveSpec, *, overwrite: bool = False) -> "Campaign":
        """Create a campaign. The objective is fixed from this point on."""
        ws.slug(objective.name)
        path = ws.write_path("campaigns", f"{objective.name}.json")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"campaign {objective.name!r} already exists. Load it instead of "
                "redeclaring: changing the objective mid-campaign invalidates "
                "every comparison already made."
            )
        camp = cls(_State(objective=objective), path)
        camp._save()
        return camp

    @classmethod
    def load(cls, name: str) -> "Campaign":
        path = ws.write_path("campaigns", f"{ws.slug(name)}.json")
        if not path.exists():
            raise FileNotFoundError(f"no campaign named {name!r} ({path})")
        return cls(_State.model_validate_json(path.read_text(encoding="utf-8")), path)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self._state.model_dump_json(indent=2), encoding="utf-8")

    # -- accessors ---------------------------------------------------------
    @property
    def objective(self) -> ObjectiveSpec:
        return self._state.objective

    @property
    def n_experiments(self) -> int:
        return len(self._state.history)

    def budget_remaining(self) -> int:
        return max(0, self.objective.eval_budget - self.n_experiments)

    def active_tier(self) -> Tier:
        """The tier currently being optimised."""
        opts = self.objective.optimize_tiers()
        if not opts:
            raise ValueError("objective has no optimize tier")
        return opts[min(self._state.active_tier_index, len(opts) - 1)]

    def set_noise_floor(self, floors: dict[str, float]) -> None:
        """Install the measured noise floor. Do this before the first comparison."""
        self.objective.noise_floor.update(floors)
        self._save()

    def best(self, metric: str) -> MetricWithCI | None:
        vals = [
            e.report.metrics[metric]
            for e in self._state.history
            if metric in e.report.metrics and np.isfinite(e.report.metrics[metric].point)
        ]
        return max(vals, key=lambda m: m.point) if vals else None

    def history(self) -> list[_Entry]:
        return list(self._state.history)

    # -- saturation --------------------------------------------------------
    def check_saturation(self, metric: str | None = None) -> tuple[bool, str]:
        """Has the active tier stopped discriminating between candidates?

        Saturated when all of these hold:

        * at least ``min_experiments`` have been run;
        * the top ``top_k`` candidates' intervals all mutually overlap, so the
          metric can no longer tell them apart;
        * the best value has not improved by more than the noise floor in the
          last ``patience`` experiments.
        """
        rule = self.objective.saturation
        metric = metric or self.active_tier().metric

        pts = [
            e.report.metrics[metric]
            for e in self._state.history
            if metric in e.report.metrics and np.isfinite(e.report.metrics[metric].point)
        ]
        if len(pts) < rule.min_experiments:
            return False, f"only {len(pts)} experiments; need {rule.min_experiments}"

        leaders = sorted(pts, key=lambda m: m.point, reverse=True)[: rule.top_k]
        for i, a in enumerate(leaders):
            for b in leaders[i + 1 :]:
                if not a.ties_with(b):
                    return False, (
                        f"top-{rule.top_k} are still distinguishable "
                        f"({a.point:+.4f} vs {b.point:+.4f}, intervals disjoint)"
                    )

        floor = self.objective.floor(metric)
        recent, earlier = pts[-rule.patience :], pts[: -rule.patience]
        if earlier:
            best_before = max(m.point for m in earlier)
            best_recent = max(m.point for m in recent)
            if best_recent > best_before + floor:
                return False, (
                    f"still improving: {best_before:+.4f} -> {best_recent:+.4f} "
                    f"exceeds the noise floor {floor:.4f}"
                )

        return True, (
            f"top-{rule.top_k} intervals all overlap and no improvement beyond "
            f"the noise floor ({floor:.4f}) in the last {rule.patience} experiments"
        )


    def effective_gates(self) -> list[Tier]:
        """Every tier currently acting as a gate.

        That is: the tiers declared as gates, plus the optimize tiers that
        have already saturated and been demoted. A demoted tier keeps
        protecting what it measures even though it no longer drives the
        search -- which is the whole point of demoting rather than dropping it.
        """
        return [
            t
            for t in self.objective.tiers
            if t.mode == "gate" or t.metric in self._state.frozen_gate
        ]

    def maybe_switch_tier(self) -> bool:
        """Demote the active tier and advance, if it has saturated.

        Demotion freezes the tier's threshold at the best value observed so
        far and turns it into a gate. Irreversible by construction: there is
        no method that moves the index back, and the state is persisted, so a
        campaign reloaded tomorrow cannot return to the metric that was
        giving nicer numbers.
        """
        if self._state.active_tier_index >= len(self.objective.optimize_tiers()) - 1:
            return False

        saturated, reason = self.check_saturation()
        if not saturated:
            return False

        demoted = self.active_tier().metric
        self._freeze(demoted)                      # the demotion itself
        for tier in self.objective.declared_gates():
            self._freeze(tier.metric)
        self._state.active_tier_index += 1

        self._state.switch_log.append(
            f"[{datetime.now(timezone.utc).isoformat()}] after {self.n_experiments} "
            f"experiments: {demoted} saturated ({reason}); demoted to a gate at "
            f"{self._state.frozen_gate.get(demoted)}; now optimising "
            f"{self.active_tier().metric}"
        )
        self._save()
        return True

    def _freeze(self, metric: str) -> None:
        """Pin a metric's gate reference to the best value observed so far.

        Never re-freezes: once set, the reference cannot drift upward as
        sampling continues, which would be a selection effect of its own.
        """
        if metric in self._state.frozen_gate:
            return
        best = self.best(metric)
        if best is not None:
            self._state.frozen_gate[metric] = best.point

    def _freeze_gates(self) -> None:
        """Freeze all declared gates. Demoted tiers are frozen at demotion."""
        for tier in self.objective.declared_gates():
            self._freeze(tier.metric)

    # -- the verdict -------------------------------------------------------
    def compare(self, candidate: EvaluationReport) -> Verdict:
        """Decide whether ``candidate`` beats the incumbent, under the objective.

        Gates are checked first, then the active optimize tier, then the lower
        tiers as tie-breaks.
        """
        if self.budget_remaining() <= 0:
            return Verdict(
                outcome="budget_exhausted",
                reason=(
                    f"the evaluation budget of {self.objective.eval_budget} experiments is "
                    "spent. Continuing would select on noise in the measured set."
                ),
            )

        # --- gates: declared ones plus every demoted (saturated) tier
        for tier in self.effective_gates():
            if tier.metric not in candidate.metrics:
                continue
            cand = candidate.metrics[tier.metric]
            ref = self._state.frozen_gate.get(tier.metric)
            if ref is None:
                best = self.best(tier.metric)
                ref = best.point if best else None
            if ref is None:
                continue
            tol = tier.tolerance if tier.tolerance is not None else self.objective.floor(tier.metric)
            # Reject only when the candidate is PROVABLY worse: its whole
            # interval sits below the reference minus tolerance. Rejecting on
            # a point estimate would discard good models over noise, which is
            # precisely what this metric is too noisy to support.
            if np.isfinite(cand.ci_high) and cand.ci_high < ref - tol:
                return Verdict(
                    outcome="gate_failed",
                    deciding_metric=tier.metric,
                    reason=(
                        f"{tier.metric} is provably worse: interval tops out at "
                        f"{cand.ci_high:+.4f}, below the frozen reference {ref:+.4f} "
                        f"minus tolerance {tol:.4f}"
                    ),
                    details={"reference": f"{ref:+.4f}", "tolerance": f"{tol:.4f}"},
                )

        # --- optimize tiers, active one first
        opts = self.objective.optimize_tiers()[self._state.active_tier_index :]
        for tier in opts:
            if tier.metric not in candidate.metrics:
                continue
            incumbent = self.best(tier.metric)
            cand = candidate.metrics[tier.metric]
            if incumbent is None:
                return Verdict(
                    outcome="improved",
                    deciding_metric=tier.metric,
                    reason=f"first measurement of {tier.metric}: {cand.point:+.4f}",
                )
            if cand.beats(incumbent):
                return Verdict(
                    outcome="improved",
                    deciding_metric=tier.metric,
                    reason=(
                        f"{tier.metric}: {cand.point:+.4f} (CI low {cand.ci_low:+.4f}) "
                        f"clears the incumbent {incumbent.point:+.4f}"
                    ),
                )
            if incumbent.beats(cand):
                return Verdict(
                    outcome="worse",
                    deciding_metric=tier.metric,
                    reason=(
                        f"{tier.metric}: {cand.point:+.4f} is below the incumbent "
                        f"{incumbent.point:+.4f} beyond the interval overlap"
                    ),
                )
            # tied on this tier -> fall through to the next one

        return Verdict(
            outcome="tied",
            reason="no tier distinguishes this candidate from the incumbent",
        )

    # -- recording ---------------------------------------------------------
    def record(
        self,
        experiment: str,
        report: EvaluationReport,
        *,
        hypothesis: str = "",
        predicted_direction: str = "",
        verdict: Verdict | None = None,
    ) -> Verdict:
        """Append an experiment, with the hypothesis that motivated it.

        Recording the hypothesis *before* the outcome is what makes the
        proposer's reasoning falsifiable. After twenty experiments you can
        check whether its predictions beat chance -- and if they do not, the
        LLM is adding nothing on that axis and a sweep would do as well.
        """
        verdict = verdict or self.compare(report)
        self._state.history.append(
            _Entry(
                experiment=experiment,
                report=report,
                hypothesis=hypothesis,
                predicted_direction=predicted_direction,  # type: ignore[arg-type]
                verdict=verdict,
            )
        )
        self._save()
        return verdict

    def hypothesis_scorecard(self) -> dict[str, int | float]:
        """How often the proposer's predicted direction matched the outcome.

        A hit rate near 0.5 means the proposals carry no information and the
        search should be handed to a deterministic sweep.
        """
        # 'tied' is inconclusive, not a miss: a prediction that the metric
        # would improve is not refuted by an experiment that could not tell.
        # Scoring ties as failures would understate the proposer systematically.
        scored = [
            e
            for e in self._state.history
            if e.predicted_direction in ("up", "down")
            and e.verdict is not None
            and e.verdict.outcome in ("improved", "worse", "gate_failed")
        ]
        if not scored:
            return {"n": 0, "hit_rate": float("nan")}
        hits = sum(
            1
            for e in scored
            if (e.predicted_direction == "up" and e.verdict.outcome == "improved")
            or (e.predicted_direction == "down" and e.verdict.outcome in ("worse", "gate_failed"))
        )
        return {"n": len(scored), "hits": hits, "hit_rate": round(hits / len(scored), 3)}

    def status(self) -> dict[str, object]:
        saturated, reason = (
            self.check_saturation() if self.n_experiments else (False, "no experiments yet")
        )
        return {
            "campaign": self.objective.name,
            "experiments": self.n_experiments,
            "budget_remaining": self.budget_remaining(),
            "active_tier": self.active_tier().metric,
            "tier_index": self._state.active_tier_index,
            "frozen_gate": dict(self._state.frozen_gate),
            "saturated": saturated,
            "saturation_reason": reason,
            "switch_log": list(self._state.switch_log),
            "hypothesis_scorecard": self.hypothesis_scorecard(),
        }
