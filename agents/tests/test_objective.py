"""The protocol: saturation, the irreversible switch, the frozen gate, verdicts.

These tests are the ones that matter most in this package. Every scenario
here is a way the search could fool itself, and the point of the module is
that it cannot.
"""

import pytest

from agents.core import objective as obj
from agents.core import workspace as ws
from agents.core.evaluation import EvaluationReport, MetricWithCI

AFF = "affinity.KD.spearman"
ENR = "enrichment.spearman"
CNT = "counts.heldout_loglik"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def metric(name, point, half=0.05, n=35):
    return MetricWithCI(
        name=name, point=point, ci_low=point - half, ci_high=point + half,
        n=n, raw_point=point, n_bootstrap=1000,
    )


def report(name, **metrics):
    """report("m1", affinity=(0.45, 0.3), enrichment=(0.6, 0.02))"""
    keymap = {"affinity": AFF, "enrichment": ENR, "counts": CNT}
    return EvaluationReport(
        model_name=name,
        metrics={
            keymap[k]: metric(keymap[k], v[0], v[1]) for k, v in metrics.items()
        },
    )


def objective(budget=50, **kw):
    return obj.ObjectiveSpec(
        name=kw.pop("name", "test-campaign"),
        tiers=[
            obj.Tier(metric=AFF, mode="gate"),
            obj.Tier(metric=ENR, mode="optimize"),
            obj.Tier(metric=CNT, mode="optimize"),
        ],
        noise_floor={AFF: 0.10, ENR: 0.02, CNT: 0.01},
        eval_budget=budget,
        saturation=obj.SaturationRule(top_k=3, patience=3, min_experiments=5),
        **kw,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "STORE", tmp_path / "store")
    return tmp_path


# --------------------------------------------------------------------------
# replication / noise floor
# --------------------------------------------------------------------------
class TestReplication:
    def test_spread_is_computed_per_metric(self):
        reps = [report(f"s{i}", affinity=(0.4 + 0.05 * i, 0.3)) for i in range(5)]
        r = obj.summarise_replication("spec", reps)
        s = r.per_metric[AFF]
        assert s.n_seeds == 5
        assert s.mean == pytest.approx(0.5)
        assert s.spread == pytest.approx(0.2)

    def test_noise_floor_is_two_sd_by_default(self):
        reps = [report(f"s{i}", affinity=(p, 0.3)) for i, p in enumerate([0.4, 0.5, 0.6])]
        r = obj.summarise_replication("spec", reps)
        floors = obj.noise_floor_from_replication(r)
        assert floors[AFF] == pytest.approx(2 * r.per_metric[AFF].sd)

    def test_single_run_is_refused(self):
        with pytest.raises(ValueError, match="at least 2"):
            obj.summarise_replication("spec", [report("only", affinity=(0.4, 0.3))])

    def test_disjoint_metrics_are_refused(self):
        with pytest.raises(ValueError, match="share no metrics"):
            obj.summarise_replication(
                "spec", [report("a", affinity=(0.4, 0.3)), report("b", enrichment=(0.5, 0.02))]
            )

    def test_summary_is_readable(self):
        reps = [report(f"s{i}", affinity=(0.4 + 0.05 * i, 0.3)) for i in range(3)]
        assert "3 seeds" in obj.summarise_replication("spec", reps).summary()


# --------------------------------------------------------------------------
# campaign lifecycle
# --------------------------------------------------------------------------
class TestCampaignLifecycle:
    def test_declare_then_load_round_trips(self, store):
        obj.Campaign.declare(objective())
        c = obj.Campaign.load("test-campaign")
        assert c.objective.eval_budget == 50
        assert c.active_tier().metric == ENR

    def test_redeclaring_is_refused(self, store):
        obj.Campaign.declare(objective())
        with pytest.raises(FileExistsError, match="invalidates"):
            obj.Campaign.declare(objective())

    def test_loading_a_missing_campaign_is_clear(self, store):
        with pytest.raises(FileNotFoundError, match="no campaign named"):
            obj.Campaign.load("never-created")

    def test_budget_counts_down(self, store):
        c = obj.Campaign.declare(objective(budget=3))
        assert c.budget_remaining() == 3
        c.record("e1", report("e1", affinity=(0.4, 0.3), enrichment=(0.5, 0.02)))
        assert c.budget_remaining() == 2

    def test_budget_exhaustion_stops_the_search(self, store):
        c = obj.Campaign.declare(objective(budget=1))
        c.record("e1", report("e1", affinity=(0.4, 0.3), enrichment=(0.5, 0.02)))
        v = c.compare(report("e2", affinity=(0.9, 0.3), enrichment=(0.99, 0.02)))
        assert v.outcome == "budget_exhausted"
        assert not v.accepted


# --------------------------------------------------------------------------
# saturation: the rule must not fire early
# --------------------------------------------------------------------------
class TestSaturation:
    def _fill(self, c, points, half=0.02):
        for i, p in enumerate(points):
            c.record(f"e{i}", report(f"e{i}", affinity=(0.4, 0.3), enrichment=(p, half)))

    def test_not_before_min_experiments(self, store):
        c = obj.Campaign.declare(objective())
        self._fill(c, [0.50, 0.50, 0.50])
        saturated, reason = c.check_saturation()
        assert not saturated and "need 5" in reason

    def test_not_while_leaders_are_distinguishable(self, store):
        c = obj.Campaign.declare(objective())
        self._fill(c, [0.10, 0.30, 0.50, 0.70, 0.90], half=0.01)
        saturated, reason = c.check_saturation()
        assert not saturated and "distinguishable" in reason

    def test_not_while_still_improving(self, store):
        c = obj.Campaign.declare(objective())
        # Leaders overlap widely, but the best keeps climbing past the floor.
        self._fill(c, [0.50, 0.52, 0.54, 0.80, 0.82, 0.84], half=0.40)
        saturated, reason = c.check_saturation()
        assert not saturated and "still improving" in reason

    def test_fires_when_tied_and_flat(self, store):
        c = obj.Campaign.declare(objective())
        self._fill(c, [0.50, 0.51, 0.49, 0.50, 0.51, 0.50], half=0.10)
        saturated, reason = c.check_saturation()
        assert saturated and "overlap" in reason


# --------------------------------------------------------------------------
# the switch: irreversible, and it freezes the gate
# --------------------------------------------------------------------------
class TestTierSwitch:
    def _saturate(self, c):
        for i, p in enumerate([0.50, 0.51, 0.49, 0.50, 0.51, 0.50]):
            c.record(
                f"e{i}",
                report(f"e{i}", affinity=(0.40 + 0.01 * i, 0.30), enrichment=(p, 0.10)),
            )

    def test_switch_advances_the_active_tier(self, store):
        c = obj.Campaign.declare(objective())
        assert c.active_tier().metric == ENR
        self._saturate(c)
        assert c.maybe_switch_tier() is True
        assert c.active_tier().metric == CNT

    def test_switch_freezes_the_gate_at_the_best_so_far(self, store):
        c = obj.Campaign.declare(objective())
        self._saturate(c)
        c.maybe_switch_tier()
        # best affinity seen was 0.40 + 0.01*5 = 0.45
        assert c.status()["frozen_gate"][AFF] == pytest.approx(0.45)

    def test_gate_does_not_drift_upward_after_freezing(self, store):
        c = obj.Campaign.declare(objective())
        self._saturate(c)
        c.maybe_switch_tier()
        frozen = c.status()["frozen_gate"][AFF]
        # A much better affinity arrives later; the reference must not move.
        c.record("later", report("later", affinity=(0.95, 0.05), enrichment=(0.5, 0.1)))
        c._freeze_gates()
        assert c.status()["frozen_gate"][AFF] == pytest.approx(frozen)

    def test_switch_is_not_offered_before_saturation(self, store):
        c = obj.Campaign.declare(objective())
        c.record("e0", report("e0", affinity=(0.4, 0.3), enrichment=(0.5, 0.02)))
        assert c.maybe_switch_tier() is False
        assert c.active_tier().metric == ENR

    def test_cannot_advance_past_the_last_tier(self, store):
        c = obj.Campaign.declare(objective())
        self._saturate(c)
        assert c.maybe_switch_tier() is True
        assert c.maybe_switch_tier() is False      # no tier left

    def test_switch_survives_reload_and_cannot_be_undone(self, store):
        c = obj.Campaign.declare(objective())
        self._saturate(c)
        c.maybe_switch_tier()
        reloaded = obj.Campaign.load("test-campaign")
        assert reloaded.active_tier().metric == CNT
        assert reloaded.status()["switch_log"]

    def test_switch_is_logged_with_its_reason(self, store):
        c = obj.Campaign.declare(objective())
        self._saturate(c)
        c.maybe_switch_tier()
        log = c.status()["switch_log"][0]
        assert ENR in log and "saturated" in log


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------
class TestVerdicts:
    def test_first_measurement_counts_as_improvement(self, store):
        c = obj.Campaign.declare(objective())
        v = c.compare(report("e1", affinity=(0.4, 0.3), enrichment=(0.5, 0.02)))
        assert v.outcome == "improved" and v.deciding_metric == ENR

    def test_improvement_requires_clearing_the_interval(self, store):
        c = obj.Campaign.declare(objective())
        c.record("e1", report("e1", affinity=(0.4, 0.3), enrichment=(0.50, 0.02)))

        clearly_better = c.compare(report("e2", affinity=(0.4, 0.3), enrichment=(0.70, 0.02)))
        assert clearly_better.outcome == "improved"

        marginal = c.compare(report("e3", affinity=(0.4, 0.3), enrichment=(0.51, 0.05)))
        assert marginal.outcome != "improved"

    def test_tie_on_the_active_tier_falls_through_to_the_next(self, store):
        c = obj.Campaign.declare(objective())
        c.record("e1", report("e1", affinity=(0.4, 0.3), enrichment=(0.50, 0.20),
                              counts=(1.0, 0.01)))
        v = c.compare(report("e2", affinity=(0.4, 0.3), enrichment=(0.52, 0.20),
                             counts=(2.0, 0.01)))
        assert v.outcome == "improved" and v.deciding_metric == CNT

    def test_worse_on_the_active_tier_is_rejected(self, store):
        c = obj.Campaign.declare(objective())
        c.record("e1", report("e1", affinity=(0.4, 0.3), enrichment=(0.80, 0.02)))
        v = c.compare(report("e2", affinity=(0.4, 0.3), enrichment=(0.20, 0.02)))
        assert v.outcome == "worse" and v.deciding_metric == ENR


class TestGate:
    def test_provably_worse_affinity_is_rejected(self, store):
        c = obj.Campaign.declare(objective())
        c.record("e1", report("e1", affinity=(0.60, 0.02), enrichment=(0.50, 0.02)))
        # tolerance is the noise floor, 0.10; candidate tops out at 0.12
        v = c.compare(report("e2", affinity=(0.10, 0.02), enrichment=(0.99, 0.02)))
        assert v.outcome == "gate_failed" and v.deciding_metric == AFF

    def test_noisily_worse_affinity_gets_the_benefit_of_the_doubt(self, store):
        """The gate must not reject on noise -- that is why it is a gate."""
        c = obj.Campaign.declare(objective())
        c.record("e1", report("e1", affinity=(0.60, 0.30), enrichment=(0.50, 0.02)))
        v = c.compare(report("e2", affinity=(0.45, 0.30), enrichment=(0.80, 0.02)))
        assert v.outcome == "improved"

    def test_gate_blocks_even_a_large_tier2_gain(self, store):
        """A big enrichment win cannot buy its way past a damaged gate."""
        c = obj.Campaign.declare(objective())
        c.record("e1", report("e1", affinity=(0.70, 0.02), enrichment=(0.10, 0.02)))
        v = c.compare(report("e2", affinity=(0.05, 0.02), enrichment=(0.99, 0.02)))
        assert v.outcome == "gate_failed"

    def test_explicit_tolerance_overrides_the_noise_floor(self, store):
        spec = objective()
        spec.tiers[0].tolerance = 0.80          # very permissive
        c = obj.Campaign.declare(spec)
        c.record("e1", report("e1", affinity=(0.70, 0.02), enrichment=(0.10, 0.02)))
        v = c.compare(report("e2", affinity=(0.05, 0.02), enrichment=(0.99, 0.02)))
        assert v.outcome == "improved"


class TestDemotion:
    """A tier is optimised while it discriminates, then becomes a gate."""

    def cascade_objective(self, budget=50):
        return obj.ObjectiveSpec(
            name="cascade",
            tiers=[
                obj.Tier(metric=AFF, mode="optimize"),   # demotes on saturation
                obj.Tier(metric=ENR, mode="optimize"),
                obj.Tier(metric=CNT, mode="optimize"),
            ],
            noise_floor={AFF: 0.05, ENR: 0.02, CNT: 0.01},
            eval_budget=budget,
            saturation=obj.SaturationRule(top_k=3, patience=3, min_experiments=5),
        )

    def _saturate_affinity(self, c):
        for i, aff in enumerate([0.30, 0.44, 0.46, 0.45, 0.46, 0.44]):
            c.record(f"e{i}", report(f"e{i}", affinity=(aff, 0.30),
                                     enrichment=(0.50 + 0.01 * i, 0.02)))

    def test_affinity_starts_as_the_active_tier(self, store):
        c = obj.Campaign.declare(self.cascade_objective())
        assert c.active_tier().metric == AFF
        assert c.effective_gates() == []

    def test_saturation_demotes_it_to_a_gate(self, store):
        c = obj.Campaign.declare(self.cascade_objective())
        self._saturate_affinity(c)
        assert c.maybe_switch_tier() is True
        assert c.active_tier().metric == ENR
        assert AFF in [t.metric for t in c.effective_gates()]
        assert c.status()["frozen_gate"][AFF] == pytest.approx(0.46)

    def test_before_demotion_affinity_cannot_block(self, store):
        """While it is still being optimised, a low affinity is a tie, not a veto."""
        c = obj.Campaign.declare(self.cascade_objective())
        c.record("e1", report("e1", affinity=(0.60, 0.02), enrichment=(0.50, 0.02)))
        v = c.compare(report("e2", affinity=(0.10, 0.02), enrichment=(0.99, 0.02)))
        assert v.outcome == "worse"          # decided ON affinity, not blocked BY it
        assert v.deciding_metric == AFF

    def test_after_demotion_affinity_blocks(self, store):
        c = obj.Campaign.declare(self.cascade_objective())
        self._saturate_affinity(c)
        c.maybe_switch_tier()
        v = c.compare(report("damaged", affinity=(-0.10, 0.02), enrichment=(0.99, 0.02)))
        assert v.outcome == "gate_failed" and v.deciding_metric == AFF

    def test_demotion_survives_reload(self, store):
        c = obj.Campaign.declare(self.cascade_objective())
        self._saturate_affinity(c)
        c.maybe_switch_tier()
        reloaded = obj.Campaign.load("cascade")
        assert reloaded.active_tier().metric == ENR
        assert AFF in [t.metric for t in reloaded.effective_gates()]

    def test_demoted_threshold_never_re_freezes(self, store):
        c = obj.Campaign.declare(self.cascade_objective())
        self._saturate_affinity(c)
        c.maybe_switch_tier()
        frozen = c.status()["frozen_gate"][AFF]
        c.record("later", report("later", affinity=(0.95, 0.02), enrichment=(0.9, 0.02)))
        c._freeze(AFF)
        assert c.status()["frozen_gate"][AFF] == pytest.approx(frozen)

    def test_cascade_runs_out_of_tiers(self, store):
        c = obj.Campaign.declare(self.cascade_objective())
        self._saturate_affinity(c)
        assert c.maybe_switch_tier() is True     # AFF -> ENR
        for i, enr in enumerate([0.50, 0.51, 0.49, 0.50, 0.51, 0.50]):
            c.record(f"x{i}", report(f"x{i}", affinity=(0.46, 0.30), enrichment=(enr, 0.10)))
        assert c.maybe_switch_tier() is True     # ENR -> CNT
        assert c.active_tier().metric == CNT
        assert c.maybe_switch_tier() is False    # nothing left


class TestHypothesisScorecard:
    def test_empty_until_predictions_are_made(self, store):
        c = obj.Campaign.declare(objective())
        assert c.hypothesis_scorecard()["n"] == 0

    def test_counts_hits_against_outcomes(self, store):
        c = obj.Campaign.declare(objective())
        c.record("e1", report("e1", affinity=(0.4, 0.3), enrichment=(0.50, 0.02)),
                 hypothesis="wider layer helps", predicted_direction="up")
        c.record("e2", report("e2", affinity=(0.4, 0.3), enrichment=(0.90, 0.02)),
                 hypothesis="wider still", predicted_direction="up")
        c.record("e3", report("e3", affinity=(0.4, 0.3), enrichment=(0.10, 0.02)),
                 hypothesis="this should hurt", predicted_direction="down")
        card = c.hypothesis_scorecard()
        assert card["n"] == 3
        assert card["hit_rate"] == pytest.approx(1.0)

    def test_chance_level_predictions_are_visible(self, store):
        c = obj.Campaign.declare(objective())
        c.record("e1", report("e1", affinity=(0.4, 0.3), enrichment=(0.50, 0.02)),
                 predicted_direction="up")
        c.record("e2", report("e2", affinity=(0.4, 0.3), enrichment=(0.20, 0.02)),
                 predicted_direction="up")      # predicted up, got worse
        assert c.hypothesis_scorecard()["hit_rate"] < 1.0

    def test_ties_are_inconclusive_not_misses(self, store):
        """An experiment that could not tell does not refute the prediction."""
        c = obj.Campaign.declare(objective())
        c.record("e1", report("e1", affinity=(0.4, 0.3), enrichment=(0.50, 0.02)),
                 predicted_direction="up")                      # improved: scored
        c.record("e2", report("e2", affinity=(0.4, 0.3), enrichment=(0.505, 0.05)),
                 predicted_direction="up")                      # tied: not scored
        card = c.hypothesis_scorecard()
        assert card["n"] == 1
        assert card["hit_rate"] == pytest.approx(1.0)


def test_status_is_self_describing(store):
    c = obj.Campaign.declare(objective())
    c.record("e1", report("e1", affinity=(0.4, 0.3), enrichment=(0.5, 0.02)))
    s = c.status()
    assert s["campaign"] == "test-campaign"
    assert s["experiments"] == 1
    assert s["active_tier"] == ENR
    assert "saturation_reason" in s
