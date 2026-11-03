"""Statistics, confidence intervals and metric orientation.

The orientation tests matter as much as the correlation ones: a sign flip in
the wrong place inverts the entire search and produces a confident, wrong
answer rather than an error.
"""

import math

import numpy as np
import pytest

from agents.core import evaluation as ev


class TestCorrelations:
    def test_pearson_perfect(self):
        assert ev.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
        assert ev.pearson([1, 2, 3, 4], [-2, -4, -6, -8]) == pytest.approx(-1.0)

    def test_pearson_known_value(self):
        # dx = [-2,-1,0,1,2], dy = [-1,-2,1,0,2] -> cov 8, denom 10
        x = [1, 2, 3, 4, 5]
        y = [2, 1, 4, 3, 5]
        assert ev.pearson(x, y) == pytest.approx(0.8, abs=1e-9)

    def test_spearman_is_monotone_invariant(self):
        x = [1, 2, 3, 4, 5]
        y = [1, 8, 27, 64, 125]          # monotone but very nonlinear
        assert ev.spearman(x, y) == pytest.approx(1.0)
        assert ev.pearson(x, y) < 0.98   # Pearson is not

    def test_spearman_averages_ties(self):
        # With all y tied the correlation is undefined, not 0.
        assert math.isnan(ev.spearman([1, 2, 3, 4], [1, 1, 1, 1]))
        # Partial ties still produce a finite value.
        assert np.isfinite(ev.spearman([1, 2, 3, 4], [1, 1, 2, 3]))

    def test_too_few_points_is_nan_not_zero(self):
        assert math.isnan(ev.pearson([1, 2], [1, 2]))

    def test_nonfinite_pairs_are_dropped(self):
        assert ev.pearson([1, 2, 3, np.nan], [2, 4, 6, 100]) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            ev.pearson([1, 2, 3], [1, 2])


class TestWeightedPearson:
    def test_equal_weights_match_unweighted(self):
        x = [1.0, 2.0, 3.0, 4.0, 7.0]
        y = [2.0, 1.0, 4.0, 3.0, 5.0]
        assert ev.weighted_pearson(x, y, [1.0] * 5) == pytest.approx(ev.pearson(x, y))

    def test_zero_weight_excludes_a_point(self):
        x = [1.0, 2.0, 3.0, 100.0]
        y = [1.0, 2.0, 3.0, -100.0]      # outlier destroys the unweighted value
        assert ev.weighted_pearson(x, y, [1, 1, 1, 0]) == pytest.approx(1.0)
        assert ev.pearson(x, y) < 0

    def test_weights_shift_the_estimate(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 2.0, 3.0, -5.0]
        heavy_on_clean = ev.weighted_pearson(x, y, [10, 10, 10, 1])
        assert heavy_on_clean > ev.pearson(x, y)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            ev.weighted_pearson([1, 2, 3], [1, 2, 3], [1, 2])


class TestBootstrap:
    def _corr_metric(self, n, rho_target, seed=0, n_boot=500):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        y = rho_target * x + math.sqrt(1 - rho_target**2) * rng.normal(size=n)
        return ev.correlate(x, y, name="t", method="pearson", n_bootstrap=n_boot, seed=1)

    def test_interval_brackets_the_estimate(self):
        m = self._corr_metric(200, 0.6)
        assert m.ci_low <= m.point <= m.ci_high

    def test_interval_narrows_with_more_data(self):
        narrow = self._corr_metric(400, 0.6, seed=3)
        wide = self._corr_metric(30, 0.6, seed=3)
        assert narrow.width < wide.width

    def test_small_n_interval_is_wide_enough_to_matter(self):
        """The motivating fact: ~35 points cannot resolve small differences."""
        m = self._corr_metric(35, 0.45, seed=7)
        assert m.width > 0.3

    def test_too_few_observations_gives_nan_not_a_fake_interval(self):
        m = ev.correlate([1.0, 2.0], [1.0, 2.0], name="t", n_bootstrap=100)
        assert math.isnan(m.ci_low) and math.isnan(m.ci_high)
        assert m.n_bootstrap == 0

    def test_reports_how_many_resamples_were_usable(self):
        m = self._corr_metric(50, 0.5, n_boot=300)
        assert 0 < m.n_bootstrap <= 300


class TestOrientation:
    def test_flip_negates_point_and_swaps_interval(self):
        x = np.linspace(0, 1, 60)
        y = -x + np.random.default_rng(0).normal(scale=0.1, size=60)

        plain = ev.correlate(x, y, name="p", method="pearson", n_bootstrap=300, seed=2)
        flipped = ev.correlate(x, y, name="f", method="pearson", orient=-1,
                               n_bootstrap=300, seed=2)

        assert plain.point < 0 and flipped.point > 0
        assert flipped.point == pytest.approx(-plain.point)
        assert flipped.ci_low == pytest.approx(-plain.ci_high)
        assert flipped.ci_high == pytest.approx(-plain.ci_low)

    def test_raw_point_survives_orientation(self):
        x = np.linspace(0, 1, 40)
        y = -x
        m = ev.correlate(x, y, name="f", method="pearson", orient=-1, n_bootstrap=100)
        assert m.raw_point == pytest.approx(-1.0)
        assert m.point == pytest.approx(1.0)

    def test_interval_stays_ordered_after_flip(self):
        x = np.linspace(0, 1, 50)
        y = -x + np.random.default_rng(1).normal(scale=0.3, size=50)
        m = ev.correlate(x, y, name="f", method="pearson", orient=-1, n_bootstrap=300, seed=1)
        assert m.ci_low <= m.ci_high


class TestMetricComparison:
    def _m(self, point, half):
        return ev.MetricWithCI(
            name="m", point=point, ci_low=point - half, ci_high=point + half,
            n=35, raw_point=point,
        )

    def test_beats_requires_clearing_the_point_estimate(self):
        assert self._m(0.9, 0.05).beats(self._m(0.5, 0.05))
        # higher point but overlapping interval is not evidence
        assert not self._m(0.55, 0.3).beats(self._m(0.5, 0.3))

    def test_ties_when_intervals_overlap(self):
        assert self._m(0.5, 0.3).ties_with(self._m(0.6, 0.3))
        assert not self._m(0.1, 0.05).ties_with(self._m(0.9, 0.05))

    def test_ties_is_symmetric(self):
        a, b = self._m(0.5, 0.3), self._m(0.9, 0.05)
        assert a.ties_with(b) == b.ties_with(a)


class TestEnrichment:
    def test_log_enrichment_sign(self):
        le = ev.log_enrichment([100, 100], [1000, 10], normalise=False)
        assert le[0] > 0 and le[1] < 0

    def test_depth_normalisation_removes_a_global_scale(self):
        """Exact invariance holds without a pseudocount."""
        nb = [100, 200, 300]
        plain = ev.log_enrichment(nb, [100, 200, 300], normalise=True, pseudocount=0.0)
        doubled = ev.log_enrichment(nb, [200, 400, 600], normalise=True, pseudocount=0.0)
        assert np.allclose(plain, doubled, atol=1e-12)

    def test_pseudocount_makes_the_invariance_approximate(self):
        """With a pseudocount the cancellation is inexact by design -- it is a
        prior on unseen sequences, and a prior does not rescale."""
        nb = [100, 200, 300]
        plain = ev.log_enrichment(nb, [100, 200, 300], normalise=True)
        doubled = ev.log_enrichment(nb, [200, 400, 600], normalise=True)
        assert not np.allclose(plain, doubled, atol=1e-9)
        assert np.allclose(plain, doubled, atol=1e-2)

    def test_weights_follow_the_delta_method(self):
        w = ev.enrichment_weights([1000], [1000], pseudocount=0.0)
        assert w[0] == pytest.approx(500.0)       # 1/(1/1000 + 1/1000)

    def test_low_counts_get_low_weight(self):
        w = ev.enrichment_weights([2, 5000], [3, 5000])
        assert w[0] < w[1] / 100

    def test_perfect_model_scores_near_plus_one(self):
        """Energy anticorrelated with enrichment is a GOOD model: point > 0."""
        rng = np.random.default_rng(0)
        n = 300
        energy = rng.normal(size=n)
        before = rng.integers(500, 2000, size=n).astype(float)
        after = before * np.exp(-2.0 * energy)      # low energy -> enriched

        m = ev.enrichment_metrics(energy, before, after, n_bootstrap=200)
        assert m["enrichment.spearman"].point > 0.9
        assert m["enrichment.spearman"].raw_point < 0      # raw is negative

    def test_min_count_filter_is_recorded(self):
        before = [1.0, 1.0, 900.0, 900.0, 900.0]
        after = [1.0, 2.0, 100.0, 500.0, 900.0]
        energy = [0.1, 0.2, 0.3, 0.4, 0.5]
        m = ev.enrichment_metrics(energy, before, after, min_count=10, n_bootstrap=50)
        assert "dropped=2" in m["enrichment.pearson"].note

    def test_weighted_and_unweighted_both_reported(self):
        rng = np.random.default_rng(2)
        n = 120
        energy = rng.normal(size=n)
        before = rng.integers(10, 3000, size=n).astype(float)
        after = before * np.exp(-energy)
        m = ev.enrichment_metrics(energy, before, after, n_bootstrap=100)
        assert {"enrichment.weighted_pearson", "enrichment.pearson",
                "enrichment.spearman"} <= set(m)


class TestAffinity:
    def test_kd_is_not_flipped(self):
        """Low Kd and low energy both mean a good binder: positive is healthy."""
        energy = np.linspace(-2, 2, 40)
        kd = 10 ** energy                       # log-linear in energy
        m = ev.affinity_metrics(energy, {"KD": kd}, n_bootstrap=200)
        assert m["affinity.KD.spearman"].point > 0.95

    def test_kon_is_flipped(self):
        """A high kon is a good binder, so kon orients opposite to Kd."""
        energy = np.linspace(-2, 2, 40)
        kon = 10 ** (-energy)                   # good binder (low E) has high kon
        m = ev.affinity_metrics(energy, {"kon": kon}, n_bootstrap=200)
        assert m["affinity.kon.spearman"].point > 0.95
        assert m["affinity.kon.spearman"].raw_point < 0

    def test_measures_are_kept_separate(self):
        energy = np.linspace(-2, 2, 30)
        m = ev.affinity_metrics(
            energy, {"KD": 10.0 ** energy, "koff": 10.0 ** energy}, n_bootstrap=100
        )
        assert set(m) == {"affinity.KD.spearman", "affinity.koff.spearman"}

    def test_nonpositive_values_become_nan_not_errors(self):
        energy = np.linspace(-2, 2, 30)
        kd = np.r_[np.zeros(3), 10.0 ** energy[3:]]
        m = ev.affinity_metrics(energy, {"KD": kd}, n_bootstrap=100)
        assert np.isfinite(m["affinity.KD.spearman"].point)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="values for"):
            ev.affinity_metrics([1.0, 2.0, 3.0], {"KD": [1.0, 2.0]})


class TestDisagreement:
    """Choosing where to spend the next affinity measurement."""

    def _dataset(self, n=50):
        """Model and enrichment in perfect agreement, as a baseline."""
        alphabet = "ADEFGHIKLMNPQRSTVWY"
        rng = np.random.default_rng(0)
        seqs = []
        while len(seqs) < n:
            s = "".join(rng.choice(list(alphabet), 15))
            if s not in seqs:
                seqs.append(s)
        energy = np.linspace(0.0, 1.0, n)
        before = np.full(n, 1000.0)
        after = 1000.0 * np.exp(-3.0 * energy)   # low energy -> enriched
        return seqs, energy, before, after

    def test_perfect_agreement_gives_no_gaps(self):
        seqs, e, nb, na = self._dataset()
        out = ev.rank_by_disagreement(seqs, e, nb, na, min_count=0)
        assert all(r.rank_gap == 0 for r in out)

    def test_planted_disagreement_surfaces_first(self):
        seqs, e, nb, na = self._dataset(n=50)
        # The best-predicted sequence is in fact the least enriched.
        na[0] = na[-1]
        out = ev.rank_by_disagreement(seqs, e, nb, na, min_count=0, top_k=3)
        assert out[0].seq == seqs[0]
        assert out[0].direction == "model_overrates"   # model says good, data says bad
        assert out[0].gap_fraction > 0.5

    def test_direction_labels_are_the_right_way_round(self):
        seqs, e, nb, na = self._dataset(n=40)
        # The worst-predicted sequence is in fact the most enriched.
        na[-1] = na[0] * 10
        out = ev.rank_by_disagreement(seqs, e, nb, na, min_count=0, top_k=1)
        assert out[0].seq == seqs[-1]
        assert out[0].direction == "model_underrates"

    def test_direction_filter(self):
        seqs, e, nb, na = self._dataset(n=40)
        na[0], na[-1] = na[-1], na[0]
        under = ev.rank_by_disagreement(
            seqs, e, nb, na, min_count=0, direction="model_underrates", top_k=10
        )
        assert under and all(r.direction == "model_underrates" for r in under)

    def test_low_count_sequences_are_dropped(self):
        seqs, e, nb, na = self._dataset(n=30)
        nb[:10] = 2.0                       # too shallow for enrichment to mean anything
        out = ev.rank_by_disagreement(seqs, e, nb, na, min_count=10, top_k=50)
        assert not (set(seqs[:10]) & {r.seq for r in out})

    def test_already_measured_are_excluded(self):
        seqs, e, nb, na = self._dataset(n=30)
        na[0] = na[-1]
        out = ev.rank_by_disagreement(
            seqs, e, nb, na, min_count=0, exclude=[seqs[0]], top_k=5
        )
        assert seqs[0] not in {r.seq for r in out}

    def test_diversity_filter_spreads_the_picks(self):
        """Without it, a measurement budget can be spent on one motif."""
        base = "ADEFGHIKLMNPQRS"
        # Ten near-identical sequences plus ten unrelated ones.
        near = [base[:i] + "W" + base[i + 1:] for i in range(10)]
        rng = np.random.default_rng(1)
        far = ["".join(rng.choice(list("ADEFGHIKLMNPQRSTVWY"), 15)) for _ in range(10)]
        seqs = near + far
        n = len(seqs)
        e = np.linspace(0, 1, n)
        nb = np.full(n, 1000.0)
        na = 1000.0 * np.exp(3.0 * e)       # inverted on purpose: everything disagrees

        greedy = ev.rank_by_disagreement(seqs, e, nb, na, min_count=0, top_k=5)
        spread = ev.rank_by_disagreement(
            seqs, e, nb, na, min_count=0, top_k=5, min_hamming=5
        )
        picked_near = sum(1 for r in spread if r.seq in near)
        assert len(spread) <= len(greedy)
        assert picked_near <= 1              # at most one from the near-identical cluster

    def test_gap_fraction_is_normalised(self):
        seqs, e, nb, na = self._dataset(n=40)
        na[0] = na[-1]
        out = ev.rank_by_disagreement(seqs, e, nb, na, min_count=0, top_k=5)
        assert all(0.0 <= r.gap_fraction <= 1.0 for r in out)

    def test_everything_filtered_returns_empty(self):
        seqs, e, nb, na = self._dataset(n=10)
        assert ev.rank_by_disagreement(seqs, e, nb, na, min_count=1e9) == []

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            ev.rank_by_disagreement(["AAA", "BBB"], [1.0], [10.0, 10.0], [5.0, 5.0])


class TestObjectiveIsSpearman:
    def test_spearman_is_labelled_the_objective(self):
        rng = np.random.default_rng(0)
        n = 80
        energy = rng.normal(size=n)
        before = rng.integers(100, 2000, size=n).astype(float)
        after = before * np.exp(-energy)
        m = ev.enrichment_metrics(energy, before, after, n_bootstrap=100)
        assert "THE OBJECTIVE" in m["enrichment.spearman"].note
        assert "diagnostic" in m["enrichment.weighted_pearson"].note
        assert "diagnostic" in m["enrichment.pearson"].note

    def test_spearman_survives_tail_compression(self):
        """Enrichment saturates at the extremes; Spearman should not care."""
        energy = np.linspace(-3, 3, 120)
        # Monotone but heavily compressed at both ends.
        le = np.tanh(-energy * 2.0)
        before = np.full(120, 1000.0)
        after = before * np.exp(le)

        m = ev.enrichment_metrics(energy, before, after, n_bootstrap=200)
        assert m["enrichment.spearman"].point > 0.99
        assert m["enrichment.pearson"].point < m["enrichment.spearman"].point


class TestReport:
    def test_build_report_combines_both_families(self):
        rng = np.random.default_rng(0)
        energy = rng.normal(size=60)
        kd = 10 ** (energy + rng.normal(scale=0.2, size=60))
        before = rng.integers(100, 1000, size=60).astype(float)
        after = before * np.exp(-energy)

        r = ev.build_report(
            "m1", energy,
            measured_affinity={"KD": kd},
            enrichment=(before, after),
            diagnostics={"final_loss": 1.23},
            seed=0,
            n_bootstrap=100,
        )
        assert "affinity.KD.spearman" in r.metrics
        assert "enrichment.weighted_pearson" in r.metrics
        assert r.diagnostics["final_loss"] == 1.23
        assert "m1" in r.summary()

    def test_get_lists_available_metrics_on_miss(self):
        r = ev.build_report("m", [1.0, 2.0, 3.0])
        with pytest.raises(KeyError, match="Available"):
            r.get("nope")
