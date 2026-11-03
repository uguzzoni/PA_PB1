"""Metrics with confidence intervals, and the evaluation report.

Design rules, all of which exist to stop a search procedure from chasing noise:

1. **Every metric carries a confidence interval.** A point estimate alone is
   not comparable to another point estimate. With ~35 measured affinities a
   Spearman has a 95% interval roughly ±0.3 wide; differences smaller than
   that are not differences.
2. **Higher is always better.** Metrics whose natural sign runs the other way
   are oriented at construction, and the raw value is kept alongside for
   diagnosis. Uniform orientation is what lets the objective machinery in
   ``objective.py`` compare tiers without special cases.
3. **No LLM computes a metric.** This module is deterministic and is the only
   thing that produces numbers the search is allowed to act on.

No scipy: the correlations and the bootstrap are implemented here so the
agent layer keeps its dependency surface to numpy + pydantic.

Sign conventions in this codebase
---------------------------------
The model emits an **energy**: lower = better binder.

* **Affinity.** Low Kd = strong binder. Energy and log-Kd therefore run in the
  *same* direction, so a positive correlation is the healthy outcome and no
  orientation flip is applied.
* **Enrichment.** High enrichment = strong binder, so energy and
  log-enrichment run in *opposite* directions. The raw correlation is
  negative when the model works, and is flipped so that higher is better.
  Getting this backwards silently inverts the whole search.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MetricWithCI",
    "pearson",
    "spearman",
    "weighted_pearson",
    "bootstrap_ci",
    "correlate",
    "log_enrichment",
    "enrichment_weights",
    "affinity_metrics",
    "enrichment_metrics",
    "Disagreement",
    "rank_by_disagreement",
    "EvaluationReport",
]

CorrMethod = Literal["pearson", "spearman", "weighted_pearson"]


# ==========================================================================
# The metric container
# ==========================================================================
class MetricWithCI(BaseModel):
    """A single number the search is allowed to act on, with its uncertainty.

    ``point`` is always oriented so that **higher is better**. ``raw_point``
    keeps the value before orientation, because the raw sign is what you read
    when diagnosing a wiring bug.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    point: float = Field(..., description="Oriented estimate. Higher is better.")
    ci_low: float
    ci_high: float
    n: int = Field(..., description="Number of observations the estimate rests on.")
    raw_point: float = Field(..., description="Before orientation; sign is diagnostic.")
    n_bootstrap: int = 0
    note: str = ""

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    def beats(self, other: "MetricWithCI") -> bool:
        """True only if this metric is better beyond the overlap of intervals.

        Deliberately strict: ``point > other.point`` is not evidence of
        anything when the intervals overlap.
        """
        return self.ci_low > other.point

    def ties_with(self, other: "MetricWithCI") -> bool:
        """True when the two intervals overlap at all."""
        return not (self.ci_low > other.ci_high or other.ci_low > self.ci_high)

    def __str__(self) -> str:
        return f"{self.name}={self.point:+.4f} [{self.ci_low:+.4f},{self.ci_high:+.4f}] n={self.n}"


# ==========================================================================
# Correlation primitives
# ==========================================================================
def _clean(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = _clean(x, y)
    if a.size < 3:
        return float("nan")
    da, db = a - a.mean(), b - b.mean()
    den = math.sqrt(float(da @ da) * float(db @ db))
    return float(da @ db) / den if den > 0 else float("nan")


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not bias the correlation."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = _clean(x, y)
    if a.size < 3:
        return float("nan")
    return pearson(_ranks(a), _ranks(b))


def weighted_pearson(
    x: Sequence[float], y: Sequence[float], w: Sequence[float]
) -> float:
    """Pearson with per-observation weights.

    Needed for enrichment, where the variance of each point depends on read
    depth. An unweighted correlation there is dominated by the noisiest
    sequences.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ww = np.asarray(w, dtype=float)
    if not (a.shape == b.shape == ww.shape):
        raise ValueError(f"shape mismatch: {a.shape}, {b.shape}, {ww.shape}")
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(ww) & (ww > 0)
    a, b, ww = a[ok], b[ok], ww[ok]
    if a.size < 3:
        return float("nan")

    ww = ww / ww.sum()
    ma, mb = float(ww @ a), float(ww @ b)
    da, db = a - ma, b - mb
    cov = float(ww @ (da * db))
    va, vb = float(ww @ (da * da)), float(ww @ (db * db))
    den = math.sqrt(va * vb)
    return cov / den if den > 0 else float("nan")


# ==========================================================================
# Bootstrap
# ==========================================================================
def bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    n_obs: int,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, int]:
    """Percentile bootstrap over observation indices.

    ``stat_fn`` receives an index array and returns the statistic for that
    resample. Resamples that produce a degenerate statistic (zero variance,
    all ties) are dropped rather than counted as zero, and the number actually
    used is returned so a thin bootstrap is visible rather than silent.
    """
    if n_obs < 3:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_obs, size=n_obs)
        v = stat_fn(idx)
        if np.isfinite(v):
            vals.append(float(v))
    if len(vals) < 20:
        return float("nan"), float("nan"), len(vals)
    arr = np.asarray(vals)
    return (
        float(np.quantile(arr, alpha / 2)),
        float(np.quantile(arr, 1 - alpha / 2)),
        len(vals),
    )


def correlate(
    x: Sequence[float],
    y: Sequence[float],
    *,
    name: str,
    method: CorrMethod = "spearman",
    weights: Sequence[float] | None = None,
    orient: Literal[1, -1] = 1,
    n_bootstrap: int = 2000,
    seed: int = 0,
    note: str = "",
) -> MetricWithCI:
    """Correlation with a bootstrap interval, oriented so higher is better.

    ``orient=-1`` flips the sign: use it when the model works *against* the
    observable, as with energy versus log-enrichment.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float) if weights is not None else None

    if method == "weighted_pearson" and w is None:
        raise ValueError("method='weighted_pearson' requires weights")

    def stat(idx: np.ndarray) -> float:
        if method == "pearson":
            return pearson(a[idx], b[idx])
        if method == "spearman":
            return spearman(a[idx], b[idx])
        return weighted_pearson(a[idx], b[idx], w[idx])  # type: ignore[index]

    full = np.arange(a.size)
    raw = stat(full)
    lo, hi, used = bootstrap_ci(stat, a.size, n_bootstrap=n_bootstrap, seed=seed)

    # Orientation flips the interval and swaps its ends.
    point = orient * raw
    if orient == -1 and np.isfinite(lo) and np.isfinite(hi):
        lo, hi = -hi, -lo

    return MetricWithCI(
        name=name,
        point=float(point),
        ci_low=float(lo),
        ci_high=float(hi),
        n=int(a.size),
        raw_point=float(raw),
        n_bootstrap=used,
        note=note,
    )


# ==========================================================================
# Enrichment
# ==========================================================================
def log_enrichment(
    counts_before: Sequence[float],
    counts_after: Sequence[float],
    *,
    pseudocount: float = 0.5,
    normalise: bool = True,
) -> np.ndarray:
    """log((n_after + c) / (n_before + c)), optionally depth-normalised.

    The pseudocount keeps zero counts finite. Depth normalisation divides each
    side by its library total, so enrichment is not confounded by how deeply
    each round was sequenced.
    """
    nb = np.asarray(counts_before, dtype=float)
    na = np.asarray(counts_after, dtype=float)
    if nb.shape != na.shape:
        raise ValueError(f"length mismatch: {nb.shape} vs {na.shape}")
    fb, fa = nb + pseudocount, na + pseudocount
    if normalise:
        fb = fb / fb.sum()
        fa = fa / fa.sum()
    return np.log(fa / fb)


def enrichment_weights(
    counts_before: Sequence[float], counts_after: Sequence[float], *, pseudocount: float = 0.5
) -> np.ndarray:
    """Inverse-variance weights for log-enrichment, from the delta method.

    For Poisson counts, ``Var[log(n_after / n_before)] ≈ 1/n_after + 1/n_before``,
    so the inverse-variance weight is::

        w = 1 / (1/n_after + 1/n_before)

    A sequence seen three times and a sequence seen three thousand times carry
    wildly different information about enrichment; weighting is what stops the
    former from dominating the correlation.
    """
    nb = np.asarray(counts_before, dtype=float) + pseudocount
    na = np.asarray(counts_after, dtype=float) + pseudocount
    return 1.0 / (1.0 / nb + 1.0 / na)


def enrichment_metrics(
    predicted_energy: Sequence[float],
    counts_before: Sequence[float],
    counts_after: Sequence[float],
    *,
    min_count: float = 0.0,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, MetricWithCI]:
    """Agreement between predicted energy and measured log-enrichment.

    **Spearman is the objective; the Pearsons are diagnostics.** Three
    reasons, none of them merely convenience:

    * Enrichment is compressed at both ends -- a sequence cannot exceed the
      whole library, nor fall below zero counts -- so its relationship with
      energy is monotone but decidedly nonlinear in the tails. Pearson is
      misspecified there by construction.
    * Ranks bound each observation's leverage. A low-count sequence with a
      noisy enrichment gets the wrong *rank*, but cannot become the
      high-leverage outlier that wrecks a Pearson. The heteroscedasticity is
      still there; its worst effect is contained without modelling it.
    * No dispersion parameter, no negative-binomial assumption, no delta
      method. The weighted Pearson below needs all three to be exactly right;
      Spearman needs none of them.

    The weighted/unweighted Pearson pair is kept because the gap between them
    is informative: it says how much of the data sits in the regime where
    counting noise still dominates. If they agree, depth is not the issue.

    Energy and enrichment run in opposite directions, so all three are
    oriented (``orient=-1``) and a positive result means the model works.
    """
    e = np.asarray(predicted_energy, dtype=float)
    nb = np.asarray(counts_before, dtype=float)
    na = np.asarray(counts_after, dtype=float)

    keep = (nb >= min_count) | (na >= min_count) if min_count > 0 else np.ones(e.size, bool)
    e, nb, na = e[keep], nb[keep], na[keep]
    dropped = int((~keep).sum())

    le = log_enrichment(nb, na)
    w = enrichment_weights(nb, na)
    note = f"min_count={min_count}, dropped={dropped}" if dropped else ""

    return {
        "enrichment.spearman": correlate(
            e, le, name="enrichment.spearman", method="spearman", orient=-1,
            n_bootstrap=n_bootstrap, seed=seed,
            note=(note + " | THE OBJECTIVE: rank-based, no distributional assumption").strip(" |"),
        ),
        "enrichment.weighted_pearson": correlate(
            e, le, name="enrichment.weighted_pearson", method="weighted_pearson",
            weights=w, orient=-1, n_bootstrap=n_bootstrap, seed=seed,
            note=(note + " | diagnostic: inverse-variance weighted (Poisson delta method)").strip(" |"),
        ),
        "enrichment.pearson": correlate(
            e, le, name="enrichment.pearson", method="pearson", orient=-1,
            n_bootstrap=n_bootstrap, seed=seed,
            note=(note + " | diagnostic: gap vs the weighted one measures depth-driven noise").strip(" |"),
        ),
    }


# ==========================================================================
# Affinity
# ==========================================================================
def affinity_metrics(
    predicted_energy: Sequence[float],
    measured: dict[str, Sequence[float]],
    *,
    log_transform: bool = True,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, MetricWithCI]:
    """Agreement with measured affinity, one entry per measure.

    ``measured`` maps a measure name to its values, e.g.
    ``{"KD": [...], "koff": [...], "kon": [...]}``. They are kept separate on
    purpose: phage display with washing steps is kinetically biased, so which
    measure the model reproduces best says what the selection actually
    captured. Collapsing them into one number throws that signal away.

    Values are log-transformed by default: these quantities span orders of
    magnitude. Spearman is reported because the relationship with energy is
    reliably monotone but not reliably linear.

    No orientation flip: low Kd and low energy both mean a good binder, so a
    positive correlation is the healthy outcome. A negative one is a
    sign-convention bug, not a weak model.
    """
    e = np.asarray(predicted_energy, dtype=float)
    out: dict[str, MetricWithCI] = {}

    for measure, values in measured.items():
        v = np.asarray(values, dtype=float)
        if v.shape != e.shape:
            raise ValueError(f"{measure}: {v.shape} values for {e.shape} predictions")
        if log_transform:
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.log10(np.where(v > 0, v, np.nan))
        # kon runs the other way: a HIGH kon is a good binder, so it is
        # oriented like enrichment rather than like Kd.
        orient: Literal[1, -1] = -1 if measure.lower() == "kon" else 1
        out[f"affinity.{measure}.spearman"] = correlate(
            e, v, name=f"affinity.{measure}.spearman", method="spearman",
            orient=orient, n_bootstrap=n_bootstrap, seed=seed,
            note="log10" if log_transform else "",
        )
    return out


# ==========================================================================
# Where to spend the next measurement
# ==========================================================================
class Disagreement(BaseModel):
    """One sequence on which the model and the enrichment data disagree."""

    model_config = ConfigDict(extra="forbid")

    seq: str
    energy: float
    log_enrichment: float
    rank_model: int = Field(..., description="1 = best predicted binder (lowest energy).")
    rank_enrichment: int = Field(..., description="1 = most enriched.")
    rank_gap: int = Field(..., description="rank_model - rank_enrichment.")
    gap_fraction: float = Field(..., description="|rank_gap| / n, so sets of different size compare.")
    direction: Literal["model_underrates", "model_overrates"]
    counts_before: float
    counts_after: float


def rank_by_disagreement(
    sequences: Sequence[str],
    predicted_energy: Sequence[float],
    counts_before: Sequence[float],
    counts_after: Sequence[float],
    *,
    min_count: float = 10.0,
    exclude: Sequence[str] = (),
    top_k: int = 20,
    min_hamming: int = 0,
    direction: Literal["model_underrates", "model_overrates"] | None = None,
) -> list[Disagreement]:
    """Sequences where model and enrichment disagree most, as measurement candidates.

    The premise is the one that makes this worth doing: the model pools
    information across sequences and imposes structure, so it is often a
    *less* noisy estimator of binding than a single sequence's enrichment.
    Where the two disagree, the model may well be right -- so those residuals
    are not errors to minimise, they are the most informative places to spend
    a scarce affinity measurement.

    Measuring the top predicted binders tells you little: the model and the
    data already agree there. Measuring a disagreement resolves a real
    conflict, and it adds a point to the affinity set, which is the metric
    the gate depends on and the one starved of data.

    Ranks are used rather than values because the two quantities are on
    different scales with a monotone, nonlinear relationship -- the same
    reason Spearman is the objective.

    Parameters
    ----------
    min_count:
        Drop sequences seen fewer than this many times in the *input* library.
        Below it the enrichment is noise, and a disagreement there is
        expected rather than informative.
    exclude:
        Sequences already measured. No point resolving a conflict twice.
    min_hamming:
        Greedy diversity filter over the selected set. Without it the top
        disagreements are frequently near-identical, and you would spend the
        whole measurement budget on one motif.
    direction:
        Restrict to one error type. ``model_underrates`` -- the data says
        better than the model does -- is usually the more interesting half:
        those are candidate binders the model is missing.
    """
    seqs = list(sequences)
    e = np.asarray(predicted_energy, dtype=float)
    nb = np.asarray(counts_before, dtype=float)
    na = np.asarray(counts_after, dtype=float)
    if not (len(seqs) == e.size == nb.size == na.size):
        raise ValueError(
            f"length mismatch: {len(seqs)} sequences, {e.size} energies, "
            f"{nb.size}/{na.size} counts"
        )

    banned = set(exclude)
    keep = np.array(
        [(nb[i] >= min_count) and (seqs[i] not in banned) and np.isfinite(e[i])
         for i in range(len(seqs))]
    )
    if not keep.any():
        return []

    seqs = [s for s, k in zip(seqs, keep) if k]
    e, nb, na = e[keep], nb[keep], na[keep]
    le = log_enrichment(nb, na)

    n = len(seqs)
    rank_model = np.empty(n, dtype=int)
    rank_model[np.argsort(e, kind="mergesort")] = np.arange(1, n + 1)        # low energy = rank 1
    rank_enr = np.empty(n, dtype=int)
    rank_enr[np.argsort(-le, kind="mergesort")] = np.arange(1, n + 1)        # high enrichment = rank 1

    gap = rank_model - rank_enr
    records = [
        Disagreement(
            seq=seqs[i],
            energy=float(e[i]),
            log_enrichment=float(le[i]),
            rank_model=int(rank_model[i]),
            rank_enrichment=int(rank_enr[i]),
            rank_gap=int(gap[i]),
            gap_fraction=round(abs(int(gap[i])) / n, 4),
            direction="model_underrates" if gap[i] > 0 else "model_overrates",
            counts_before=float(nb[i]),
            counts_after=float(na[i]),
        )
        for i in range(n)
    ]
    if direction is not None:
        records = [r for r in records if r.direction == direction]
    records.sort(key=lambda r: abs(r.rank_gap), reverse=True)

    if min_hamming > 0:
        chosen: list[Disagreement] = []
        for r in records:
            if all(
                sum(a != b for a, b in zip(r.seq, c.seq)) >= min_hamming
                for c in chosen
                if len(c.seq) == len(r.seq)
            ):
                chosen.append(r)
            if len(chosen) >= top_k:
                break
        return chosen

    return records[:top_k]


# ==========================================================================
# The report
# ==========================================================================
class EvaluationReport(BaseModel):
    """Everything measured about one trained model, in one object.

    One report per experiment, produced by this module alone. Because the full
    set of metrics is always computed, no part of the search can quietly
    choose which number to look at after seeing the results -- the choice is
    made in advance by the ObjectiveSpec.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    metrics: dict[str, MetricWithCI] = Field(default_factory=dict)
    diagnostics: dict[str, float] = Field(
        default_factory=dict,
        description="Convergence and fit diagnostics: final loss, train/val gap, wall time.",
    )
    seed: int | None = None
    note: str = ""

    def get(self, name: str) -> MetricWithCI:
        if name not in self.metrics:
            raise KeyError(f"no metric {name!r}. Available: {sorted(self.metrics)}")
        return self.metrics[name]

    def summary(self) -> str:
        lines = [f"{self.model_name} (seed={self.seed})"]
        lines += [f"  {m}" for m in self.metrics.values()]
        if self.diagnostics:
            lines.append("  diagnostics: " + ", ".join(
                f"{k}={v:.4g}" for k, v in self.diagnostics.items()
            ))
        return "\n".join(lines)


def build_report(
    model_name: str,
    predicted_energy: Sequence[float],
    *,
    measured_affinity: dict[str, Sequence[float]] | None = None,
    enrichment: tuple[Sequence[float], Sequence[float]] | None = None,
    enrichment_energy: Sequence[float] | None = None,
    diagnostics: dict[str, float] | None = None,
    seed: int | None = None,
    min_count: float = 0.0,
    n_bootstrap: int = 2000,
    rng_seed: int = 0,
) -> EvaluationReport:
    """Assemble a full report.

    ``predicted_energy`` aligns with the affinity measurements.
    ``enrichment_energy`` aligns with the count arrays, which are usually a
    much larger and different set of sequences; it falls back to
    ``predicted_energy`` when the two sets coincide.
    """
    metrics: dict[str, MetricWithCI] = {}

    if measured_affinity:
        metrics.update(
            affinity_metrics(
                predicted_energy, measured_affinity,
                n_bootstrap=n_bootstrap, seed=rng_seed,
            )
        )
    if enrichment is not None:
        nb, na = enrichment
        metrics.update(
            enrichment_metrics(
                enrichment_energy if enrichment_energy is not None else predicted_energy,
                nb, na, min_count=min_count, n_bootstrap=n_bootstrap, seed=rng_seed,
            )
        )

    return EvaluationReport(
        model_name=model_name,
        metrics=metrics,
        diagnostics=diagnostics or {},
        seed=seed,
    )
