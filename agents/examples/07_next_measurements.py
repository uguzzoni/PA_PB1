"""Choosing which sequences to measure next, from model-enrichment disagreement.

    PYTHONPATH=. python agents/examples/07_next_measurements.py

The premise: the model pools information across sequences and imposes
structure, so it is often a *less* noisy estimator of binding than a single
sequence's enrichment. Where the two disagree, the model may be right -- so
those residuals are the most informative places to spend a scarce affinity
measurement, not errors to minimise.

Measuring the top predicted binders adds little: model and data already agree
there. Measuring a disagreement resolves a real conflict AND adds a point to
the affinity set, which is the metric the gate depends on and the one starved
of data.

Synthetic data, so it runs anywhere.
"""

from __future__ import annotations

import numpy as np

from agents.core import evaluation as ev

ALPHABET = "ADEFGHIKLMNPQRSTVWY"


def synthetic(n=400, seed=0):
    """A model that is right about the physics but disagrees with noisy counts."""
    rng = np.random.default_rng(seed)
    seqs = list({"".join(rng.choice(list(ALPHABET), 15)) for _ in range(n + 50)})[:n]
    n = len(seqs)

    true_energy = rng.normal(size=n)
    # The model is a good but imperfect estimator of the truth.
    model_energy = true_energy + rng.normal(scale=0.25, size=n)

    # Counts: uneven sequencing depth, and the selection follows the TRUTH.
    before = rng.lognormal(mean=5.0, sigma=1.2, size=n).round() + 1
    expected = before * np.exp(-1.5 * true_energy)
    after = rng.poisson(np.clip(expected, 0.1, 1e7)).astype(float)
    return seqs, model_energy, before, after


def main() -> int:
    seqs, energy, before, after = synthetic()
    already_measured = seqs[:35]          # the existing affinity set

    m = ev.enrichment_metrics(energy, before, after, min_count=10, n_bootstrap=400)
    print("Model vs enrichment")
    for metric in m.values():
        print(f"  {metric}")
    print(
        "\n(the Spearman is the objective; the Pearson pair is diagnostic --\n"
        " their gap says how much data sits where counting noise dominates)"
    )

    print(f"\n{'─' * 72}\nTop disagreements, excluding the {len(already_measured)} "
          f"already measured\n{'─' * 72}")

    picks = ev.rank_by_disagreement(
        seqs, energy, before, after,
        min_count=10,
        exclude=already_measured,
        top_k=8,
        min_hamming=4,                    # do not spend the budget on one motif
    )

    print(f"{'sequence':17s} {'E':>7s} {'logE':>7s} {'rank_m':>7s} {'rank_e':>7s} "
          f"{'gap':>6s}  direction")
    for r in picks:
        print(
            f"{r.seq:17s} {r.energy:+7.3f} {r.log_enrichment:+7.3f} "
            f"{r.rank_model:7d} {r.rank_enrichment:7d} {r.rank_gap:+6d}  {r.direction}"
        )

    under = [r for r in picks if r.direction == "model_underrates"]
    print(
        f"\n{len(under)} of {len(picks)} are 'model_underrates' -- the data ranks them "
        "better\nthan the model does. Those are candidate binders the model may be "
        "missing,\nand usually the more interesting half to measure."
    )

    # Contrast: what you would pick by simply taking the best predicted binders.
    best_first = [s for s in np.array(seqs)[np.argsort(energy)] if s not in set(already_measured)][:8]
    overlap = len(set(best_first) & {r.seq for r in picks})
    print(
        f"\nOverlap with 'just measure the top predicted binders': {overlap}/8.\n"
        "Those sequences are where model and data already agree, so measuring\n"
        "them confirms what you know rather than resolving anything."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
