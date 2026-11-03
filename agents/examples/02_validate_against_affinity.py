"""
02_validate_against_affinity.py

Correlate the promoted model's predictions against measured binding affinities.
No Julia, no GPU.

This is the zero-cost validation worth running before any agent, any LLM and
any GPU time: it exercises the weight loader, the alphabet ordering and the
sign convention in one shot, against data the model is scored on rather than
numbers it produced itself.

Reading the sign
----------------
Kd is a dissociation constant: a LOW Kd means a strong binder.
The model emits an energy: a LOW energy means a good binder.
Both run in the same direction, so a well-behaved model gives a POSITIVE
Spearman correlation. A negative one means the sign convention is inverted
somewhere, which is a bug, not a weak model.

Note the caveat: these peptides overlap the model's training data, so this
measures whether the stack is wired correctly, not how well the model
generalises.

Usage:
    PYTHONPATH=. python agents/examples/02_validate_against_affinity.py
    PYTHONPATH=. python agents/examples/02_validate_against_affinity.py --register-as ref-2R
"""

import argparse
import csv
import math

from agents.core import registry, workspace as ws
from agents.core.schemas import ArchSpec, EvalMetrics, Layer, ModelEntry
from agents.core.tools import training

PROMOTED = "data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json"
AFFINITIES = "data/affinity_measurements_PA-PB1_formatted.csv"


def read_affinities(repo_relative: str) -> tuple[list[str], list[float]]:
    """Pull (sequence, Kd_nM) pairs out of the measurement table.

    ``analysis.read_seed_sequences`` cannot be used here: it takes the first
    whitespace token of each line, which on this CSV yields header and
    annotation text rather than sequences. A real CSV reader is required.
    """
    seqs: list[str] = []
    kds: list[float] = []
    with ws.read_path(repo_relative).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            seq = (row.get("Sequence") or "").strip().upper()
            raw = (row.get("Kd_nM") or "").strip()
            if not seq or not raw:
                continue
            try:
                kd = float(raw)
            except ValueError:
                continue
            if kd <= 0:
                continue
            seqs.append(seq)
            kds.append(kd)
    return seqs, kds


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the promoted model against measured Kd.")
    ap.add_argument("--weights", default=PROMOTED)
    ap.add_argument("--affinities", default=AFFINITIES)
    ap.add_argument("--register-as", default="promoted-PNB-2R-3lay",
                    help="registry name to give the promoted weights")
    args = ap.parse_args()

    seqs, kds = read_affinities(args.affinities)
    print(f"{len(seqs)} measured sequences from {args.affinities}")
    print(f"Kd range: {min(kds):.3f} - {max(kds):.1f} nM\n")

    # evaluate_model() works off the registry, so give the promoted weights an
    # entry. weights_path stays repo-relative: load_energy_model resolves an
    # existing repo-relative path through workspace.read_path, so the file is
    # read where it already lives and nothing is copied into the store.
    arch = ArchSpec(
        name=args.register_as,
        n_rounds=2,
        layers=[Layer(units=45, activation="relu"),
                Layer(units=15, activation="relu"),
                Layer(units=1, activation="identity")],
        likelihood="negbinom",
        seq_length=15,
        n_aa=21,
        notes="the promoted model the ColabDesign pipeline loads; registered here to evaluate it",
    )
    ws.ensure_store()
    registry.register_model(
        ModelEntry(
            name=args.register_as,
            arch=arch,
            metrics=EvalMetrics(n=0),
            weights_path=args.weights,
            n_params=arch.n_params(),
            promoted=True,
        ),
        overwrite=True,
    )

    metrics = training.evaluate_model(args.register_as, seqs, kds)
    print(f"Spearman (energy vs Kd): {metrics.spearman:+.4f}")
    print(f"Pearson  (energy vs Kd): {metrics.pearson:+.4f}")
    print(f"n: {metrics.n}")

    verdict = (
        "positive, as expected: low energy tracks low Kd"
        if (metrics.spearman or 0) > 0
        else "NEGATIVE -- the sign convention is inverted somewhere; investigate before trusting any energy"
    )
    print(f"\n{verdict}")

    # Pearson on raw Kd is dominated by the long tail; log Kd is the fairer read.
    log_kd = [math.log10(k) for k in kds]
    from agents.core.tools.training import _pearson, _spearman  # small, scipy-free helpers
    from agents.core.tools.fitness import predict_fitness

    preds = [s.energy for s in predict_fitness(seqs, args.register_as)]
    print(f"\nagainst log10(Kd):  Pearson {_pearson(preds, log_kd):+.4f}  "
          f"Spearman {_spearman(preds, log_kd):+.4f}")
    print("(Spearman is rank-based, so it is identical to the value above.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
