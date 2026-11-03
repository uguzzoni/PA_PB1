"""
03_train_model.py

Train an energy model with PhageNegBinom.jl through the agent layer.

Requires Julia. The work happens in agents/julia/train.jl, which this script
reaches via agents.core.execution.LocalJuliaRunner: Python writes a config.json,
Julia writes a result.json, and both land in the run directory.

The trained weights go to agents/store/models/<name>/ and the model is added to
the registry. Nothing is written outside the store, and the ColabDesign
pipeline is untouched until a human promotes the model by hand.

The default --iters is deliberately tiny so a first run finishes in under a
minute. That produces a SMOKE ARTEFACT, not a trained model -- raise it by two
or three orders of magnitude for anything you intend to use.

Usage:
    PYTHONPATH=. python agents/examples/03_train_model.py                      # 40 iters, ~40 s
    PYTHONPATH=. python agents/examples/03_train_model.py --name real --iters 20000
    PYTHONPATH=. python agents/examples/03_train_model.py --name wide --units 60 20 1
"""

import argparse

from agents.core import registry
from agents.core.schemas import ArchSpec, DatasetSpec, Hyperparams, Layer
from agents.core.tools import architecture, training

MANIFEST = "agents/config/dataset_2R.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Train an energy model via PhageNegBinom.jl.")
    ap.add_argument("--name", default="doc-demo", help="registry name for the trained model")
    ap.add_argument("--manifest", default=MANIFEST,
                    help="repo-relative counts manifest, or a wide CSV")
    ap.add_argument("--units", type=int, nargs="+", default=[45, 15, 1],
                    help="layer widths; the last must be 1")
    ap.add_argument("--iters", type=int, default=40,
                    help="training iterations, split across the two stages")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing model of this name")
    args = ap.parse_args()

    acts = ["relu"] * (len(args.units) - 1) + ["identity"]
    arch = ArchSpec(
        name=args.name,
        n_rounds=2,
        layers=[Layer(units=u, activation=a) for u, a in zip(args.units, acts)],
        likelihood="negbinom",
        seq_length=15,
        n_aa=21,                      # gapped alphabet; see agents/USAGE.md
        regularization={"l2": args.l2},
        notes=f"example run, {args.iters} iterations",
    )

    # Cheap pre-flight: a bad path or an over-parameterised design fails here,
    # in milliseconds, instead of after Julia has spun up.
    check = architecture.validate_architecture(arch)
    print(f"architecture: {check['layer_sizes']}  input_dim={check['input_dim']}  "
          f"n_params={check['n_params']}")
    for warning in check["warnings"]:
        print(f"  warning: {warning}")

    dataset = DatasetSpec(counts_file=args.manifest,
                          val_fraction=args.val_fraction, split_seed=0)
    pre = training.prepare_dataset(dataset)
    if not pre["ok"]:
        print(f"dataset unusable: {pre['problems']}")
        return 1
    print(f"dataset: {args.manifest}\n")

    hyper = Hyperparams(epochs=args.iters, learning_rate=args.lr,
                        batch_size=args.batch_size,
                        early_stopping_patience=0, seed=0)

    print(f"training {args.name!r} -- this calls Julia, first run pays precompilation")
    result = training.train_model(arch, dataset, args.name, hyper, overwrite=args.overwrite)

    print(f"\nval_loss    : {result.metrics.val_loss:,.2f}  (on {result.metrics.n} held-out sequences)")
    print(f"n_params    : {result.n_params}")
    print(f"weights     : {result.weights_path}  (relative to agents/store/)")
    print(f"duration    : {result.duration_s}s")

    if args.iters < 1000:
        print(f"\n{args.iters} iterations is a plumbing check, not a trained model.")
    print(f"\nto use it elsewhere in the pipeline:\n{registry.promotion_instructions(args.name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
