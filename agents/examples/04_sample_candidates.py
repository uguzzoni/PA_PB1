"""
04_sample_candidates.py

Generate candidate binders from a trained model, then assess and save them.

Three samplers are available, and the routing differs:

    mcmc      Metropolis over point mutations. Robust default. Runs on the
              vectorised numpy fast path unless --prefer-julia is given.
    greedy    First-improvement descent. Converges hard; least diverse.
    gradient  Descent on soft logits, then discretise. Needs a differentiable
              model, so it exists only in Julia.

greedy and gradient route to agents/julia/sample.jl automatically and therefore
need Julia. mcmc does not.

Fitness alone is not success: a sampler optimising energy will happily collapse
onto one motif, so diversity is checked here alongside it.

Usage:
    PYTHONPATH=. python agents/examples/04_sample_candidates.py --model doc-demo
    PYTHONPATH=. python agents/examples/04_sample_candidates.py --model doc-demo --method gradient
    PYTHONPATH=. python agents/examples/04_sample_candidates.py --model doc-demo --compare
"""

import argparse

import numpy as np

from agents.core import registry
from agents.core.schemas import AA_ALPHABET, SampleConfig, SeqConstraints
from agents.core.tools import analysis, fitness

METHODS = ("mcmc", "greedy", "gradient")


def random_baseline(model, n: int, length: int, forbidden: str, seed: int = 0) -> float:
    alphabet = [a for a in AA_ALPHABET if a not in forbidden.upper()]
    rng = np.random.default_rng(seed)
    seqs = ["".join(rng.choice(alphabet, length)) for _ in range(n)]
    return float(np.mean(model.energy(fitness.encode_onehot(seqs, length, model.n_aa))))


def run_one(method: str, args, model) -> list:
    config = SampleConfig(
        model_name=args.model,
        n=args.n,
        length=args.length,
        method=method,
        objective="minimize",            # the model is a cost: lower is better
        temperature=args.temperature,
        n_steps=args.steps,
        seed=args.seed,
        constraints=SeqConstraints(
            forbidden_aa=args.forbidden,
            # 0-based positions. Pin anchors here when you know them.
            fixed_positions=dict(args.fixed) if args.fixed else {},
        ),
    )
    result = fitness.sample_sequences(config, prefer_julia=args.prefer_julia)
    where = "julia" if (args.prefer_julia or method != "mcmc") else "numpy"
    acc = result.accepted_fraction
    print(f"\n--- {method} ({where})"
          + (f"  acceptance={acc:.3f}" if acc is not None else ""))
    return result.sequences


def report(records, model, baseline: float, args) -> None:
    energies = [r.energy for r in records]
    print(f"    mean {np.mean(energies):+.4f}   best {min(energies):+.4f}   "
          f"random baseline {baseline:+.4f}")

    div = analysis.diversity_metrics(records_to_seqs(records))
    print(f"    unique {div.n_unique}/{div.n}   clusters {div.n_clusters}   "
          f"mean_hamming {div.mean_hamming}   entropy {div.mean_position_entropy} bits")
    if div.n_clusters <= 1 or div.mean_hamming < 3.0:
        print("    COLLAPSED -- raise --temperature or --steps")

    for r in analysis.rank_sequences(records, top_k=5):
        print(f"      {r.seq}  {r.energy:+.4f}")


def records_to_seqs(records) -> list[str]:
    return [r.seq for r in records]


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample candidate binders from a trained model.")
    ap.add_argument("--model", default="doc-demo", help="registered model name")
    ap.add_argument("--method", choices=METHODS, default="mcmc")
    ap.add_argument("--compare", action="store_true", help="run all three methods")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--length", type=int, default=15)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--forbidden", default="C",
                    help="residues that may never appear; C avoids spurious disulfides")
    ap.add_argument("--fixed", nargs="*", default=None, metavar="POS:AA",
                    type=lambda s: (int(s.split(":")[0]), s.split(":")[1]),
                    help="0-based anchors, e.g. --fixed 0:M 14:K")
    ap.add_argument("--prefer-julia", action="store_true",
                    help="send mcmc through PhageMC.jl instead of the numpy path")
    ap.add_argument("--save-as", default=None, help="save the shortlist under this name")
    args = ap.parse_args()

    entry = registry.get_model(args.model)
    model = fitness.load_energy_model(entry.weights_path,
                                      seq_length=entry.arch.seq_length, name=entry.name)
    print(f"model {entry.name}  ({entry.n_params} params, n_aa={model.n_aa})")
    baseline = random_baseline(model, 200, args.length, args.forbidden)

    methods = METHODS if args.compare else (args.method,)
    best: list = []
    for method in methods:
        records = run_one(method, args, model)
        report(records, model, baseline, args)
        best.extend(records)

    # Drop duplicates and anything violating the constraints, then shortlist.
    kept = analysis.filter_sequences(
        best,
        SeqConstraints(forbidden_aa=args.forbidden,
                       fixed_positions=dict(args.fixed) if args.fixed else {}),
    )
    shortlist = analysis.rank_sequences(kept, top_k=10)
    print(f"\nshortlist: {len(shortlist)} of {len(best)} sampled "
          f"({len(best) - len(kept)} dropped as duplicate or non-compliant)")

    if args.save_as:
        paths = analysis.save_sequences(shortlist, args.save_as)
        print(f"saved: {paths}")
        print("note: load_sequences() reads the JSON; the CSV is for humans and Excel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
