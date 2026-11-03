"""
01_score_promoted_model.py

Score peptides with the promoted energy model. No Julia, no GPU, no training.

Loads the weights the ColabDesign pipeline actually uses --
data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json -- and
scores the known binders in data/seqs_aff_alberto_darren.txt against a random
baseline. Lower energy means a better predicted binder.

Start here. If this runs, the weight loader, the alphabet ordering and the sign
convention are all working, and everything else in agents/ builds on them.

Usage:
    PYTHONPATH=. python agents/examples/01_score_promoted_model.py
    PYTHONPATH=. python agents/examples/01_score_promoted_model.py --n-random 500
"""

import argparse

import numpy as np

from agents.core import workspace as ws
from agents.core.schemas import AA_ALPHABET
from agents.core.tools import analysis, fitness

PROMOTED = "data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json"
KNOWN_BINDERS = "data/seqs_aff_alberto_darren.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--weights", default=PROMOTED, help="repo-relative weights JSON")
    ap.add_argument("--seqs", default=KNOWN_BINDERS, help="repo-relative sequence file")
    ap.add_argument("--n-random", type=int, default=200, help="random sequences for the baseline")
    args = ap.parse_args()

    model = fitness.load_energy_model(args.weights, seq_length=15, name="promoted")

    # n_aa is a property of the loaded model, never an assumption. The campaign's
    # models are 21 wide: 20 residues plus a gap column a designed binder never
    # occupies. Assuming 20 here is the classic way to get confident nonsense.
    print(f"model      : {args.weights}")
    print(f"layers     : {[w.shape for w, _ in model.layers]}")
    print(f"activations: {model.activations}")
    print(f"n_aa       : {model.n_aa}  (21 = gapped alphabet)")
    print(f"input_dim  : {model.input_dim}\n")

    known = analysis.read_seed_sequences(args.seqs)
    energies = model.energy(fitness.encode_onehot(known, 15, model.n_aa))

    print(f"known binders from {args.seqs}  (n={len(known)}, lower is better)")
    for seq, e in sorted(zip(known, energies), key=lambda t: t[1]):
        print(f"  {seq}  {e:+.4f}")

    rng = np.random.default_rng(0)
    random_seqs = ["".join(rng.choice(list(AA_ALPHABET), 15)) for _ in range(args.n_random)]
    baseline = model.energy(fitness.encode_onehot(random_seqs, 15, model.n_aa))

    print(f"\nknown binders  mean {energies.mean():+.4f}   best {energies.min():+.4f}")
    print(f"random (n={args.n_random:<4d}) mean {baseline.mean():+.4f}   best {baseline.min():+.4f}")
    print(
        "\nA model that has learned something scores the known binders below the\n"
        "random baseline. Read this as a sanity check on the loader, not as\n"
        "evidence about the model: these sequences were in its training data."
    )
    print(f"\nrepo root: {ws.REPO_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
