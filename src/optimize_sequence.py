"""
optimize_sequence.py

Optimizes a given input sequence using hard (discrete) mode with no dropout.

Usage:
    python optimize_sequence.py --seq ACDEFGHIKLMNPQRSTVWY --ew 0.2 --iters 100
    python optimize_sequence.py --seq ACDEFGHIKLMNPQRSTVWY --ew 0.2 --iters 100 -o result.json
"""

import os
import json
import argparse

from colabdesign import mk_afdesign_model
from colabdesign.energy_model.model_3layer import make_energy_fn

from config import PDB, PARAMS_DIR, WEIGHTS_PATH, HOTSPOT_RESIDUES

OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
HOTSPOT_STR = ",".join([f"A{r}" for r in HOTSPOT_RESIDUES])


def optimize(model, raw_energy_fn, seq: str, energy_weight: float, iters: int) -> dict:
    """
    Single-stage optimization: hard (discrete), no dropout.
    Starts from the provided sequence.
    """
    model.restart(seq=seq)

    model.set_opt(soft=False, hard=True, dropout=False, temp=1e-6)
    model.opt["weights"]["energy"] = energy_weight
    for _ in range(iters):
        model.step()

    seq_probs  = model.aux["seq"]["pseudo"]
    raw_energy = float(raw_energy_fn(seq_probs))
    log        = model.aux["log"]
    loss_total = float(log["loss"])
    loss_af    = loss_total - energy_weight * raw_energy

    result = {
        "input_seq":  seq,
        "output_seq": model.get_seqs()[0],
        "loss_total": loss_total,
        "loss_af":    loss_af,
        "energy":     raw_energy,
    }
    for key in ("i_ptm", "ptm", "plddt"):
        if key in log:
            result[key] = float(log[key])

    return result


def main():
    parser = argparse.ArgumentParser(description="Optimize a sequence: hard mode, no dropout.")
    parser.add_argument("--seq",   required=True,
                        help="Input amino-acid sequence to optimize")
    parser.add_argument("--ew",    type=float, default=0.2,
                        help="Energy weight (default: 0.2)")
    parser.add_argument("--iters", type=int,   default=100,
                        help="Number of optimization iterations (default: 100)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON file (default: optimized_<seq[:8]>_ew<ew>_i<iters>.json)")
    args = parser.parse_args()

    out_file = args.output or os.path.join(
        OUT_DIR,
        f"optimized_{args.seq[:8]}_ew{args.ew:.2f}_i{args.iters}.json"
    )

    raw_energy_fn = make_energy_fn(WEIGHTS_PATH, energy_weight=1.0)

    model = mk_afdesign_model(
        protocol="binder",
        data_dir=PARAMS_DIR,
        energy_fn=raw_energy_fn,
        energy_weight=args.ew,
    )
    model.prep_inputs(
        pdb_filename=PDB,
        chain="A",
        binder_len=len(args.seq),
        hotspot=HOTSPOT_STR,
        rm_target_seq=False,
        model_names=["model_3", "model_4", "model_5"],
    )

    print(f"Optimizing seq={args.seq}  ew={args.ew}  iters={args.iters}", flush=True)
    r = optimize(model, raw_energy_fn, args.seq, args.ew, args.iters)

    print(
        f"  input ={r['input_seq']}\n"
        f"  output={r['output_seq']}\n"
        f"  loss={r['loss_total']:.4f}  loss_af={r['loss_af']:.4f}  "
        f"energy={r['energy']:.4f}  "
        f"i_ptm={r.get('i_ptm', float('nan')):.4f}"
    )

    protocol = {
        "soft": False, "hard": True, "dropout": False,
        "temp": 1e-6, "energy_weight": args.ew, "iters": args.iters,
    }
    with open(out_file, "w") as f:
        json.dump({"protocol": protocol, "result": r}, f, indent=2)
    print(f"\nSaved result to {out_file}")


if __name__ == "__main__":
    main()
