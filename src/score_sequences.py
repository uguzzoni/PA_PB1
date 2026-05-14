"""
score_sequences.py

AF2 scoring (no optimization) for one or more sequences using colabdesign.

For each sequence performs a single forward pass with hard=True, dropout=False
to score the exact input sequence and reports AF2 confidence metrics plus MLP
energy. No sequence modification — the input sequence is scored as-is.

Metrics reported: seq, i_ptm, ptm, plddt, energy

Usage:
    python score_sequences.py --seq ACDEFGHIKLMNPQRST
    python score_sequences.py --seq ACDEFGHIKLMNPQRST -o result.json
    python score_sequences.py --seq_file sequences.txt -o scores_af2.json
"""

import os
import json
import argparse

from colabdesign import mk_afdesign_model
from colabdesign.energy_model.model_3layer import make_energy_fn

# ── CONFIG ────────────────────────────────────────────────────────────────────
from config import PDB, PARAMS_DIR, WEIGHTS_PATH, HOTSPOT_RESIDUES
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
HOTSPOT_STR = ",".join([f"A{r}" for r in HOTSPOT_RESIDUES])
# ─────────────────────────────────────────────────────────────────────────────


def score_one(model, raw_energy_fn, seq: str) -> dict:
    """
    Single AF2 forward pass for `seq`.

    Initialises from the one-hot sequence, sets hard=True/dropout=False,
    then runs a forward pass (no gradient update).  Reads metrics from
    model.aux["log"] and computes MLP energy from model.aux["seq"]["pseudo"].
    """
    model.restart(seq=seq, seed=0)
    model.set_opt(soft=False, hard=True, dropout=False, temp=1e-6)
    model.run(backprop=False)

    seq_probs  = model.aux["seq"]["pseudo"]
    raw_energy = float(raw_energy_fn(seq_probs))
    log        = model.aux["log"]

    result = {"seq": seq, "energy": raw_energy}
    for key in ("i_ptm", "ptm", "plddt"):
        if key in log:
            result[key] = float(log[key])
    return result


def load_sequences(seq_file: str) -> list:
    seqs = []
    with open(seq_file) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                seqs.append(s)
    if not seqs:
        raise ValueError(f"No sequences found in {seq_file}")
    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        raise ValueError(
            f"All sequences must have the same length; found: {sorted(lengths)}"
        )
    return seqs


def main():
    parser = argparse.ArgumentParser(
        description="Score sequence(s) with AF2 (colabdesign) + MLP energy."
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--seq",      default=None, help="Single sequence to score")
    grp.add_argument("--seq_file", default=None,
                     help="Text file with one sequence per line (# = comment)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON file (default: af2_scores.json)")
    args = parser.parse_args()

    if args.seq_file:
        sequences = load_sequences(args.seq_file)
        src_label = os.path.basename(args.seq_file)
    else:
        sequences = [args.seq]
        src_label = args.seq

    out_file = args.output or os.path.join(OUT_DIR, "af2_scores.json")

    binder_len = len(sequences[0])
    raw_energy_fn = make_energy_fn(WEIGHTS_PATH, energy_weight=1.0)
    model = mk_afdesign_model(
        protocol="binder",
        data_dir=PARAMS_DIR,
        energy_fn=raw_energy_fn,
        energy_weight=0.0,
    )
    model.prep_inputs(
        pdb_filename=PDB,
        chain="A",
        binder_len=binder_len,
        hotspot=HOTSPOT_STR,
        rm_target_seq=False,
        model_names=["model_3", "model_4", "model_5"],
    )

    results = []
    for i, seq in enumerate(sequences):
        print(f"[{i+1}/{len(sequences)}] scoring {seq} ...", flush=True)
        r = score_one(model, raw_energy_fn, seq)
        r["idx"] = i
        results.append(r)
        print(
            f"  i_ptm={r.get('i_ptm', float('nan')):.4f}  "
            f"ptm={r.get('ptm', float('nan')):.4f}  "
            f"plddt={r.get('plddt', float('nan')):.4f}  "
            f"energy={r['energy']:.4f}"
        )

    output = {"source": src_label, "results": results}
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} scores to {out_file}")


if __name__ == "__main__":
    main()
