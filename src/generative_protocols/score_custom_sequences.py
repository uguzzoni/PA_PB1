"""
score_custom_sequences.py

AF2 scoring (single forward pass, no optimisation) for a list of input sequences.
Uses the same PDB, AF2 params, energy weights, hotspot residues, and model ensemble
as the generate/optimize scripts.

Output JSON is written to  <RESULTS_DIR>/custom/<output_name>.json
and follows the same {"protocol": {...}, "results": [...]} schema used by all
other ColabDesign scripts, so it is picked up automatically by the notebooks.

Usage:
    python score_custom_sequences.py --seq_file my_seqs.txt
    python score_custom_sequences.py --seq_file my_seqs.txt -o my_run
"""

import os
import json
import argparse
import pathlib

from colabdesign import mk_afdesign_model
from colabdesign.energy_model.model_3layer import make_energy_fn

from config import PDB, PARAMS_DIR, WEIGHTS_PATH, HOTSPOT_RESIDUES, RESULTS_DIR

HOTSPOT_STR = ",".join([f"A{r}" for r in HOTSPOT_RESIDUES])
MODEL_NAMES = ["model_3", "model_4", "model_5"]


def load_sequences(seq_file: str) -> list:
    seqs = []
    with open(seq_file) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                seqs.append(s.upper())
    if not seqs:
        raise ValueError(f"No sequences found in {seq_file}")
    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        raise ValueError(
            f"All sequences must have the same length; found: {sorted(lengths)}"
        )
    return seqs


def score_one(model, raw_energy_fn, seq: str) -> dict:
    """Single AF2 forward pass for `seq` — no gradient update."""
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score custom sequences with AF2 (colabdesign) + MLP energy. "
            "Output written to <RESULTS_DIR>/custom/<name>.json"
        )
    )
    parser.add_argument(
        "--seq_file", required=True,
        help="Text file with one sequence per line (lines starting with # are skipped)",
    )
    parser.add_argument(
        "-o", "--output_name", default=None,
        help="Output JSON stem (default: input filename stem)",
    )
    args = parser.parse_args()

    sequences = load_sequences(args.seq_file)
    out_stem  = args.output_name or pathlib.Path(args.seq_file).stem
    out_dir   = pathlib.Path(RESULTS_DIR) / "custom"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file  = out_dir / f"{out_stem}.json"

    print(f"Scoring {len(sequences)} sequences  →  {out_file}\n")

    binder_len    = len(sequences[0])
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
        model_names=MODEL_NAMES,
    )

    protocol = {
        "type":        "score_only",
        "soft":        False,
        "hard":        True,
        "dropout":     False,
        "temp":        1e-6,
        "model_names": MODEL_NAMES,
        "source_file": str(args.seq_file),
    }

    results = []
    for idx, seq in enumerate(sequences):
        print(f"[{idx+1}/{len(sequences)}] {seq} ...", flush=True)
        r = score_one(model, raw_energy_fn, seq)
        r["idx"] = idx
        results.append(r)
        print(
            f"  i_ptm={r.get('i_ptm', float('nan')):.4f}  "
            f"ptm={r.get('ptm', float('nan')):.4f}  "
            f"plddt={r.get('plddt', float('nan')):.4f}  "
            f"energy={r['energy']:.4f}"
        )

    output = {"protocol": protocol, "results": results}
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} scores to {out_file}")

    print("\n" + "=" * 72)
    print(f"{'idx':>4}  {'i_ptm':>6}  {'ptm':>6}  {'plddt':>6}  {'energy':>8}  seq")
    print("-" * 72)
    for r in results:
        print(
            f"{r['idx']:>4}  "
            f"{r.get('i_ptm', float('nan')):>6.4f}  "
            f"{r.get('ptm',   float('nan')):>6.4f}  "
            f"{r.get('plddt', float('nan')):>6.4f}  "
            f"{r['energy']:>8.4f}  "
            f"{r['seq']}"
        )


if __name__ == "__main__":
    main()
