#!/usr/bin/env python3
"""
collect_predictions.py

Merges per-sequence prediction JSONs written by run_predictions.sh into two
final output files:

  predictions_af2.json   — AF2 metrics (i_ptm, ptm, plddt, energy)
  predictions_boltz.json — Boltz metrics (iptm, ptm, confidence_score,
                           complex_plddt, complex_iplddt)

Per-sequence files expected in pred_dir:
  seq_{N}_af2.json    — written by score_sequences.py
  seq_{N}_boltz.json  — written inline from the confidence JSON

Usage:
    python collect_predictions.py --pred_dir predictions/
    python collect_predictions.py \\
        --pred_dir results/predictions \\
        --af2_out results/predictions_af2.json \\
        --boltz_out results/predictions_boltz.json
"""

import json
import argparse
import sys
from pathlib import Path


def load_af2(path: Path) -> dict | None:
    """Load a seq_N_af2.json written by score_sequences.py."""
    try:
        data = json.loads(path.read_text())
        # score_sequences.py wraps single-seq output as {"source":..., "results":[...]}
        # but run_predictions.sh calls it with --seq (single), so results has one entry
        if "results" in data:
            return data["results"][0]
        # Fallback: direct dict
        return data
    except Exception as e:
        print(f"  WARNING: could not parse {path.name}: {e}", file=sys.stderr)
        return None


def load_boltz(path: Path) -> dict | None:
    """Load a seq_N_boltz.json."""
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  WARNING: could not parse {path.name}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-sequence prediction JSONs into two summary files."
    )
    parser.add_argument("--pred_dir",    required=True,
                        help="Directory containing seq_N_af2.json / seq_N_boltz.json files")
    parser.add_argument("--source_file", default=None,
                        help="Original sequence file (recorded in output metadata)")
    parser.add_argument("--af2_out",     default=None,
                        help="Output path for AF2 summary JSON")
    parser.add_argument("--boltz_out",   default=None,
                        help="Output path for Boltz summary JSON")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        print(f"ERROR: pred_dir not found: {pred_dir}", file=sys.stderr)
        sys.exit(1)

    af2_out   = Path(args.af2_out)   if args.af2_out   else pred_dir / "predictions_af2.json"
    boltz_out = Path(args.boltz_out) if args.boltz_out else pred_dir / "predictions_boltz.json"

    # Collect all indices present in pred_dir
    af2_files   = sorted(pred_dir.glob("seq_*_af2.json"),
                         key=lambda p: int(p.stem.split("_")[1]))
    boltz_files = sorted(pred_dir.glob("seq_*_boltz.json"),
                         key=lambda p: int(p.stem.split("_")[1]))

    if not af2_files and not boltz_files:
        print(f"ERROR: no seq_*_af2.json or seq_*_boltz.json found in {pred_dir}",
              file=sys.stderr)
        sys.exit(1)

    # ── AF2 results ───────────────────────────────────────────────────────────
    af2_results = []
    for fpath in af2_files:
        r = load_af2(fpath)
        if r is None:
            continue
        # Normalise idx from filename if not in dict
        if "idx" not in r:
            r["idx"] = int(fpath.stem.split("_")[1])
        # Canonical field order
        entry = {"idx": r["idx"], "seq": r.get("seq", "")}
        for k in ("i_ptm", "ptm", "plddt", "energy"):
            if k in r:
                entry[k] = r[k]
        af2_results.append(entry)
        print(f"  AF2  seq_{r['idx']:>3}  {entry.get('seq','')}  "
              f"i_ptm={entry.get('i_ptm', float('nan')):.4f}  "
              f"plddt={entry.get('plddt', float('nan')):.4f}  "
              f"energy={entry.get('energy', float('nan')):.4f}")

    # ── Boltz results ─────────────────────────────────────────────────────────
    boltz_results = []
    for fpath in boltz_files:
        r = load_boltz(fpath)
        if r is None:
            continue
        if "idx" not in r:
            r["idx"] = int(fpath.stem.split("_")[1])
        entry = {"idx": r["idx"], "seq": r.get("seq", "")}
        for k in ("iptm", "ptm", "confidence_score", "complex_plddt", "complex_iplddt"):
            if k in r:
                entry[k] = r[k]
        if "error" in r:
            entry["error"] = r["error"]
        boltz_results.append(entry)
        print(f"  Boltz seq_{r['idx']:>3}  {entry.get('seq','')}  "
              f"iptm={entry.get('iptm', float('nan')):.4f}  "
              f"conf={entry.get('confidence_score', float('nan')):.4f}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    source_label = str(args.source_file) if args.source_file else str(pred_dir)

    af2_out.parent.mkdir(parents=True, exist_ok=True)
    with open(af2_out, "w") as f:
        json.dump({"source": source_label, "results": af2_results}, f, indent=2)

    boltz_out.parent.mkdir(parents=True, exist_ok=True)
    with open(boltz_out, "w") as f:
        json.dump({"source": source_label, "results": boltz_results}, f, indent=2)

    print(f"\nAF2   : {len(af2_results)} sequences → {af2_out}")
    print(f"Boltz : {len(boltz_results)} sequences → {boltz_out}")


if __name__ == "__main__":
    main()
