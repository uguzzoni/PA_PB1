#!/usr/bin/env python3
"""
parse_multimer_out.py

Reconstruct JSON result files from SLURM .out files produced by
generate_sequences_multimer.py and optimize_sequences_multimer.py.

Parses two output formats:
  - generate_sequences_multimer.py : "[N/M] seed=S ..."  then "  seq=SEQ  loss=..."
  - optimize_sequences_multimer.py : "[N/M] input=SEQ  seed=S ..."  then "  → seq=SEQ  loss=..."

plddt / ptm are averaged from hard=1 iteration lines in each block.
If no hard=1 lines are present they are set to NaN.

Usage (from the slurm_subs directory):
    python parse_multimer_out.py                      # process all *.out files
    python parse_multimer_out.py pa_pb1_wide_*.out    # explicit list
    python parse_multimer_out.py --force              # overwrite existing JSONs
    python parse_multimer_out.py --outdir /path/dir   # custom output dir
"""

import re
import json
import math
import sys
import copy
import argparse
from pathlib import Path


# ── Protocol configurations (from submit_multimer_search.sh) ─────────────────

PROTOCOL_CONFIGS = {
    "gen_energy_C": {
        "type": "gen",
        "protocol": {
            "stage_1a": {"iters": 50, "soft": True,  "hard": False, "dropout": True,  "temp": 1.0,   "energy_weight": 0.05},
            "stage_1b": {"iters": 50, "soft": True,  "hard": False, "dropout": True,  "temp": 1.0,   "energy_weight": 0.20},
            "stage_2":  {"iters": 50, "soft": False, "hard": False, "dropout": True,  "temp": "1→0", "energy_weight": 0.05},
            "stage_3":  {"iters": 30, "soft": False, "hard": True,  "dropout": False, "temp": 1e-6,  "energy_weight": 0.50},
        },
    },
    "opt_anneal_noenergy": {
        "type": "opt",
        "n_seeds_per_seq": 4,
        "protocol": {
            "source": "input_sequences.txt",
            "n_input_seqs": None,
            "n_seeds_per_seq": 4,
            "stage_2": {"iters": 50, "soft": False, "hard": False, "dropout": True,  "temp": "1→0", "energy_weight": 0.00},
            "stage_3": {"iters": 30, "soft": False, "hard": True,  "dropout": False, "temp": 1e-6,  "energy_weight": 0.00},
        },
    },
    "opt_anneal_energy_C": {
        "type": "opt",
        "n_seeds_per_seq": 4,
        "protocol": {
            "source": "input_sequences.txt",
            "n_input_seqs": None,
            "n_seeds_per_seq": 4,
            "stage_2": {"iters": 50, "soft": False, "hard": False, "dropout": True,  "temp": "1→0", "energy_weight": 0.05},
            "stage_3": {"iters": 30, "soft": False, "hard": True,  "dropout": False, "temp": 1e-6,  "energy_weight": 0.50},
        },
    },
}


# ── Regex patterns ─────────────────────────────────────────────────────────────

FLOAT = r'(-?[\d]+\.?[\d]*(?:[eE][-+]?\d+)?)'

# Header: "Array task   : 0  →  protocol: gen_energy_C"
RE_PROTOCOL = re.compile(r'Array task\s*:.*protocol:\s*(\S+)')

# Block starts
RE_GEN_BLOCK = re.compile(r'^\[(\d+)/(\d+)\]\s+seed=(\d+)')
RE_OPT_BLOCK = re.compile(r'^\[(\d+)/(\d+)\]\s+input=(\S+)\s+seed=(\d+)')

# Inline result lines
RE_GEN_SEQ = re.compile(
    rf'^\s+seq=(\S+)\s+loss={FLOAT}\s+loss_af={FLOAT}\s+energy={FLOAT}\s+i_ptm={FLOAT}'
)
RE_OPT_SEQ = re.compile(
    rf'^\s+→\s+seq=(\S+)\s+loss={FLOAT}\s+loss_af={FLOAT}\s+energy={FLOAT}\s+i_ptm={FLOAT}'
)

# hard=1 iteration line (contains plddt and ptm)
# e.g.: "42 models [0] recycles 0 hard 1 soft 0 temp 0.00 loss 3.67 i_con 3.69 plddt 0.37 ptm 0.88 ..."
RE_HARD_ITER = re.compile(
    rf'^\d+ models \[\d+\] recycles \d+ hard 1 soft \d+ temp {FLOAT}'
    rf' loss {FLOAT} i_con {FLOAT} plddt {FLOAT} ptm {FLOAT}'
)


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_hard_iter(line):
    """Return (plddt, ptm) from a hard=1 iteration line, or None."""
    m = RE_HARD_ITER.match(line)
    if m:
        # groups: temp(1), loss(2), i_con(3), plddt(4), ptm(5)
        return float(m.group(4)), float(m.group(5))
    return None


def _parse_inline(lines, ptype):
    """
    Parse inline block-start and seq= / → seq= lines.
    plddt/ptm are averaged from hard=1 iteration lines; NaN if none found.
    """
    results    = []
    current    = None
    hard_iters = []

    RE_BLOCK = RE_OPT_BLOCK if ptype == "opt" else RE_GEN_BLOCK
    RE_SEQ   = RE_OPT_SEQ   if ptype == "opt" else RE_GEN_SEQ

    for line in lines:
        # New block start
        m = RE_BLOCK.match(line)
        if m:
            current    = {}
            hard_iters = []
            if ptype == "opt":
                current["input_seq"] = m.group(3)
                current["seed"]      = int(m.group(4))
            else:
                current["seed"] = int(m.group(3))
            continue

        if current is None:
            continue

        # Collect hard=1 iteration metrics
        vals = _parse_hard_iter(line)
        if vals:
            hard_iters.append(vals)
            continue

        # Inline result line
        m = RE_SEQ.match(line)
        if m:
            if hard_iters:
                avg_plddt = sum(x[0] for x in hard_iters) / len(hard_iters)
                avg_ptm   = sum(x[1] for x in hard_iters) / len(hard_iters)
            else:
                avg_plddt = float("nan")
                avg_ptm   = float("nan")

            r = {"seed": current["seed"]}
            if ptype == "opt":
                r["input_seq"] = current["input_seq"]
            r["seq"]        = m.group(1)
            r["loss_total"] = float(m.group(2))
            r["loss_af"]    = float(m.group(3))
            r["energy"]     = float(m.group(4))
            r["i_ptm"]      = float(m.group(5))
            r["ptm"]        = round(avg_ptm,   6)
            r["plddt"]      = round(avg_plddt, 6)

            results.append(r)
            current    = None
            hard_iters = []

    return results


def parse_out_file(path):
    """
    Parse a single .out file.
    Returns (proto_name, protocol_dict, results_list).
    """
    path  = Path(path)
    lines = path.read_text(errors="replace").splitlines()

    # Identify protocol from header
    proto_name = None
    for line in lines[:10]:
        m = RE_PROTOCOL.match(line)
        if m:
            proto_name = m.group(1)
            break

    if proto_name is None:
        raise ValueError(f"Could not find protocol name in {path.name}")
    if proto_name not in PROTOCOL_CONFIGS:
        raise ValueError(f"Unknown protocol '{proto_name}' in {path.name} "
                         f"(known: {list(PROTOCOL_CONFIGS)})")

    cfg      = PROTOCOL_CONFIGS[proto_name]
    ptype    = cfg["type"]
    protocol = copy.deepcopy(cfg["protocol"])

    results = _parse_inline(lines, ptype)

    # Fill n_input_seqs for opt protocols
    if ptype == "opt" and protocol.get("n_input_seqs") is None:
        n_seeds = cfg["n_seeds_per_seq"]
        total   = None
        for line in lines:
            m = RE_OPT_BLOCK.match(line)
            if m:
                total = int(m.group(2))
                break
        if total is not None:
            protocol["n_input_seqs"] = total // n_seeds
        elif results:
            protocol["n_input_seqs"] = len({r.get("input_seq") for r in results})

    return proto_name, protocol, results


# ── NaN-aware JSON encoder ─────────────────────────────────────────────────────

class _NaNEncoder(json.JSONEncoder):
    """Serialize float NaN as JSON null."""
    def iterencode(self, o, _one_shot=False):
        # Walk the structure and replace nan with None before encoding
        return super().iterencode(_replace_nan(o), _one_shot)


def _replace_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _replace_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_nan(v) for v in obj]
    return obj


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent
    default_outdir = here.parent / "results" / "run_multimer"

    parser = argparse.ArgumentParser(
        description="Reconstruct JSON results from multimer SLURM .out files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files", nargs="*",
        help=".out files to process (default: all *.out in this directory)"
    )
    parser.add_argument(
        "--outdir", default=str(default_outdir),
        help=f"Output directory for JSON files (default: {default_outdir})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing JSON files"
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(f) for f in args.files] if args.files else sorted(here.glob("*.out"))
    if not files:
        print("No .out files found.", file=sys.stderr)
        sys.exit(1)

    ok = failed = skipped = 0
    for fpath in files:
        print(f"\n{'─'*60}")
        print(f"  {fpath.name}")
        try:
            proto_name, protocol, results = parse_out_file(fpath)
        except ValueError as e:
            print(f"  SKIP: {e}")
            skipped += 1
            continue
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failed += 1
            continue

        out_path = out_dir / f"{proto_name}.json"
        if out_path.exists() and not args.force:
            print(f"  SKIP (exists): {out_path.name}  — use --force to overwrite")
            skipped += 1
            continue

        nan_count = sum(
            1 for r in results
            if math.isnan(r.get("plddt", 0.0)) or math.isnan(r.get("ptm", 0.0))
        )
        output = {"protocol": protocol, "results": results}
        with open(out_path, "w") as f:
            json.dump(_replace_nan(output), f, indent=2)

        note = f"  [{nan_count} with plddt/ptm=null]" if nan_count else ""
        print(f"  Protocol : {proto_name}")
        print(f"  Results  : {len(results)} sequences{note}")
        print(f"  Written  : {out_path}")
        ok += 1

    print(f"\n{'='*60}")
    print(f"Done — {ok} written, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
