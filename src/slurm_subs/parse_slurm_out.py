#!/usr/bin/env python3
"""
parse_slurm_out.py

Reconstruct JSON result files from SLURM .out files produced by
generate_sequences.py and optimize_sequences.py when the cluster killed
the job before saving.

Parses two output formats:
  - generate_sequences.py  : "[N/M] seed=S ..."  then "  seq=SEQ  loss=..."
  - optimize_sequences.py  : "[N/M] input=SEQ  seed=S ..."  then "  → seq=SEQ  loss=..."

For ptm/plddt (not printed in the inline summary line):
  - If the SUMMARY table was printed before the kill, values come from there.
  - Otherwise, averages the hard=1 iteration lines in each block.

Usage (from the slurm_subs directory):
    python parse_slurm_out.py                      # process all *.out files
    python parse_slurm_out.py pa_pb1_wide_*.out    # explicit list
    python parse_slurm_out.py --force              # overwrite existing JSONs
    python parse_slurm_out.py --outdir /path/dir   # custom output dir
"""

import re
import json
import sys
import copy
import argparse
from pathlib import Path


# ── Protocol configurations (hardcoded from submit_wide_search.sh) ────────────

PROTOCOL_CONFIGS = {
    "gen_energy_A": {
        "type": "gen",
        "protocol": {
            "stage_1a": {"iters": 50, "soft": True,  "hard": False, "dropout": True,  "temp": 1.0,   "energy_weight": 0.02},
            "stage_1b": {"iters": 50, "soft": True,  "hard": False, "dropout": True,  "temp": 1.0,   "energy_weight": 0.05},
            "stage_2":  {"iters": 50, "soft": False, "hard": False, "dropout": True,  "temp": "1\u21920", "energy_weight": 0.02},
            "stage_3":  {"iters": 30, "soft": False, "hard": True,  "dropout": False, "temp": 1e-6,  "energy_weight": 0.05},
        },
    },
    "gen_energy_C": {
        "type": "gen",
        "protocol": {
            "stage_1a": {"iters": 50, "soft": True,  "hard": False, "dropout": True,  "temp": 1.0,   "energy_weight": 0.05},
            "stage_1b": {"iters": 50, "soft": True,  "hard": False, "dropout": True,  "temp": 1.0,   "energy_weight": 0.20},
            "stage_2":  {"iters": 50, "soft": False, "hard": False, "dropout": True,  "temp": "1\u21920", "energy_weight": 0.05},
            "stage_3":  {"iters": 30, "soft": False, "hard": True,  "dropout": False, "temp": 1e-6,  "energy_weight": 0.50},
        },
    },
    "opt_hard_energy_C": {
        "type": "opt",
        "n_seeds_per_seq": 1,
        "protocol": {
            "source": "input_sequences.txt",
            "n_input_seqs": None,   # filled at parse time
            "n_seeds_per_seq": 1,
            "stage_2": {"iters": 0,  "soft": False, "hard": False, "dropout": True,  "temp": "1\u21920", "energy_weight": 0.0,  "skipped": True},
            "stage_3": {"iters": 30, "soft": False, "hard": True,  "dropout": False, "temp": 1e-6,  "energy_weight": 0.50},
        },
    },
    "opt_anneal_noenergy": {
        "type": "opt",
        "n_seeds_per_seq": 4,
        "protocol": {
            "source": "input_sequences.txt",
            "n_input_seqs": None,
            "n_seeds_per_seq": 4,
            "stage_2": {"iters": 50, "soft": False, "hard": False, "dropout": True,  "temp": "1\u21920", "energy_weight": 0.00, "skipped": False},
            "stage_3": {"iters": 30, "soft": False, "hard": True,  "dropout": False, "temp": 1e-6,  "energy_weight": 0.00},
        },
    },
    "opt_anneal_energy_B": {
        "type": "opt",
        "n_seeds_per_seq": 4,
        "protocol": {
            "source": "input_sequences.txt",
            "n_input_seqs": None,
            "n_seeds_per_seq": 4,
            "stage_2": {"iters": 50, "soft": False, "hard": False, "dropout": True,  "temp": "1\u21920", "energy_weight": 0.02, "skipped": False},
            "stage_3": {"iters": 30, "soft": False, "hard": True,  "dropout": False, "temp": 1e-6,  "energy_weight": 0.20},
        },
    },
    "opt_anneal_energy_C": {
        "type": "opt",
        "n_seeds_per_seq": 4,
        "protocol": {
            "source": "input_sequences.txt",
            "n_input_seqs": None,
            "n_seeds_per_seq": 4,
            "stage_2": {"iters": 50, "soft": False, "hard": False, "dropout": True,  "temp": "1\u21920", "energy_weight": 0.05, "skipped": False},
            "stage_3": {"iters": 30, "soft": False, "hard": True,  "dropout": False, "temp": 1e-6,  "energy_weight": 0.50},
        },
    },
}


# ── Regex patterns ─────────────────────────────────────────────────────────────

FLOAT = r'(-?[\d]+\.?[\d]*(?:[eE][-+]?\d+)?)'

# Header: "Array task   : 0  →  protocol: gen_energy_A"
RE_PROTOCOL = re.compile(r'Array task\s*:.*protocol:\s*(\S+)')

# Block starts
RE_GEN_BLOCK = re.compile(r'^\[(\d+)/(\d+)\]\s+seed=(\d+)')
RE_OPT_BLOCK = re.compile(r'^\[(\d+)/(\d+)\]\s+input=(\S+)\s+seed=(\d+)')

# Inline result lines (printed after each block)
RE_GEN_SEQ = re.compile(
    rf'^\s+seq=(\S+)\s+loss={FLOAT}\s+loss_af={FLOAT}\s+energy={FLOAT}\s+i_ptm={FLOAT}'
)
RE_OPT_SEQ = re.compile(
    rf'^\s+→\s+seq=(\S+)\s+loss={FLOAT}\s+loss_af={FLOAT}\s+energy={FLOAT}\s+i_ptm={FLOAT}'
)

# Iteration line (hard=1): contains plddt and ptm
# e.g.: "42 models [0] recycles 0 hard 1 soft 0 temp 0.00 loss 3.67 i_con 3.69 plddt 0.37 ptm 0.88 i_ptm 0.38 [energy -1.73]"
RE_HARD_ITER = re.compile(
    rf'^\d+ models \[\d+\] recycles \d+ hard 1 soft \d+ temp {FLOAT}'
    rf' loss {FLOAT} i_con {FLOAT} plddt {FLOAT} ptm {FLOAT}'
)

# Summary table header
RE_SUM_HEADER = re.compile(r'^\s*seed\s+loss_total')


# ── Parsing functions ──────────────────────────────────────────────────────────

def _parse_hard_iter(line):
    """Return (plddt, ptm) from a hard=1 iteration line, or None."""
    m = RE_HARD_ITER.match(line)
    if m:
        # groups: temp(1), loss(2), i_con(3), plddt(4), ptm(5)
        return float(m.group(4)), float(m.group(5))
    return None


def _parse_summary_table(lines, ptype):
    """
    Parse the SUMMARY table if present.
    Returns list of result dicts or None if no table found.
    """
    results = []
    in_summary = False
    past_header = False

    for line in lines:
        stripped = line.strip()

        if not in_summary:
            if stripped == "SUMMARY":
                in_summary = True
            continue

        if not past_header:
            if RE_SUM_HEADER.match(line):
                past_header = True
            continue

        # Skip separator / empty lines
        if not stripped or stripped.startswith('-') or stripped.startswith('='):
            continue

        # Break if we hit another section or unrecognisable content
        parts = stripped.split()
        if len(parts) < 8:
            break

        try:
            seed       = int(parts[0])
            loss_total = float(parts[1])
            loss_af    = float(parts[2])
            energy     = float(parts[3])
            i_ptm      = float(parts[4])
            ptm        = float(parts[5])
            plddt      = float(parts[6])
        except ValueError:
            break

        if ptype == "opt":
            # " seed  loss_total  loss_af  energy  i_ptm  ptm  plddt  input_seq  →  seq "
            if len(parts) < 10 or parts[8] != '→':
                break
            input_seq = parts[7]
            seq       = parts[9]
            results.append({
                "seed": seed, "input_seq": input_seq, "seq": seq,
                "loss_total": loss_total, "loss_af": loss_af, "energy": energy,
                "i_ptm": i_ptm, "ptm": ptm, "plddt": plddt,
            })
        else:
            # " seed  loss_total  loss_af  energy  i_ptm  ptm  plddt  seq "
            seq = parts[7]
            results.append({
                "seed": seed, "seq": seq,
                "loss_total": loss_total, "loss_af": loss_af, "energy": energy,
                "i_ptm": i_ptm, "ptm": ptm, "plddt": plddt,
            })

    return results if results else None


def _parse_inline(lines, ptype):
    """
    Parse inline block-start and seq= lines.
    ptm/plddt come from averaging hard=1 iteration lines in each block.
    """
    results    = []
    current    = None   # dict for the block currently being read
    hard_iters = []     # (plddt, ptm) from hard=1 lines in current block

    RE_BLOCK = RE_OPT_BLOCK if ptype == "opt" else RE_GEN_BLOCK
    RE_SEQ   = RE_OPT_SEQ   if ptype == "opt" else RE_GEN_SEQ

    for line in lines:
        # ── New block start ──────────────────────────────────────────────────
        m = RE_BLOCK.match(line)
        if m:
            current    = {}
            hard_iters = []
            if ptype == "opt":
                current["input_seq"] = m.group(3)
                current["seed"]      = int(m.group(4))
                current["total"]     = int(m.group(2))
            else:
                current["seed"]  = int(m.group(3))
                current["total"] = int(m.group(2))
            continue

        if current is None:
            continue

        # ── Collect hard=1 iteration metrics ────────────────────────────────
        vals = _parse_hard_iter(line)
        if vals:
            hard_iters.append(vals)
            continue

        # ── Inline result line ───────────────────────────────────────────────
        m = RE_SEQ.match(line)
        if m:
            r = {
                "seed":       current["seed"],
                "seq":        m.group(1),
                "loss_total": float(m.group(2)),
                "loss_af":    float(m.group(3)),
                "energy":     float(m.group(4)),
                "i_ptm":      float(m.group(5)),
            }
            if ptype == "opt":
                r["input_seq"] = current["input_seq"]

            # ptm / plddt from averaged hard-stage iterations
            if hard_iters:
                avg_plddt = sum(x[0] for x in hard_iters) / len(hard_iters)
                avg_ptm   = sum(x[1] for x in hard_iters) / len(hard_iters)
                r["ptm"]   = round(avg_ptm,   6)
                r["plddt"] = round(avg_plddt, 6)

            # Build result in canonical field order
            ordered = {"seed": r["seed"]}
            if ptype == "opt":
                ordered["input_seq"] = r["input_seq"]
            ordered["seq"]        = r["seq"]
            ordered["loss_total"] = r["loss_total"]
            ordered["loss_af"]    = r["loss_af"]
            ordered["energy"]     = r["energy"]
            ordered["i_ptm"]      = r["i_ptm"]
            if "ptm"   in r: ordered["ptm"]   = r["ptm"]
            if "plddt" in r: ordered["plddt"] = r["plddt"]

            results.append(ordered)
            current    = None
            hard_iters = []

    return results


def parse_out_file(path):
    """
    Main entry point: parse a single .out file.
    Returns (proto_name, protocol_dict, results_list).
    """
    path  = Path(path)
    lines = path.read_text(errors="replace").splitlines()

    # 1. Identify protocol
    proto_name = None
    for line in lines[:10]:
        m = RE_PROTOCOL.match(line)
        if m:
            proto_name = m.group(1)
            break

    if proto_name is None:
        raise ValueError(f"Could not find protocol name in {path.name}")
    if proto_name not in PROTOCOL_CONFIGS:
        raise ValueError(f"Unknown protocol '{proto_name}' in {path.name}")

    cfg      = PROTOCOL_CONFIGS[proto_name]
    ptype    = cfg["type"]
    protocol = copy.deepcopy(cfg["protocol"])

    # 2. Try SUMMARY table (most reliable; has ptm/plddt at full precision)
    results = _parse_summary_table(lines, ptype)
    source  = "SUMMARY table"

    # 3. Fall back to inline seq= lines + averaged hard-iter for ptm/plddt
    if not results:
        results = _parse_inline(lines, ptype)
        source  = "inline seq= lines"

    # 4. Fill n_input_seqs for opt_ protocols
    if ptype == "opt" and protocol.get("n_input_seqs") is None:
        n_seeds = cfg["n_seeds_per_seq"]
        # Try to infer from [N/total] in the file
        total = None
        RE_BLOCK = RE_OPT_BLOCK
        for line in lines:
            m = RE_BLOCK.match(line)
            if m:
                total = int(m.group(2))
                break
        if total is not None:
            protocol["n_input_seqs"] = total // n_seeds
        elif results:
            # Count unique input sequences
            protocol["n_input_seqs"] = len({r.get("input_seq") for r in results})

    return proto_name, protocol, results, source


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent
    default_outdir = here.parent / "results" / "run2_wide"

    parser = argparse.ArgumentParser(
        description="Reconstruct JSON results from SLURM .out files.",
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
            proto_name, protocol, results, source = parse_out_file(fpath)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failed += 1
            continue

        out_path = out_dir / f"{proto_name}.json"
        if out_path.exists() and not args.force:
            print(f"  SKIP (exists): {out_path.name}  — use --force to overwrite")
            skipped += 1
            continue

        output = {"protocol": protocol, "results": results}
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        ptm_note = "" if any("ptm" in r for r in results) else "  [ptm/plddt missing]"
        print(f"  Protocol : {proto_name}")
        print(f"  Source   : {source}")
        print(f"  Results  : {len(results)} sequences{ptm_note}")
        print(f"  Written  : {out_path}")
        ok += 1

    print(f"\n{'='*60}")
    print(f"Done — {ok} written, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
