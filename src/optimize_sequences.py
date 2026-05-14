"""
optimize_sequences.py

Takes one or more input sequences and runs a 2-stage optimization protocol:

  Stage 2 (50 steps, optional) : temperature annealing 1→0, dropout ON,  energy_weight=EW_2
  Stage 3 (30 steps)           : hard (discrete), dropout OFF, temp~0,   energy_weight=EW_3

The input sequence(s) are used as initialization (one-hot). Stage 2 allows gradient-based
refinement in continuous logit space (with temperature annealing) before stage 3 locks in
discrete choices. Set --iters_2 0 to skip stage 2 and go directly to stage 3.

Multiple seeds repeat the optimization from the same input sequence with different random
states (affects dropout stochasticity in stage 2). Stage 3 alone (no dropout) is
deterministic — use -n 1 in that case.

Input: a single sequence (positional) or a file of sequences (--seq_file, one per line).
When --seq_file is used, all sequences must have the same length.

For each result reports: input_seq, seed, loss_total, loss_af, energy (MLP), i_ptm, ptm, plddt.
Results are saved to a JSON file and a summary table is printed.

Usage:
    python optimize_sequences.py ACDEFGHIKLMNPQRST
    python optimize_sequences.py ACDEFGHIKLMNPQRST -n 4 --ew_2 0.05 --ew_3 0.2
    python optimize_sequences.py ACDEFGHIKLMNPQRST --iters_2 0 --ew_3 0.5
    python optimize_sequences.py --seq_file input_sequences.txt -n 4 --ew_2 0.02 --ew_3 0.2
    python optimize_sequences.py --seq_file input_sequences.txt --iters_2 0 --ew_3 0 -n 1
"""

import os
import json
import argparse

from colabdesign import mk_afdesign_model
from colabdesign.energy_model.model_3layer import make_energy_fn

# ── CONFIG ────────────────────────────────────────────────────────────────────
from config import PDB, PARAMS_DIR, WEIGHTS_PATH, HOTSPOT_RESIDUES
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

HOTSPOT_STR = ",".join([f"A{r}" for r in HOTSPOT_RESIDUES])

SEED_START = 0     # seeds: SEED_START, SEED_START+1, ...

# Default iteration counts per stage
ITERS_2_DEFAULT = 50
ITERS_3_DEFAULT = 30
# ─────────────────────────────────────────────────────────────────────────────


def optimize_one(model, raw_energy_fn, input_seq: str, seed: int,
                 ew_2: float, ew_3: float,
                 iters_2: int, iters_3: int) -> dict:
    """
    2-stage optimization starting from input_seq.

    Stage 2 (optional): anneal temp 1→0, dropout ON  → gradient refinement
    Stage 3           : hard (discrete), no dropout   → stable convergence

    Energy is always computed at the end regardless of weight values.
    """
    model.restart(seed=seed, seq=input_seq)

    # ── Stage 2: temperature annealing ────────────────────────────────────────
    if iters_2 > 0:
        model.set_opt(soft=False, hard=False, dropout=True)
        model.opt["weights"]["energy"] = ew_2
        for i in range(iters_2):
            model.set_opt(temp=1.0 - i / iters_2)
            model.step()

    # ── Stage 3: hard discrete, no dropout ────────────────────────────────────
    model.set_opt(soft=False, hard=True, dropout=False, temp=1e-6)
    model.opt["weights"]["energy"] = ew_3
    for _ in range(iters_3):
        model.step()

    # ── Collect results ───────────────────────────────────────────────────────
    seq_probs  = model.aux["seq"]["pseudo"]
    raw_energy = float(raw_energy_fn(seq_probs))
    log        = model.aux["log"]
    loss_total = float(log["loss"])
    loss_af    = loss_total - ew_3 * raw_energy

    result = {
        "seed":       seed,
        "input_seq":  input_seq,
        "seq":        model.get_seqs()[0],
        "loss_total": loss_total,
        "loss_af":    loss_af,
        "energy":     raw_energy,
    }
    for key in ("i_ptm", "ptm", "plddt"):
        if key in log:
            result[key] = float(log[key])

    return result


def load_sequences(seq_file: str) -> list:
    """Read sequences from a text file (one per line, # lines ignored)."""
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
            f"All sequences must have the same length; found lengths: {sorted(lengths)}"
        )
    return seqs


def main():
    parser = argparse.ArgumentParser(
        description="Optimize binder sequence(s) with per-stage energy weights."
    )
    # Input: single sequence or file
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("input_seq", nargs="?", default=None,
                           help="Single input sequence (positional)")
    input_grp.add_argument("--seq_file", default=None,
                           help="Text file with one sequence per line (# = comment)")

    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON file (auto-named by weights if omitted)")
    parser.add_argument("-n", "--n_seeds", type=int, default=1,
                        help="Seeds per input sequence (default: 1; use >1 for stochastic stage 2)")
    parser.add_argument("--ew_2",    type=float, default=0.05,
                        help="Energy weight for stage 2 (default: 0.05)")
    parser.add_argument("--ew_3",    type=float, default=0.2,
                        help="Energy weight for stage 3 (default: 0.2)")
    parser.add_argument("--iters_2", type=int, default=ITERS_2_DEFAULT,
                        help=f"Iterations for stage 2 (default: {ITERS_2_DEFAULT}; 0 = skip)")
    parser.add_argument("--iters_3", type=int, default=ITERS_3_DEFAULT,
                        help=f"Iterations for stage 3 (default: {ITERS_3_DEFAULT})")
    args = parser.parse_args()

    # ── Resolve input sequences ───────────────────────────────────────────────
    if args.seq_file is not None:
        sequences = load_sequences(args.seq_file)
        src_label = os.path.basename(args.seq_file)
    else:
        sequences = [args.input_seq]
        src_label = args.input_seq

    N_SEEDS = args.n_seeds
    EW_2    = args.ew_2
    EW_3    = args.ew_3
    ITERS_2 = args.iters_2
    ITERS_3 = args.iters_3

    stage2_tag = f"anneal{ITERS_2}" if ITERS_2 > 0 else "hard_only"
    out_file = args.output or os.path.join(
        OUT_DIR,
        f"optimized_{stage2_tag}_ew{EW_2:.2f}_{EW_3:.2f}.json"
    )

    protocol = {
        "source":  src_label,
        "n_input_seqs": len(sequences),
        "n_seeds_per_seq": N_SEEDS,
        "stage_2": {"iters": ITERS_2, "soft": False, "hard": False, "dropout": True,
                    "temp": "1→0", "energy_weight": EW_2, "skipped": ITERS_2 == 0},
        "stage_3": {"iters": ITERS_3, "soft": False, "hard": True,  "dropout": False,
                    "temp": 1e-6,  "energy_weight": EW_3},
    }

    raw_energy_fn = make_energy_fn(WEIGHTS_PATH, energy_weight=1.0)

    binder_len = len(sequences[0])
    model = mk_afdesign_model(
        protocol="binder",
        data_dir=PARAMS_DIR,
        energy_fn=raw_energy_fn,
        energy_weight=EW_3,
    )
    model.prep_inputs(
        pdb_filename=PDB,
        chain="A",
        binder_len=binder_len,
        hotspot=HOTSPOT_STR,
        rm_target_seq=False,
        model_names=["model_3", "model_4", "model_5"],
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    results = []
    total = len(sequences) * N_SEEDS
    idx = 0
    for input_seq in sequences:
        for i in range(N_SEEDS):
            seed = SEED_START + i
            idx += 1
            print(f"[{idx}/{total}] input={input_seq}  seed={seed} ...", flush=True)
            r = optimize_one(model, raw_energy_fn, input_seq, seed, EW_2, EW_3, ITERS_2, ITERS_3)
            results.append(r)
            print(
                f"  → seq={r['seq']}  loss={r['loss_total']:.4f}  "
                f"loss_af={r['loss_af']:.4f}  energy={r['energy']:.4f}  "
                f"i_ptm={r.get('i_ptm', float('nan')):.4f}"
            )

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {"protocol": protocol, "results": results}
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_file}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"{'seed':>5}  {'loss_total':>10}  {'loss_af':>8}  {'energy':>8}  "
          f"{'i_ptm':>6}  {'ptm':>6}  {'plddt':>6}  input_seq  →  seq")
    print("-" * 110)
    for r in results:
        print(
            f"{r['seed']:>5}  "
            f"{r['loss_total']:>10.4f}  "
            f"{r['loss_af']:>8.4f}  "
            f"{r['energy']:>8.4f}  "
            f"{r.get('i_ptm', float('nan')):>6.4f}  "
            f"{r.get('ptm',   float('nan')):>6.4f}  "
            f"{r.get('plddt', float('nan')):>6.4f}  "
            f"{r['input_seq']}  →  {r['seq']}"
        )


if __name__ == "__main__":
    main()
