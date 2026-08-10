"""
generate_sequences.py

Generates N binder sequences with a custom 4-stage energy-guided protocol:

  Stage 1a (50 steps) : soft logits, dropout ON,  temp=1.0, energy_weight=EW_1A
  Stage 1b (50 steps) : soft logits, dropout ON,  temp=1.0, energy_weight=EW_1B
  Stage 2  (50 steps) : temperature annealing 1→0, dropout ON,  energy_weight=EW_2
  Stage 3  (30 steps) : hard (discrete), dropout OFF, temp~0,   energy_weight=EW_3

For each sequence reports: loss_total, loss_af, energy (MLP), i_ptm, ptm, plddt.
Energy is always computed and reported regardless of energy weights.
Results are saved to a JSON file and a summary table is printed.

Usage:
    python generate_sequences.py                        # all defaults
    python generate_sequences.py -o results.json        # custom output file
    python generate_sequences.py -n 50 --ew_1a 0.0 --ew_1b 0.0 --ew_2 0.0 --ew_3 0.0
    python generate_sequences.py --ew_1a 0.02 --ew_1b 0.2 --ew_2 0.05 --ew_3 0.2
"""

import os
import json
import argparse

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from colabdesign import mk_afdesign_model
from colabdesign.energy_model.model_3layer import make_energy_fn

# ── CONFIG ────────────────────────────────────────────────────────────────────
from config import PDB, PARAMS_DIR, WEIGHTS_PATH, HOTSPOT_RESIDUES  # set via env vars, see .env.example
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

HOTSPOT_STR = ",".join([f"A{r}" for r in HOTSPOT_RESIDUES])

SEED_START = 0     # seeds: SEED_START, SEED_START+1, ...

# Iteration counts per stage (fixed)
ITERS_1A   = 50
ITERS_1B   = 50
ITERS_2    = 50
ITERS_3    = 30
# ─────────────────────────────────────────────────────────────────────────────


def design_one(model, raw_energy_fn, seed: int,
               ew_1a: float, ew_1b: float, ew_2: float, ew_3: float) -> dict:
    """
    Custom 4-stage design for one seed.

    Stage 1a: soft, dropout, temp=1, ew_1a  → broad exploration
    Stage 1b: soft, dropout, temp=1, ew_1b  → energy-guided shaping
    Stage 2 : anneal temp 1→0, dropout,     → sequence focusing
    Stage 3 : hard (discrete), no dropout   → stable convergence

    Energy is always computed at the end regardless of weight values.
    """
    model.restart(seed=seed)

    # ── Stage 1a ──────────────────────────────────────────────────────────────
    model.set_opt(soft=True, hard=False, dropout=True, temp=1.0)
    model.opt["weights"]["energy"] = ew_1a
    for _ in range(ITERS_1A):
        model.step()

    # ── Stage 1b ──────────────────────────────────────────────────────────────
    model.opt["weights"]["energy"] = ew_1b
    for _ in range(ITERS_1B):
        model.step()

    # ── Stage 2: temperature annealing ────────────────────────────────────────
    model.set_opt(soft=False, hard=False, dropout=True)
    model.opt["weights"]["energy"] = ew_2
    for i in range(ITERS_2):
        model.set_opt(temp=1.0 - i / ITERS_2)
        model.step()

    # ── Stage 3: hard discrete, no dropout ────────────────────────────────────
    model.set_opt(soft=False, hard=True, dropout=False, temp=1e-6)
    model.opt["weights"]["energy"] = ew_3
    for _ in range(ITERS_3):
        model.step()

    # ── Collect results ───────────────────────────────────────────────────────
    seq_probs  = model.aux["seq"]["pseudo"]
    raw_energy = float(raw_energy_fn(seq_probs))   # always computed
    log        = model.aux["log"]
    loss_total = float(log["loss"])
    loss_af    = loss_total - ew_3 * raw_energy    # AF component (last stage weight)

    result = {
        "seed":       seed,
        "seq":        model.get_seqs()[0],
        "loss_total": loss_total,
        "loss_af":    loss_af,
        "energy":     raw_energy,
    }
    for key in ("i_ptm", "ptm", "plddt"):
        if key in log:
            result[key] = float(log[key])

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate binder sequences with per-stage energy weights.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON file (default: generated_sequences_ew<1a>_<1b>_<2>_<3>.json)")
    parser.add_argument("-n", "--n_seqs", type=int, default=10,
                        help="Number of sequences to generate (default: 10)")
    parser.add_argument("--ew_1a", type=float, default=0.05,
                        help="Energy weight for stage 1a (default: 0.05)")
    parser.add_argument("--ew_1b", type=float, default=0.2,
                        help="Energy weight for stage 1b (default: 0.2)")
    parser.add_argument("--ew_2",  type=float, default=0.05,
                        help="Energy weight for stage 2  (default: 0.05)")
    parser.add_argument("--ew_3",  type=float, default=0.2,
                        help="Energy weight for stage 3  (default: 0.2)")
    args = parser.parse_args()

    N_SEQS = args.n_seqs
    EW_1A  = args.ew_1a
    EW_1B  = args.ew_1b
    EW_2   = args.ew_2
    EW_3   = args.ew_3
    out_file = args.output or os.path.join(
        OUT_DIR, f"generated_sequences_ew{EW_1A:.2f}_{EW_1B:.2f}_{EW_2:.2f}_{EW_3:.2f}.json"
    )

    protocol = {
        "stage_1a": {"iters": ITERS_1A, "soft": True,  "hard": False, "dropout": True,  "temp": 1.0,   "energy_weight": EW_1A},
        "stage_1b": {"iters": ITERS_1B, "soft": True,  "hard": False, "dropout": True,  "temp": 1.0,   "energy_weight": EW_1B},
        "stage_2":  {"iters": ITERS_2,  "soft": False, "hard": False, "dropout": True,  "temp": "1→0", "energy_weight": EW_2},
        "stage_3":  {"iters": ITERS_3,  "soft": False, "hard": True,  "dropout": False, "temp": 1e-6,  "energy_weight": EW_3},
    }

    raw_energy_fn = make_energy_fn(WEIGHTS_PATH, energy_weight=1.0)

    # Initialise with stage-1a weight; overridden stage-by-stage in design_one.
    model = mk_afdesign_model(
        protocol="binder",
        data_dir=PARAMS_DIR,
        energy_fn=raw_energy_fn,
        energy_weight=EW_1A,
    )
    model.prep_inputs(
        pdb_filename=PDB,
        chain="A",
        binder_len=15,
        hotspot=HOTSPOT_STR,
        rm_target_seq=False,
        #num_models=5,
        model_names=["model_3", "model_4", "model_5"],
    )

    results = []
    for i in range(N_SEQS):
        seed = SEED_START + i
        print(f"[{i+1}/{N_SEQS}] seed={seed} ...", flush=True)
        r = design_one(model, raw_energy_fn, seed, EW_1A, EW_1B, EW_2, EW_3)
        results.append(r)
        print(
            f"  seq={r['seq']}  loss={r['loss_total']:.4f}  "
            f"loss_af={r['loss_af']:.4f}  energy={r['energy']:.4f}  "
            f"i_ptm={r.get('i_ptm', float('nan')):.4f}"
        )

    # Save protocol + results.
    output = {"protocol": protocol, "results": results}
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_file}")

    # Summary table.
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'seed':>5}  {'loss_total':>10}  {'loss_af':>8}  {'energy':>8}  "
          f"{'i_ptm':>6}  {'ptm':>6}  {'plddt':>6}  seq")
    print("-" * 80)
    for r in results:
        print(
            f"{r['seed']:>5}  "
            f"{r['loss_total']:>10.4f}  "
            f"{r['loss_af']:>8.4f}  "
            f"{r['energy']:>8.4f}  "
            f"{r.get('i_ptm', float('nan')):>6.4f}  "
            f"{r.get('ptm',   float('nan')):>6.4f}  "
            f"{r.get('plddt', float('nan')):>6.4f}  "
            f"{r['seq']}"
        )


if __name__ == "__main__":
    main()
