"""
generate_sequences_baseline.py

Generates N binder sequences with the same 4-stage protocol as
generate_sequences.py but with no energy guidance (energy_weight=0):

  Stage 1a (50 steps) : soft logits, dropout ON,  temp=1.0
  Stage 1b (50 steps) : soft logits, dropout ON,  temp=1.0
  Stage 2  (100 steps): temperature annealing 1→0, dropout ON
  Stage 3  (50 steps) : hard (discrete), dropout OFF, temp~0  (convergence)

For each sequence reports: loss_total, i_ptm, ptm, plddt.
Results are saved to a JSON file and a summary table is printed.

Usage:
    python generate_sequences_baseline.py   # uses defaults in CONFIG block
"""

import os
import json

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from colabdesign import mk_afdesign_model

# ── CONFIG ────────────────────────────────────────────────────────────────────
from prg.PA_PB1.config import PDB, PARAMS_DIR  # set via env vars, see .env.example
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

HOTSPOT_RESIDUES = [408, 411, 412, 415, 594, 595, 599,
                    617, 618, 619, 620, 621, 622, 623,
                    629, 631, 639, 640, 643, 666, 667,
                    670, 673, 706, 709, 710, 713, 714]
HOTSPOT_STR = ",".join([f"A{r}" for r in HOTSPOT_RESIDUES])

N_SEQS     = 100   # number of independent sequences to generate
SEED_START = 0     # seeds: SEED_START, SEED_START+1, ...

# Iteration counts per stage (identical to guided protocol)
ITERS_1A   = 50
ITERS_1B   = 50
ITERS_2    = 100
ITERS_3    = 50
# ─────────────────────────────────────────────────────────────────────────────


def design_one(model, seed: int) -> dict:
    """
    Same 4-stage protocol as generate_sequences.py with energy_weight=0.

    Stage 1a: soft, dropout, temp=1
    Stage 1b: soft, dropout, temp=1  (identical to 1a — no energy weight change)
    Stage 2 : anneal temp 1→0, dropout
    Stage 3 : hard (discrete), no dropout
    """
    model.restart(seed=seed)

    # ── Stage 1a ──────────────────────────────────────────────────────────────
    model.set_opt(soft=True, hard=False, dropout=True, temp=1.0)
    for _ in range(ITERS_1A):
        model.step()

    # ── Stage 1b ──────────────────────────────────────────────────────────────
    for _ in range(ITERS_1B):
        model.step()

    # ── Stage 2: temperature annealing ────────────────────────────────────────
    model.set_opt(soft=False, hard=False, dropout=True)
    for i in range(ITERS_2):
        model.set_opt(temp=1.0 - i / ITERS_2)
        model.step()

    # ── Stage 3: hard discrete, no dropout ────────────────────────────────────
    model.set_opt(soft=False, hard=True, dropout=False, temp=1e-6)
    for _ in range(ITERS_3):
        model.step()

    # ── Collect results ───────────────────────────────────────────────────────
    log        = model.aux["log"]
    loss_total = float(log["loss"])

    result = {
        "seed":       seed,
        "seq":        model.get_seqs()[0],
        "loss_total": loss_total,
    }
    for key in ("i_ptm", "ptm", "plddt"):
        if key in log:
            result[key] = float(log[key])

    return result


def main():
    model = mk_afdesign_model(
        protocol="binder",
        data_dir=PARAMS_DIR,
    )
    model.prep_inputs(
        pdb_filename=PDB,
        chain="A",
        binder_len=15,
        hotspot=HOTSPOT_STR,
        rm_target_seq=False,
        num_models=5,
    )

    results = []
    for i in range(N_SEQS):
        seed = SEED_START + i
        print(f"[{i+1}/{N_SEQS}] seed={seed} ...", flush=True)
        r = design_one(model, seed)
        results.append(r)
        print(
            f"  seq={r['seq']}  loss={r['loss_total']:.4f}  "
            f"i_ptm={r.get('i_ptm', float('nan')):.4f}"
        )

    # Save full results.
    out_file = os.path.join(OUT_DIR, "generated_sequences_baseline.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_file}")

    # Summary table.
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'seed':>5}  {'loss_total':>10}  {'i_ptm':>6}  {'ptm':>6}  {'plddt':>6}  seq")
    print("-" * 80)
    for r in results:
        print(
            f"{r['seed']:>5}  "
            f"{r['loss_total']:>10.4f}  "
            f"{r.get('i_ptm', float('nan')):>6.4f}  "
            f"{r.get('ptm',   float('nan')):>6.4f}  "
            f"{r.get('plddt', float('nan')):>6.4f}  "
            f"{r['seq']}"
        )


if __name__ == "__main__":
    main()
