"""
scan_energy_weight.py

Runs binder design with energy guidance at several energy_weight values
and monitors:
  - loss_af   : AlphaFold loss component (total loss minus energy term)
  - energy    : raw MLP energy (energy_weight=1 scale)
  - loss_total: combined loss (loss_af + energy_weight * energy)

Results are saved per-weight to JSON files and a summary table is printed.
"""

import os
import json

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax.numpy as jnp
from colabdesign import mk_afdesign_model
from colabdesign.energy_model.model_3layer import make_energy_fn

from config import PDB, PARAMS_DIR, WEIGHTS_PATH, HOTSPOT_RESIDUES  # set via env vars, see .env.example
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

HOTSPOT_STR = ",".join([f"A{r}" for r in HOTSPOT_RESIDUES])

ENERGY_WEIGHTS = [0.0, 0.01, 0.1, 0.5]

SOFT_ITERS = 10#100
TEMP_ITERS = 10#100
HARD_ITERS = 5#50
SEED = 42


def run_design(energy_weight: float, seed: int = SEED):
    """
    Run design_3stage with the given energy_weight.

    At every step the callback records:
      loss_total  — total loss seen by the optimiser
      loss_af     — AlphaFold component  (loss_total - energy_weight * energy)
      energy      — raw MLP energy (at energy_weight=1 scale)
      seq         — discrete sequence
    plus any other keys present in model.aux["log"].

    Returns (results_list, final_seq, final_energy_raw).
    """
    results = []

    # Build the energy function with energy_weight=1 so the raw value is
    # weight-independent; ColabDesign applies the weight separately via
    # opt["weights"]["energy"].
    raw_energy_fn = make_energy_fn(WEIGHTS_PATH, energy_weight=1.0)

    if energy_weight > 0.0:
        model = mk_afdesign_model(
            protocol="binder",
            data_dir=PARAMS_DIR,
            energy_fn=raw_energy_fn,
            energy_weight=energy_weight,
        )
    else:
        model = mk_afdesign_model(protocol="binder", data_dir=PARAMS_DIR)

    model.prep_inputs(
        pdb_filename=PDB,
        chain="A",
        binder_len=15,
        hotspot=HOTSPOT_STR,
        rm_target_seq=False,
        model_names=["model_4", "model_5"],
    )

    def step_callback(m):
        log = m.aux.get("log", {})
        entry = {}
        for k, v in log.items():
            try:
                entry[k] = float(v)
            except (TypeError, ValueError):
                pass

        # Raw energy from the MLP (energy_weight=1 scale).
        seq_probs = m.aux["seq"]["pseudo"]
        raw_energy = float(raw_energy_fn(seq_probs))
        entry["energy"] = raw_energy

        # AlphaFold-only loss = total loss - weighted energy contribution.
        loss_total = entry.get("loss", float("nan"))
        entry["loss_total"] = loss_total
        entry["loss_af"] = loss_total - energy_weight * raw_energy

        try:
            entry["seq"] = m.get_seqs()[0]
        except Exception:
            pass

        results.append(entry)

    model.restart(seed=seed)
    model.design_3stage(
        soft_iters=SOFT_ITERS,
        temp_iters=TEMP_ITERS,
        hard_iters=HARD_ITERS,
        callback=step_callback,
        verbose=10,
    )

    final_seq_probs = model.aux["seq"]["pseudo"]
    final_energy = float(raw_energy_fn(final_seq_probs))

    # Read the final total loss directly from model.aux after the run.
    # This is more reliable than reading from the last callback entry,
    # which can be NaN during the hard (discrete) phase.
    final_loss_total = float(model.aux["log"]["loss"])
    final_loss_af = final_loss_total - energy_weight * final_energy

    return results, model.get_seqs()[0], final_energy, final_loss_total, final_loss_af


def main():
    summary = []

    for ew in ENERGY_WEIGHTS:
        print("\n" + "=" * 60)
        print(f"energy_weight = {ew}")
        print("=" * 60)

        results, seq, final_energy, final_loss_total, final_loss_af = run_design(energy_weight=ew)

        # Save per-step log.
        out_file = os.path.join(OUT_DIR, f"scan_ew{ew:.2f}_results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {out_file}")

        summary.append({
            "energy_weight": ew,
            "seq": seq,
            "final_loss_total": final_loss_total,
            "final_loss_af":    final_loss_af,
            "final_energy":     final_energy,
        })

    # Summary table.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'ew':>6}  {'loss_total':>12}  {'loss_af':>10}  {'energy':>10}  seq")
    print("-" * 70)
    for row in summary:
        print(
            f"{row['energy_weight']:>6.2f}  "
            f"{row['final_loss_total']:>12.4f}  "
            f"{row['final_loss_af']:>10.4f}  "
            f"{row['final_energy']:>10.4f}  "
            f"{row['seq']}"
        )


if __name__ == "__main__":
    main()

#########################################################################

# ============================================================
# SUMMARY
# ============================================================
#     ew    loss_total     loss_af      energy  seq
# ----------------------------------------------------------------------
#   0.00        5.4704      5.4704      3.6495  AACGCCCCCCLLCLL
#   0.01        5.4570      5.4225      3.4516  AACGCVCCVLLLCLL
#   0.10        4.3055      4.5979     -2.9233  AIGGLECLVCFLIFL
#   0.50       -0.3287      4.2098     -9.0770  NWTGLEFLVNYYAFC

#######################################################################