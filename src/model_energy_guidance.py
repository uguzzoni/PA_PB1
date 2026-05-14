"""
model_energy_guidance.py

Runs binder design using the MLP energy model from model_3layer.py.
Compares a baseline run (no energy guidance) against a guided run
using the pre-trained weights at COLABDESIGN_WEIGHTS_PATH (see .env.example).
"""

import os
import json
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax.numpy as jnp
from colabdesign import mk_afdesign_model
from colabdesign.energy_model.model_3layer import make_energy_fn

from prg.PA_PB1.config import PDB, PARAMS_DIR, WEIGHTS_PATH  # set via env vars, see .env.example
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

HOTSPOT_RESIDUES = [408, 411, 412, 415, 594, 595, 599,
                    617, 618, 619, 620, 621, 622, 623,
                    629, 631, 639, 640, 643, 666, 667,
                    670, 673, 706, 709, 710, 713, 714]
HOTSPOT_STR = ",".join([f"A{r}" for r in HOTSPOT_RESIDUES])


def run_design(with_energy: bool, seed: int = 42):
    """Run design_3stage and return per-step log + final sequence + final energy."""

    results = []

    # Always build the energy function so we can evaluate the final sequence.
    energy_fn = make_energy_fn(WEIGHTS_PATH, energy_weight=1.0)

    if with_energy:
        model = mk_afdesign_model(
            protocol="binder",
            data_dir=PARAMS_DIR,
            energy_fn=energy_fn,
            energy_weight=1.0,
        )
    else:
        model = mk_afdesign_model(protocol="binder", data_dir=PARAMS_DIR)

    model.prep_inputs(
        pdb_filename=PDB,
        chain="A",
        binder_len=15,
        hotspot=HOTSPOT_STR,
        rm_target_seq=False,
        model_names=["model_1"]
    )

    def step_callback(m):
        log = m.aux.get("log", {})
        entry = {}
        for k, v in log.items():
            try:
                entry[k] = float(v)
            except (TypeError, ValueError):
                pass

        seq_probs = m.aux["seq"]["pseudo"]  # (1, L, 20)
        entry["mean_prob_entropy"] = float(
            -jnp.mean(jnp.sum(seq_probs * jnp.log(seq_probs + 1e-8), axis=-1))
        )

        try:
            entry["seq"] = m.get_seqs()[0]
        except Exception:
            pass

        if "energy" in log:
            entry["energy"] = float(log["energy"])

        results.append(entry)

    model.restart(seed=seed)
    model.design_3stage(
        soft_iters=2,
        temp_iters=2,
        hard_iters=1,
        callback=step_callback,
        verbose=10,
    )

    # Recompute energy on the final soft sequence probabilities.
    final_seq_probs = model.aux["seq"]["pseudo"]
    final_energy = float(energy_fn(final_seq_probs))

    return results, model.get_seqs()[0], final_energy


def main():
    print("=" * 60)
    print("Run WITHOUT energy guidance")
    print("=" * 60)
    results_base, seq_base, energy_base = run_design(with_energy=False, seed=42)

    out_base = os.path.join(OUT_DIR, "model_energy_results_baseline.json")
    with open(out_base, "w") as f:
        json.dump(results_base, f, indent=2)
    print(f"Saved baseline results to {out_base}")

    print("\n" + "=" * 60)
    print(f"Run WITH energy guidance (weights: {WEIGHTS_PATH})")
    print("=" * 60)
    results_guided, seq_guided, energy_guided = run_design(with_energy=True, seed=42)

    out_guided = os.path.join(OUT_DIR, "model_energy_results_guided.json")
    with open(out_guided, "w") as f:
        json.dump(results_guided, f, indent=2)
    print(f"Saved guided results to {out_guided}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline sequence : {seq_base}")
    print(f"Guided   sequence : {seq_guided}")

    print(f"\nFinal energy (baseline) : {energy_base:.4f}")
    print(f"Final energy (guided)   : {energy_guided:.4f}")


if __name__ == "__main__":
    main()
