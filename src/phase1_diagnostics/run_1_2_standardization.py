"""
run_1_2_standardization.py

Fase 1.2 del piano di design (Workplan_phase1_instantiated.md §1.2):
standardizzazione dell'energia, z = (E - mu_train) / sigma_train.
sigma_train era gia' noto da Fase 0 (0.1); qui si calcola mu_train (non
salvato in 0.1) e si verifica la standardizzazione, riusando
model_3layer_v2.make_energy_fn/make_energy_aux (§1.1) con gli argomenti
mu/sigma appena aggiunti.

Usage:
    JAX_PLATFORMS=cpu uv run python src/phase1_diagnostics/run_1_2_standardization.py
"""

import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORK_ROOT = "/home/guido/Projects/protein_design/methods/colabdesign_energy_guidance"
WEIGHTS_PATH = os.path.join(REPO_ROOT, "data", "energy_model_params",
                             "PNB_2R_3lay_negbinom_energy_model_weights.json")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase1", "1_2_standardization")

sys.path.insert(0, os.path.join(REPO_ROOT, "src", "phase0_diagnostics"))
from data_loading import load_training_sequences_onehot  # noqa: E402

sys.path.insert(0, FORK_ROOT)
from colabdesign.energy_model import model_3layer_v2 as v2  # noqa: E402

with open(os.path.join(REPO_ROOT, "results", "phase0", "0_1_simplex_variance", "metrics.json")) as f:
    metrics_01 = json.load(f)
SIGMA_TRAIN_01 = metrics_01["sigma_train"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Caricamento training set (stesso di 0.1) e calcolo mu_train/sigma_train...")
    train_onehot, train_seqs, n_dropped = load_training_sequences_onehot()
    params = v2.load_energy_model(WEIGHTS_PATH)
    forward_batch = jax.jit(jax.vmap(lambda xi: v2.mlp_forward(params, xi[None])))
    train_E = np.asarray(forward_batch(jnp.asarray(train_onehot)))

    mu_train = float(train_E.mean())
    sigma_train = float(train_E.std())
    print(f"  N={len(train_seqs)} (scartate {n_dropped})  mu_train={mu_train:.6f}  sigma_train={sigma_train:.6f}")

    # sigma_train ricalcolato qui deve combaciare con quello gia' salvato in 0.1
    # (stesso identico calcolo, train_E.std()) - discrepanza indicherebbe un
    # problema di riproducibilita', non di formula.
    sigma_match = abs(sigma_train - SIGMA_TRAIN_01) < 1e-6
    print(f"  sigma_train ricalcolato vs 0.1/metrics.json: {'coincide' if sigma_match else 'DIVERGENTE!'} "
          f"({sigma_train:.6f} vs {SIGMA_TRAIN_01:.6f})")
    if not sigma_match:
        raise RuntimeError("sigma_train ricalcolato diverge da quello di 0.1 — verificare prima di procedere")

    params_out = {"mu_train": mu_train, "sigma_train": sigma_train,
                  "n_train_sequences": len(train_seqs), "n_train_dropped": int(n_dropped)}
    with open(os.path.join(OUT_DIR, "standardization_params.json"), "w") as f:
        json.dump(params_out, f, indent=2)

    # --- verifica: z ha media~0, std~1 sul training set stesso ---
    z_train = (train_E - mu_train) / sigma_train
    print(f"\nVerifica su z = (E - mu_train)/sigma_train sul training set stesso:")
    print(f"  mean(z) = {z_train.mean():.2e} (atteso ~0)")
    print(f"  std(z)  = {z_train.std():.6f} (atteso ~1)")

    # --- verifica via make_energy_fn/make_energy_aux (mu/sigma appena aggiunti in v2) ---
    energy_fn_raw = v2.make_energy_fn(WEIGHTS_PATH, energy_weight=1.0, binder_len=15)
    energy_fn_std = v2.make_energy_fn(WEIGHTS_PATH, energy_weight=1.0, binder_len=15, mu=mu_train, sigma=sigma_train)
    aux_std_soft = v2.make_energy_aux(WEIGHTS_PATH, binder_len=15, mode="soft", mu=mu_train, sigma=sigma_train)
    aux_std_st = v2.make_energy_aux(WEIGHTS_PATH, binder_len=15, mode="st", mu=mu_train, sigma=sigma_train)

    rng = np.random.default_rng(0)
    AAS_AF = "ARNDCQEGHILKMFPSTWYV"
    max_err_formula = 0.0
    max_gap_st_std = 0.0
    for _ in range(50):
        seq_probs = jnp.asarray(rng.dirichlet(np.ones(20), size=(1, 15)).astype(np.float32))
        raw = float(energy_fn_raw(seq_probs))
        std = float(energy_fn_std(seq_probs))
        expected = (raw - mu_train) / sigma_train
        max_err_formula = max(max_err_formula, abs(std - expected))
        max_gap_st_std = max(max_gap_st_std, abs(float(aux_std_st(seq_probs)["gap"])))

    print(f"\nmake_energy_fn(mu=,sigma=) coerente con (raw-mu)/sigma: errore massimo = {max_err_formula:.2e}")
    print(f"gap in modalita' st, standardizzato, resta identicamente nullo: max|gap| = {max_gap_st_std:.2e}")

    # --- confronto qualitativo: range di z sui run reali di 0.7 (gia' in unita' di sigma_train) ---
    traj_path = os.path.join(REPO_ROOT, "results", "phase0", "0_7_trajectory_gap", "summary_scalars_by_run.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(z_train, bins=60, color="steelblue", alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(r"$\tilde E = (E - \mu_{\rm train})/\sigma_{\rm train}$")
    ax.set_ylabel("# sequenze (training set)")
    ax.set_title(f"Distribuzione di $\\tilde E$ sul training set (N={len(train_seqs)})\n"
                 f"mean={z_train.mean():.3f}, std={z_train.std():.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "z_distribution.png"), dpi=150)
    plt.close(fig)
    if os.path.exists(traj_path):
        print(f"\n(0.7 gia' riportava gap in unita' di sigma_train — stesso sigma_train qui, "
              f"vedi {traj_path} per il range osservato su run reali: mediane 5.7-29.5 sigma_train)")

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
