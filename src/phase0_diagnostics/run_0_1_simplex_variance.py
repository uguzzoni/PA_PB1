"""
run_0_1_simplex_variance.py

Fase 0.1 del piano di design (Workplan_phase0_instantiated.md §0.1): quantifica
di quanto il forward soft E(a) diverge dai valori che il modello di energia
assegna a sequenze reali del training set, campionando l'interno del simplesso
con Dirichlet a diversa concentrazione.

Usage:
    uv run python src/phase0_diagnostics/run_0_1_simplex_variance.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import BINDER_LEN, REPO_ROOT, encode_sequence, load_training_sequences_onehot

from colabdesign.energy_model.model_3layer import load_energy_model, mlp_forward

WEIGHTS_PATH = os.path.join(
    REPO_ROOT, "data", "energy_model_params", "PNB_2R_3lay_negbinom_energy_model_weights.json"
)
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_1_simplex_variance")

N_ALPHA = 25
N_SAMPLES_PER_ALPHA = 400  # 25 * 400 = 10_000, per il piano
ALPHA_MIN, ALPHA_MAX = 0.05, 50.0
SEED = 0

# Energie già calcolate dal vero modello (via ColabDesign/make_energy_fn) su queste
# 6 sequenze — results/colabdesign/custom/seqs_aff_alberto_darren_6best.json.
# Usate per validare che l'encoder diretto (ordine Julia, senza passare da ColabDesign)
# riproduca esattamente il percorso via AF-order -> Julia-order permutation.
VALIDATION_SEQS = {
    "MDFNPWLLFLKVPAQ": -1.0169219970703125,
    "VDFNPWLLFLKVPAQ": -1.1298952102661133,
    "IDFNPYLLFLKVPAQ": 0.9208765625953674,
    "MDWNPLLLHLKRPAQ": 3.178508758544922,
    "VDYNPWLLFLRKPKQ": 1.1072289943695068,
    "IDYNPYLLFLKQPKQ": -1.1511486768722534,
}
VALIDATION_TOL = 1e-4


def validate_encoder(params) -> None:
    print("Validazione encoder contro le energie già calcolate da ColabDesign...")
    max_err = 0.0
    for seq, expected in VALIDATION_SEQS.items():
        x = encode_sequence(seq, allow_gap=False)[None]  # (1, 15, 21)
        got = float(mlp_forward(params, jnp.asarray(x)))
        err = abs(got - expected)
        max_err = max(max_err, err)
        print(f"  {seq}: expected={expected:.6f} got={got:.6f} |err|={err:.2e}")
    if max_err > VALIDATION_TOL:
        raise RuntimeError(
            f"Validazione FALLITA: |err| massimo={max_err:.2e} > {VALIDATION_TOL:.0e}. "
            "Sospettare l'ordine di flattening o l'alfabeto in data_loading.encode_sequence, "
            "non il caricamento dei pesi (vedi Workplan_phase0_instantiated.md §Perimetro)."
        )
    print(f"OK — |err| massimo = {max_err:.2e}\n")


def dirichlet_alpha_grid(n_alpha: int = N_ALPHA) -> np.ndarray:
    return np.logspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), n_alpha)


def sample_interior_points(rng: np.random.Generator, n_alpha: int = N_ALPHA,
                            n_per_alpha: int = N_SAMPLES_PER_ALPHA):
    """Dirichlet(alpha * 1_20) indipendente per posizione, colonna gap (21esima)
    fissata a zero — replica cosa il modello vede in uso reale (design), non lo
    spazio a 21 categorie del training set (vedi Workplan §Perimetro).

    Returns: points (n_alpha*n_per_alpha, 15, 21), alphas (stessa lunghezza).
    """
    alphas_grid = dirichlet_alpha_grid(n_alpha)
    all_points, all_alphas = [], []
    for alpha in alphas_grid:
        draws = rng.dirichlet(alpha * np.ones(20), size=(n_per_alpha, BINDER_LEN))  # (n,15,20)
        padded = np.concatenate(
            [draws, np.zeros((n_per_alpha, BINDER_LEN, 1))], axis=-1
        ).astype(np.float32)
        all_points.append(padded)
        all_alphas.append(np.full(n_per_alpha, alpha))
    return np.concatenate(all_points, axis=0), np.concatenate(all_alphas, axis=0)


def make_forward_batch(params):
    single = lambda xi: mlp_forward(params, xi[None])  # xi: (15,21) -> (1,15,21)
    return jax.jit(jax.vmap(single))


def vertices_from_soft(points: np.ndarray) -> np.ndarray:
    """One-hot del argmax per posizione (sulle 20 categorie reali, gap escluso)."""
    argmax_idx = points[..., :20].argmax(axis=-1)  # (N, 15)
    vertex = np.zeros_like(points)
    n = points.shape[0]
    for pos in range(BINDER_LEN):
        vertex[np.arange(n), pos, argmax_idx[:, pos]] = 1.0
    return vertex


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    params = load_energy_model(WEIGHTS_PATH)
    validate_encoder(params)
    forward_batch = make_forward_batch(params)

    print("Caricamento training set (4_counts: F2+F3+R2+R3)...")
    train_onehot, train_seqs, n_dropped = load_training_sequences_onehot()
    print(f"  {len(train_seqs)} sequenze valide, {n_dropped} scartate (residui non standard)")

    train_E = np.asarray(forward_batch(jnp.asarray(train_onehot)))
    E_min, E_max = float(train_E.min()), float(train_E.max())
    sigma_train = float(train_E.std())
    print(f"  E_min={E_min:.4f}  E_max={E_max:.4f}  sigma_train={sigma_train:.4f}\n")

    print(f"Campionamento {N_ALPHA * N_SAMPLES_PER_ALPHA} punti Dirichlet sull'interno del simplesso...")
    rng = np.random.default_rng(SEED)
    points, alphas = sample_interior_points(rng)

    E_soft = np.asarray(forward_batch(jnp.asarray(points)))
    vertices = vertices_from_soft(points)
    E_vertex = np.asarray(forward_batch(jnp.asarray(vertices)))

    violation = (E_soft < E_min) | (E_soft > E_max)
    violation_amplitude = np.where(
        E_soft < E_min,
        (E_min - E_soft) / sigma_train,
        np.where(E_soft > E_max, (E_soft - E_max) / sigma_train, 0.0),
    )

    metrics = {
        "n_train_sequences": len(train_seqs),
        "n_train_dropped": int(n_dropped),
        "E_min_train": E_min,
        "E_max_train": E_max,
        "sigma_train": sigma_train,
        "n_interior_points": int(len(points)),
        "violation_fraction": float(violation.mean()),
        "violation_amplitude_max_sigma": float(violation_amplitude.max()),
        "violation_amplitude_median_sigma_among_violations": (
            float(np.median(violation_amplitude[violation])) if violation.any() else 0.0
        ),
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # --- scatter E(a) vs E(vertice), colorato per alpha ---
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(E_vertex, E_soft, c=np.log10(alphas), s=4, alpha=0.4, cmap="viridis")
    lo = min(E_vertex.min(), E_soft.min())
    hi = max(E_vertex.max(), E_soft.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identità")
    ax.axhline(E_min, color="red", ls=":", lw=1, label="E_min/E_max (training)")
    ax.axhline(E_max, color="red", ls=":", lw=1)
    ax.set_xlabel("E(one-hot(argmax a))")
    ax.set_ylabel("E(a)  [soft]")
    ax.legend(fontsize=8, loc="best")
    plt.colorbar(sc, label="log10(alpha)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scatter.png"), dpi=150)
    plt.close(fig)

    # --- violazione (frazione + ampiezza mediana) vs alpha ---
    alphas_grid = dirichlet_alpha_grid()
    frac_by_alpha, amp_by_alpha = [], []
    for alpha in alphas_grid:
        mask = np.isclose(alphas, alpha)
        frac_by_alpha.append(violation[mask].mean())
        vals = violation_amplitude[mask]
        pos_vals = vals[vals > 0]
        amp_by_alpha.append(np.median(pos_vals) if pos_vals.size else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(alphas_grid, frac_by_alpha, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("alpha (concentrazione Dirichlet)")
    axes[0].set_ylabel("frazione di violazioni")
    axes[1].plot(alphas_grid, amp_by_alpha, marker="o", color="darkorange")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("alpha (concentrazione Dirichlet)")
    axes[1].set_ylabel("ampiezza mediana violazione (sigma_train)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "violation_vs_alpha.png"), dpi=150)
    plt.close(fig)

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
