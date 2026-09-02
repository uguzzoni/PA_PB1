"""
run_0_8a_partial_correlation.py

Fase 0.8.a del piano di design (Workplan_phase0_instantiated.md §0.8.a): il
modello di energia contiene segnale predittivo su Kd oltre al solo conteggio
di residui aromatici? 0.3' ha mostrato che gli aromatici discriminano
selezionato/non selezionato ma non correlano con i quartili di Kd; qui si
verifica in continuo (N=23, non per quartili) se l'energia predice log(Kd)
anche dopo aver rimosso l'effetto lineare del conteggio aromatico.

Usage:
    uv run python src/phase0_diagnostics/run_0_8a_partial_correlation.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import REPO_ROOT, encode_sequence, load_bli_population

from colabdesign.energy_model.model_3layer import load_energy_model, mlp_forward

WEIGHTS_PATH = os.path.join(
    REPO_ROOT, "data", "energy_model_params", "PNB_2R_3lay_negbinom_energy_model_weights.json"
)
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_8_closing_checks")
AROMATIC = set("FWY")


def aromatic_count(seq: str) -> int:
    return sum(c in AROMATIC for c in seq)


def simple_regression(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    res = stats.linregress(x, y)
    return {
        "label": label, "n": len(x), "slope": res.slope, "intercept": res.intercept,
        "r": res.rvalue, "r_squared": res.rvalue ** 2, "pvalue": res.pvalue,
        "stderr": res.stderr,
    }


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residui di y dopo regressione lineare su x (via least squares con intercetta)."""
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def partial_correlation(y: np.ndarray, z: np.ndarray, control: np.ndarray) -> dict:
    """Correlazione parziale fra y e z, controllando per `control` (1 variabile)."""
    ry = residualize(y, control)
    rz = residualize(z, control)
    r, _ = stats.pearsonr(ry, rz)
    n = len(y)
    df = n - 3  # N - 2 (variabili) - 1 (variabile di controllo)
    t = r * np.sqrt(df / max(1 - r ** 2, 1e-12))
    pvalue = 2 * stats.t.sf(np.abs(t), df)
    return {"r_partial": r, "df": df, "t": t, "pvalue": pvalue, "n": n}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Caricamento popolazione BLI e modello di energia...")
    bli, _, _ = load_bli_population()
    params = load_energy_model(WEIGHTS_PATH)

    log_kd = np.log(bli["Kd_nM_geomean"].to_numpy())
    energy = np.array([
        float(mlp_forward(params, encode_sequence(s, allow_gap=False)[None]))
        for s in bli["Sequence"]
    ])
    arom = np.array([aromatic_count(s) for s in bli["Sequence"]], dtype=float)
    n = len(bli)
    print(f"N = {n} sequenze\n")

    reg_arom = simple_regression(arom, log_kd, "log(Kd) ~ conteggio_aromatico")
    reg_energy = simple_regression(energy, log_kd, "log(Kd) ~ energia")
    r_direct, p_direct = stats.pearsonr(energy, log_kd)
    partial = partial_correlation(log_kd, energy, arom)

    results = {
        "n_sequences": n,
        "regression_logKd_vs_aromatic": reg_arom,
        "regression_logKd_vs_energy": reg_energy,
        "pearson_energy_vs_logKd_direct": {"r": r_direct, "pvalue": p_direct},
        "partial_correlation_energy_vs_logKd_controlling_aromatic": partial,
    }
    with open(os.path.join(OUT_DIR, "0_8a_partial_correlation.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("log(Kd) ~ conteggio aromatico:")
    print(f"  r={reg_arom['r']:.3f}  R²={reg_arom['r_squared']:.3f}  p={reg_arom['pvalue']:.3g}")
    print("log(Kd) ~ energia (diretta, non controllata):")
    print(f"  r={r_direct:.3f}  R²={reg_energy['r_squared']:.3f}  p={p_direct:.3g}")
    print("Correlazione parziale energia vs log(Kd), controllando per conteggio aromatico:")
    print(f"  r_partial={partial['r_partial']:.3f}  df={partial['df']}  p={partial['pvalue']:.3g}")

    survives = partial["pvalue"] < 0.05
    print(f"\n{'SOPRAVVIVE' if survives else 'NON sopravvive'} (soglia p<0.05, N={n} — potenza comunque limitata)")

    # --- scatter: log(Kd) vs energia, colore = conteggio aromatico; e vs aromatico ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sc = axes[0].scatter(energy, log_kd, c=arom, cmap="viridis", s=40)
    xs = np.linspace(energy.min(), energy.max(), 50)
    axes[0].plot(xs, reg_energy["intercept"] + reg_energy["slope"] * xs, "k--", lw=1)
    axes[0].set_xlabel("energia (modello)")
    axes[0].set_ylabel("log(Kd_nM)")
    axes[0].set_title(f"r={r_direct:.2f}  (r_partial={partial['r_partial']:.2f})")
    plt.colorbar(sc, ax=axes[0], label="# residui aromatici")

    axes[1].scatter(arom, log_kd, color="darkorange", s=40)
    xs2 = np.linspace(arom.min(), arom.max(), 50)
    axes[1].plot(xs2, reg_arom["intercept"] + reg_arom["slope"] * xs2, "k--", lw=1)
    axes[1].set_xlabel("# residui aromatici (F+W+Y)")
    axes[1].set_ylabel("log(Kd_nM)")
    axes[1].set_title(f"r={reg_arom['r']:.2f}")

    fig.suptitle(f"N={n} sequenze BLI — 0.8.a correlazione parziale")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "0_8a_scatter_logKd_vs_energy_and_aromatic.png"), dpi=150)
    plt.close(fig)

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
