"""
run_1b_1_leverage_check.py

Fase 1-bis (Workplan_v2_phase1bis_instantiated.md §1b.1): pre-controllo di
degenerazione dello stimatore di Laplace sull'ultimo strato, da eseguire PRIMA
di costruire il resto della sottofase (§1b.2/§1b.3) -- costa minuti e puo'
chiudere l'intera fase.

Calcola la leva h(x) = phi(x)^T (Phi^T Phi + lam I)^-1 phi(x)
(Workplan_v2_fasi_1bis_2_3.md §1b.1) sul modello di PRODUZIONE soltanto
(non le repliche -- quelle entrano solo in §1b.2/§1b.3), su quattro
popolazioni, confrontata con l'attesa teorica h_bar = p/N.

Usage:
    JAX_PLATFORMS=cpu uv run python src/phase1bis_diagnostics/run_1b_1_leverage_check.py
"""

import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORK_ROOT = "/home/guido/Projects/protein_design/methods/colabdesign_energy_guidance"
WEIGHTS_PATH = os.path.join(REPO_ROOT, "data", "energy_model_params",
                            "PNB_2R_3lay_negbinom_energy_model_weights.json")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase1bis", "1b_1_leverage_check")
CLUSTER_ASSIGNMENT_CSV = os.path.join(REPO_ROOT, "results", "phase0", "0_2_clustering", "cluster_assignment.csv")
HAMMING_REFERENCE_CSV = os.path.join(REPO_ROOT, "results", "phase1", "1_10_variance_diagnostics",
                                     "variance_by_population.csv")
LAMBDA_SMALL = 1e-6  # placeholder solo per questo pre-controllo -- non il lambda calibrato di §1b.2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _features import compute_leverage, extract_phi_batch  # noqa: E402

sys.path.insert(0, os.path.join(REPO_ROOT, "src", "phase0_diagnostics"))
from data_loading import (load_bli_population, load_designed_sequences,  # noqa: E402
                          load_training_sequences_onehot, try_encode_sequence)

sys.path.insert(0, FORK_ROOT)
from colabdesign.energy_model.model_3layer_v2 import load_energy_model  # noqa: E402


def load_cluster_representatives_90pct():
    """Stessa logica di run_1_10_variance_diagnostics.py (Fase 1, §1.10): rappresentanti
    di cluster che coprono il 90.1% della massa di read in round 3, cluster_read_mass
    decrescente -- ricostruito da cluster_assignment.csv, non hardcoded."""
    df = pd.read_csv(CLUSTER_ASSIGNMENT_CSV)
    reps = df[df["is_representative"]].sort_values("cluster_read_mass", ascending=False).reset_index(drop=True)
    total_mass = df.groupby("cluster_id")["cluster_read_mass"].first().sum()
    cum = reps["cluster_read_mass"].cumsum() / total_mass
    n_reps = int(np.searchsorted(cum.to_numpy(), 0.901) + 1)
    return reps.iloc[:n_reps]["Sequence"].tolist()


def encode_population(sequences):
    encoded, kept = [], []
    for s in sequences:
        x = try_encode_sequence(s, allow_gap=False)
        if x is None:
            continue
        encoded.append(x)
        kept.append(s)
    return np.stack(encoded), kept


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Caricamento pesi del modello di produzione...")
    params = load_energy_model(WEIGHTS_PATH)

    print("Caricamento training set di produzione (F+R completo)...")
    train_onehot, train_seqs, n_dropped = load_training_sequences_onehot()
    n_train = len(train_seqs)
    print(f"  N={n_train} (scartate {n_dropped})")

    print("Costruzione di Phi e della matrice 16x16 sul training set...")
    phi_train = extract_phi_batch(params, train_onehot)
    p_dim = phi_train.shape[1]
    h_bar_theoretical = p_dim / n_train
    print(f"  p={p_dim}, N={n_train}, h_bar teorico = p/N = {h_bar_theoretical:.6e}")

    h_train = compute_leverage(phi_train, phi_train, LAMBDA_SMALL)
    h_bar_observed = float(h_train.mean())
    rel_err = abs(h_bar_observed - h_bar_theoretical) / h_bar_theoretical
    print(f"  h_bar osservato sul training set stesso = {h_bar_observed:.6e} "
          f"(controllo di correttezza, deve coincidere entro l'errore numerico per lambda->0)")
    print(f"  errore relativo = {rel_err:.2%}")
    if rel_err > 0.05:
        raise RuntimeError(
            f"h_bar osservato diverge dall'atteso teorico p/N oltre il 5% (err={rel_err:.2%}) -- "
            "verificare extract_phi_batch/compute_leverage prima di fidarsi del resto.")

    print("Caricamento popolazione BLI (23 sequenze)...")
    bli, _, _ = load_bli_population()
    bli_onehot, bli_seqs = encode_population(bli["Sequence"].tolist())

    print("Caricamento rappresentanti di cluster (90.1% massa round 3)...")
    cluster_reps = load_cluster_representatives_90pct()
    cluster_onehot, cluster_seqs = encode_population(cluster_reps)
    print(f"  {len(cluster_seqs)} rappresentanti")

    print("Caricamento design di Fase 0 (tutte le campagne ColabDesign)...")
    designs_df = load_designed_sequences()
    design_onehot, design_seqs = encode_population(designs_df["Sequence"].tolist())
    print(f"  {len(design_seqs)} design validi su {len(designs_df)} totali")

    populations = {
        "training_set": (train_onehot, train_seqs),
        "bli_23": (bli_onehot, bli_seqs),
        "cluster_representative": (cluster_onehot, cluster_seqs),
        "fase0_designs": (design_onehot, design_seqs),
    }

    rows, summary_by_pop = [], {}
    for label, (onehot, seqs) in populations.items():
        phi = extract_phi_batch(params, onehot)
        h = compute_leverage(phi, phi_train, LAMBDA_SMALL)
        ratio = h / h_bar_theoretical
        for seq, hv, r in zip(seqs, h, ratio):
            rows.append({"population": label, "sequence": seq, "h": float(hv), "h_over_hbar": float(r)})
        summary_by_pop[label] = {
            "n": len(seqs),
            "h_over_hbar_median": float(np.median(ratio)),
            "h_over_hbar_max": float(np.max(ratio)),
            "frac_over_100x": float((ratio > 100).mean()),
            "frac_over_1000x": float((ratio > 1000).mean()),
        }
        s = summary_by_pop[label]
        print(f"  {label}: N={s['n']}  h/h_bar mediana={s['h_over_hbar_median']:.2f}  "
              f"max={s['h_over_hbar_max']:.2f}  frazione>100x={s['frac_over_100x']:.2%}  "
              f"frazione>1000x={s['frac_over_1000x']:.2%}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, "leverage_by_population.csv"), index=False)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
    for ax, label in zip(axes, populations):
        vals = df_out.loc[df_out["population"] == label, "h_over_hbar"].clip(lower=1e-3)
        ax.hist(np.log10(vals), bins=30, color="steelblue", alpha=0.8)
        ax.axvline(0, color="black", lw=1, ls="--", label=r"$h/\bar h=1$")
        ax.set_xlabel(r"$\log_{10}(h/\bar h)$")
        ax.set_title(f"{label} (N={len(vals)})")
    axes[0].set_ylabel("# sequenze")
    axes[0].legend(fontsize=8)
    fig.suptitle(r"Leva relativa all'attesa teorica $\bar h = p/N$ -- scala log10")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "leverage_ratio_histograms.png"), dpi=150)
    plt.close(fig)

    # --- nota sui proxy di distanza: riusa la distanza di Hamming gia' calcolata in Fase 1 (§1.10) ---
    hamming_note = None
    if os.path.exists(HAMMING_REFERENCE_CSV):
        hdf = pd.read_csv(HAMMING_REFERENCE_CSV)
        d = hdf["hamming_dist_to_training"].dropna()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.hist(d, bins=range(0, 16), color="firebrick", alpha=0.8)
        ax.set_xlabel("distanza di Hamming minima dal training set")
        ax.set_ylabel("# punti (popolazioni di §1.10: Dirichlet, BLI, rappresentanti di cluster)")
        ax.set_title("Verifica della saturazione attesa (§1b.1, nota sui proxy di distanza)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "hamming_saturation_check.png"), dpi=150)
        plt.close(fig)
        frac_8_10 = float(((d >= 8) & (d <= 10)).mean())
        hamming_note = {
            "source": HAMMING_REFERENCE_CSV, "median": float(d.median()),
            "min": float(d.min()), "max": float(d.max()), "frac_between_8_and_10": frac_8_10,
            "reading": ("saturazione confermata -- il fallimento del criterio di distanza di Hamming in "
                        "Fase 1 §1.10 era non informativo, non un esito negativo" if frac_8_10 > 0.5 else
                        "saturazione NON confermata come atteso -- rivedere l'interpretazione di §1.10"),
        }
        print(f"\nDistanza di Hamming (dati di §1.10): mediana={hamming_note['median']:.1f}, "
              f"frazione in [8,10]={frac_8_10:.1%} -- {hamming_note['reading']}")
    else:
        print(f"\n{HAMMING_REFERENCE_CSV} non trovato -- salto la verifica di saturazione.")

    # Criterio (Workplan_v2_fasi_1bis_2_3.md §1b.1): "h/h_bar resta entro un ordine di
    # grandezza su tutte le popolazioni" si legge sul comportamento TIPICO (mediana), non
    # sul singolo outlier massimo -- un massimo isolato e' normale nella distribuzione della
    # leva anche quando lo stimatore non discrimina affatto. "Supera 2-3 ordini di grandezza
    # su una FRAZIONE APPREZZABILE" e' catturato da frac_over_100x, non dal massimo.
    appreciable_fraction_threshold = 0.05  # soglia per "frazione apprezzabile", non data numericamente dal piano madre
    median_within_one_oom = all(s["h_over_hbar_median"] < 10 for s in summary_by_pop.values())
    appreciable_high_leverage_designs = (
        summary_by_pop["fase0_designs"]["frac_over_100x"] > appreciable_fraction_threshold)
    degenerate = median_within_one_oom and not appreciable_high_leverage_designs

    summary = {
        "p": p_dim, "n_train": n_train, "h_bar_theoretical": h_bar_theoretical,
        "h_bar_observed_on_training_set": h_bar_observed, "relative_error": rel_err,
        "lambda_used_placeholder": LAMBDA_SMALL,
        "by_population": summary_by_pop,
        "hamming_saturation_check": hamming_note,
        "appreciable_fraction_threshold": appreciable_fraction_threshold,
        "criterion_median_within_one_order_of_magnitude_all_populations": median_within_one_oom,
        "criterion_appreciable_fraction_designs_over_100x": appreciable_high_leverage_designs,
        "verdict_degenerate": degenerate,
        "next_step": ("§1b.4 con esito 'degenerazione' -- rimandare a Fase 4, non scrivere §1b.2/§1b.3"
                      if degenerate else "procedere a §1b.2"),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nVerdetto: {'DEGENERE' if degenerate else 'NON degenere'} -- {summary['next_step']}")
    print(f"Output scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
