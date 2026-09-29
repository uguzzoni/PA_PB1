"""
run_1_8_baseline_bootstrap.py

Fase 1.8 del piano di design (Workplan_phase1_addendum.md §1.8): robustezza
statistica del confronto fra baseline di §1.4. La conclusione di §1.4 ("solo
l'MLP a piena capacita' generalizza in modo non banale") poggia su N=19
sequenze di test -- con N=19 l'IC al 95% di r=0.34 e' circa [-0.14, 0.69],
non distinguibile da r=0 ne' da r=-0.06. Qui si quantifica quell'incertezza
con un bootstrap non parametrico e, soprattutto, un CONFRONTO APPAIATO
(bootstrap della differenza r_MLP - r_baseline ricampionando le STESSE
sequenze per entrambi i modelli), che e' il test che conta davvero perche'
elimina la varianza dovuta a quali sequenze finiscono nel test set.

Prerequisito: eseguire manualmente
PD_energy_model/training/training_2rounds/workplan_phase1/dump_baseline_test_predictions.jl
(Julia).

Usage:
    uv run python src/phase1_diagnostics/run_1_8_baseline_bootstrap.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED_DIR = os.path.join(REPO_ROOT, "PD_energy_model", "training", "training_2rounds",
                         "workplan_phase1", "exported_predictions")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase1", "1_8_baseline_bootstrap")

N_BOOTSTRAP = 10_000
SEED = 0

MODELS = [
    ("A_baseline1_composition", "composizione (baseline 1)"),
    ("A_baseline2_onehot", "one-hot additivo (baseline 2)"),
    ("A_mlp_production", "MLP produzione (§1.4)"),
    ("B_baseline1_composition", "composizione, libraryF vincolato (sez. B)"),
    ("B_baseline2_onehot", "one-hot, libraryF vincolato (sez. B)"),
]
REFERENCE_MODEL = "A_mlp_production"
# Il terzo campo e' la direzione ATTESA del miglioramento (r_MLP - r_baseline): per la
# selectivity un modello migliore ha r PIU' POSITIVO (arricchimento e selectivity attesa si
# muovono nella stessa direzione); per l'energia un modello migliore ha r PIU' NEGATIVO
# (energia piu' bassa -> arricchimento maggiore, convenzione usata ovunque in questo progetto,
# es. §1.4 "energia Pearson -0.34" riportato come segnale, non come anomalia). Un controllo
# ingenuo "diff>0" sbaglierebbe sistematicamente il verdetto sull'energia.
METRICS = [("energy_selection_pred", "energia", False), ("selectivity_pred", "selectivity", True)]


def load_predictions():
    frames = {}
    for label, _ in MODELS:
        path = os.path.join(PRED_DIR, f"1_4_per_sequence_{label}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} mancante -- eseguire prima dump_baseline_test_predictions.jl (manuale, "
                "PD_energy_model/training/training_2rounds/workplan_phase1/)")
        frames[label] = pd.read_csv(path)
    return frames


def align_on_sequence(df_a: pd.DataFrame, df_b: pd.DataFrame, split: str, metric_col: str):
    """Le due tabelle possono differire nel filtro counts>10 realizzato (stessa soglia, ma
    applicata a predizioni di modelli diversi -> nessuna differenza di soglia qui, solo
    verifica difensiva) -- si allinea per sequenza per garantire lo stesso set esatto in
    entrambi i bracci del confronto appaiato."""
    a = df_a[df_a["split"] == split][["sequence", metric_col, "enrichment_obs"]].rename(
        columns={metric_col: "pred_a"})
    b = df_b[df_b["split"] == split][["sequence", metric_col, "enrichment_obs"]].rename(
        columns={metric_col: "pred_b"})
    merged = a.merge(b, on="sequence", suffixes=("_a", "_b"))
    assert np.allclose(merged["enrichment_obs_a"], merged["enrichment_obs_b"]), \
        "enrichment_obs diverge fra i due modelli sulla stessa sequenza -- dump Julia incoerente"
    return merged["pred_a"].to_numpy(), merged["pred_b"].to_numpy(), merged["enrichment_obs_a"].to_numpy()


def bootstrap_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOTSTRAP):
    n = len(x)
    boot_r = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_r[b] = np.corrcoef(x[idx], y[idx])[0, 1]
    return boot_r


def bootstrap_paired_diff(pred_a: np.ndarray, pred_b: np.ndarray, obs: np.ndarray,
                           rng: np.random.Generator, n_boot: int = N_BOOTSTRAP):
    """Ricampiona LE STESSE sequenze (stessi indici) per entrambi i modelli ad ogni
    ricampionamento -- questo e' il confronto che elimina la varianza dovuta a quali
    sequenze sono nel test set (addendum §1.8 passo B, punto 3)."""
    n = len(obs)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r_a = np.corrcoef(pred_a[idx], obs[idx])[0, 1]
        r_b = np.corrcoef(pred_b[idx], obs[idx])[0, 1]
        diffs[b] = r_a - r_b
    return diffs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = load_predictions()
    rng = np.random.default_rng(SEED)

    ci_rows, diff_rows = [], []
    for split in ("train", "test"):
        for metric_col, metric_label, expect_positive_diff in METRICS:
            ref_df = frames[REFERENCE_MODEL]
            for label, model_label in MODELS:
                pred_ref, pred_model, obs = align_on_sequence(ref_df, frames[label], split, metric_col)
                n = len(obs)

                point_r = np.corrcoef(pred_model, obs)[0, 1]
                boot_r = bootstrap_ci(pred_model, obs, rng)
                ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
                ci_rows.append({"split": split, "metric": metric_label, "model": label,
                                 "n": n, "pearson": point_r, "ci_lo_95": ci_lo, "ci_hi_95": ci_hi})

                if label == REFERENCE_MODEL:
                    continue
                # diffs = r_MLP - r_baseline per ricampionamento (bootstrap_paired_diff(pred_a=ref, pred_b=baseline, ...))
                diffs = bootstrap_paired_diff(pred_ref, pred_model, obs, rng)
                point_diff = np.corrcoef(pred_ref, obs)[0, 1] - point_r  # r_MLP - r_baseline, stesso segno di diffs
                frac_expected_sign = float((diffs > 0).mean()) if expect_positive_diff else float((diffs < 0).mean())
                diff_rows.append({
                    "split": split, "metric": metric_label,
                    "baseline": label, "reference": REFERENCE_MODEL,
                    "n": n, "point_diff": point_diff, "expected_diff_sign": "+" if expect_positive_diff else "-",
                    "frac_expected_sign": frac_expected_sign,
                    "ci_lo_95": np.percentile(diffs, 2.5), "ci_hi_95": np.percentile(diffs, 97.5),
                    "discriminates_at_95pct": frac_expected_sign >= 0.95,
                })

    ci_df = pd.DataFrame(ci_rows)
    diff_df = pd.DataFrame(diff_rows)
    ci_df.to_csv(os.path.join(OUT_DIR, "bootstrap_ci.csv"), index=False)
    diff_df.to_csv(os.path.join(OUT_DIR, "paired_differences.csv"), index=False)

    print("Intervalli di confidenza al 95% (percentile bootstrap):")
    print(ci_df.to_string(index=False))
    print("\nConfronto appaiato MLP - baseline (frazione di ricampionamenti con segno atteso):")
    print(diff_df.to_string(index=False))

    # --- criterio di lettura fissato dall'addendum PRIMA di guardare i risultati ---
    test_rows = diff_df[diff_df["split"] == "test"]
    conclusion_holds = bool(test_rows["discriminates_at_95pct"].all())
    print(f"\nCriterio (addendum §1.8): differenza appaiata con segno atteso in >=95% dei "
          f"ricampionamenti, su TUTTI i confronti test -- {'SODDISFATTO: §1.4 regge' if conclusion_holds else 'NON SODDISFATTO: §1.4 non discrimina con questi dati'}")

    # --- controllo di correttezza dell'implementazione: sul train (N=126) le differenze
    # devono essere nette -- se non lo sono, il problema e' nel bootstrap, non nei dati.
    train_rows = diff_df[diff_df["split"] == "train"]
    train_check_ok = bool(train_rows["discriminates_at_95pct"].all())
    print(f"Controllo di correttezza (train, atteso netto): {'OK' if train_check_ok else 'ANOMALO -- verificare l implementazione del bootstrap prima di fidarsi del risultato sul test'}")

    # --- forest plot: IC al 95% per modello, split test, entrambe le metriche ---
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 4.5), squeeze=False)
    for col, (metric_col, metric_label, _) in enumerate(METRICS):
        ax = axes[0][col]
        sub = ci_df[(ci_df["split"] == "test") & (ci_df["metric"] == metric_label)]
        y = np.arange(len(sub))
        ax.errorbar(sub["pearson"], y,
                     xerr=[sub["pearson"] - sub["ci_lo_95"], sub["ci_hi_95"] - sub["pearson"]],
                     fmt="o", capsize=3)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["model"])
        ax.set_xlabel(f"Pearson r ({metric_label}, test, N={sub['n'].iloc[0] if len(sub) else '?'})")
        ax.set_title(f"IC 95% bootstrap -- {metric_label}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "forest_plot.png"), dpi=150)
    plt.close(fig)

    summary = {
        "n_bootstrap": N_BOOTSTRAP,
        "reference_model": REFERENCE_MODEL,
        "conclusion_1_4_holds_at_95pct_on_test": conclusion_holds,
        "train_sanity_check_ok": train_check_ok,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
