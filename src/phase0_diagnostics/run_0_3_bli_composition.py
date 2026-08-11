"""
run_0_3_bli_composition.py

Fase 0.3' del piano di design (Workplan_phase0_instantiated.md §0.3'):
composizione amminoacidica dei leganti validati in BLI, confrontata con WT,
libreria iniziale (round 1) e training set stratificato per arricchimento
(proxy = abbondanza normalizzata in round 3, l'alternativa "più semplice"
indicata nel piano rispetto al rapporto round3/round2 con pseudocount).

Nota sui dati BLI (non ovvia dal solo conteggio "39 righe" del piano
originale): `data/affinity_measurements_PA-PB1_formatted.csv` ha 39 RIGHE ma
solo 28 sequenze uniche — molte sono misure ripetute della stessa sequenza in
round sperimentali diversi (scouting preliminare, "best hits", side-by-side
finale, confronto con MPNN). Inoltre:
  - 4 righe hanno Kd_nM = NA (nessun segnale misurabile): P7, P10 (una delle
    sue righe, le altre 2 sono valide), MPNN_06, MPNN_07.
  - Il nome "WT" compare su 2 sequenze DIVERSE: MDVNPTLLFLKLPAQ (2 righe) e
    MDVNPTLLFLKVPAQ (1 riga, posizione 12: L vs V) — quest'ultima è quella
    hardcoded nel piano originale. Sembra un typo di trascrizione in 2 righe
    su 3, non due costrutti WT distinti — segnalato esplicitamente in output
    (wt_discrepancy in composition_table dropped-rows log), non silenziato.

Per la popolazione "BLI" si usa quindi: dedup per Sequence (non per Name),
Kd_nM aggregata come media geometrica sulle misure valide della stessa
sequenza, righe "WT" escluse dalla popolazione dei leganti disegnati (la WT
è trattata come popolazione a sé, 1 sola sequenza, per costruzione).
N effettivo per la composizione BLI: vedi output (atteso ~23, non 39).

Usage:
    uv run python src/phase0_diagnostics/run_0_3_bli_composition.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import (AAS_JULIA, REPO_ROOT, WT_SEQ, load_bli_population, load_round1_counts,
                           load_round3_counts, load_training_counts, try_encode_sequence)

OUT_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_3_bli_composition")
STANDARD_AA = list(AAS_JULIA[:20])
N_DECILES = 10


def filter_standard(df: pd.DataFrame, seq_col: str = "Sequence"):
    """Tiene solo sequenze sui 20 AA standard (niente stop codon '*' /
    ambiguità NGS come '_') — per la composizione amminoacidica un read con
    stop codon non è un 15-mero reale, va escluso (diversamente da 0.1 dove
    il canale gap è valido input per il modello di energia)."""
    mask = df[seq_col].apply(lambda s: try_encode_sequence(s, allow_gap=False) is not None)
    return df[mask].copy(), int((~mask).sum())


def aa_composition(seqs, weights=None) -> pd.Series:
    """Frequenza per lettera (20 categorie), pesata per `weights` (es. Count
    NGS) se fornito, altrimenti un peso di 1 per sequenza (popolazioni di
    costrutti individuali come BLI/WT, non pool di read)."""
    if weights is None:
        weights = np.ones(len(seqs))
    counts = pd.Series(0.0, index=STANDARD_AA)
    for seq, w in zip(seqs, weights):
        for c in seq:
            counts[c] += w
    return counts / counts.sum()


def aromatic_fraction(freq: pd.Series) -> float:
    return float(freq[["F", "W", "Y"]].sum())


def training_enrichment_deciles():
    """Training set (0.2: union F2+F3+R2+R3), stratificato in decili per
    abbondanza normalizzata in round 3 (proxy di arricchimento, alternativa
    più semplice al rapporto round3/round2 con pseudocount — vedi piano
    §0.3', scelta qui per rapidità).

    Sequenze assenti in round 3 (proxy=0, la maggioranza: overlap
    round2/round3 = 2425 su 23.677+6.154 unici) finiscono nei decili più
    bassi. Rank(method='first') invece di qcut diretto sul valore: con
    ~78% di proxy=0 (assenti da round3) i bin per valore collasserebbero,
    method='first' forza comunque 10 bin di uguale numerosità.
    """
    train_raw = load_training_counts()
    train, n_dropped = filter_standard(train_raw)

    r3 = load_round3_counts()
    r3_total = int(r3["Count"].sum())
    r3_map = dict(zip(r3["Sequence"], r3["Count"]))
    train["enrichment_proxy"] = train["Sequence"].map(lambda s: r3_map.get(s, 0) / r3_total)

    rank = train["enrichment_proxy"].rank(method="first")
    train["decile"] = pd.qcut(rank, N_DECILES, labels=False)  # 0 = meno arricchito, 9 = più arricchito
    return train, n_dropped


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg, senza dipendere da statsmodels (non nelle deps)."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def enrichment_tests(pop_seqs, pop_weights, lib_seqs, lib_weights, pop_label: str) -> pd.DataFrame:
    """Fisher esatto per amminoacido: conteggio AA vs resto, popolazione vs
    libreria iniziale (round 1)."""
    pop_counts = pd.Series(0, index=STANDARD_AA, dtype=np.int64)
    for seq, w in zip(pop_seqs, pop_weights):
        for c in seq:
            pop_counts[c] += w
    lib_counts = pd.Series(0, index=STANDARD_AA, dtype=np.int64)
    for seq, w in zip(lib_seqs, lib_weights):
        for c in seq:
            lib_counts[c] += w

    pop_total, lib_total = int(pop_counts.sum()), int(lib_counts.sum())
    rows = []
    for aa in STANDARD_AA:
        table = [[int(pop_counts[aa]), pop_total - int(pop_counts[aa])],
                 [int(lib_counts[aa]), lib_total - int(lib_counts[aa])]]
        odds_ratio, pvalue = fisher_exact(table)
        rows.append({
            "population": pop_label, "amino_acid": aa,
            "count_in_population": int(pop_counts[aa]), "total_in_population": pop_total,
            "count_in_library": int(lib_counts[aa]), "total_in_library": lib_total,
            "odds_ratio": odds_ratio, "pvalue": pvalue,
        })
    out = pd.DataFrame(rows)
    out["pvalue_bh"] = bh_adjust(out["pvalue"].to_numpy())
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Caricamento popolazione BLI...")
    bli, excluded_no_kd, wt_discrepancy = load_bli_population()
    print(f"  {len(bli)} sequenze disegnate con Kd valido (dedup per Sequence, media geometrica su misure ripetute)")
    print(f"  escluse per Kd sempre NA: {excluded_no_kd}")
    print(f"  {wt_discrepancy['note']}\n")

    print("Caricamento libreria iniziale (round 1)...")
    lib_raw = load_round1_counts()
    lib, lib_dropped = filter_standard(lib_raw)
    print(f"  {len(lib)} sequenze uniche valide ({lib_dropped} scartate per stop/ambiguità), {int(lib['Count'].sum())} reads\n")

    print("Caricamento training set stratificato in decili di arricchimento...")
    train, train_dropped = training_enrichment_deciles()
    print(f"  {len(train)} sequenze valide ({train_dropped} scartate), 10 decili per abbondanza round3\n")

    # --- composizione per popolazione ---
    rows = []

    freq = aa_composition(bli["Sequence"])
    rows.append({"population": "BLI_all", "n_sequences": len(bli), "n_reads": None,
                 "aromatic_fraction": aromatic_fraction(freq), **freq.to_dict()})

    freq_wt = aa_composition([WT_SEQ])
    rows.append({"population": "WT", "n_sequences": 1, "n_reads": None,
                 "aromatic_fraction": aromatic_fraction(freq_wt), **freq_wt.to_dict()})

    freq_lib = aa_composition(lib["Sequence"], lib["Count"])
    rows.append({"population": "library_round1", "n_sequences": len(lib), "n_reads": int(lib["Count"].sum()),
                 "aromatic_fraction": aromatic_fraction(freq_lib), **freq_lib.to_dict()})

    for d in range(N_DECILES):
        sub = train[train["decile"] == d]
        freq_d = aa_composition(sub["Sequence"], sub["Count"])
        rows.append({"population": f"training_decile_{d}", "n_sequences": len(sub),
                     "n_reads": int(sub["Count"].sum()), "aromatic_fraction": aromatic_fraction(freq_d),
                     **freq_d.to_dict()})

    # --- stratificazione BLI per quartile di Kd (N piccolo, vedi caveat nel piano) ---
    bli_sorted = bli.sort_values("Kd_nM_geomean")
    try:
        bli_sorted["Kd_quartile"] = pd.qcut(bli_sorted["Kd_nM_geomean"], 4,
                                             labels=["Q1_tightest", "Q2", "Q3", "Q4_weakest"])
    except ValueError:
        bli_sorted["Kd_quartile"] = pd.cut(bli_sorted["Kd_nM_geomean"].rank(method="first"), 4,
                                            labels=["Q1_tightest", "Q2", "Q3", "Q4_weakest"])
    for q, sub in bli_sorted.groupby("Kd_quartile", observed=True):
        freq_q = aa_composition(sub["Sequence"])
        rows.append({"population": f"BLI_{q}", "n_sequences": len(sub), "n_reads": None,
                     "aromatic_fraction": aromatic_fraction(freq_q), **freq_q.to_dict()})

    composition_table = pd.DataFrame(rows)
    composition_table.to_csv(os.path.join(OUT_DIR, "composition_table.csv"), index=False)
    print("composition_table.csv scritto — N per quartile BLI:")
    print(bli_sorted.groupby("Kd_quartile", observed=True).size().to_string())
    print("(N totale = {}, ~5-6 per quartile: puramente descrittivo, non robusto — "
          "vedi piano §0.3' sulla scarsità di Kd_Error)\n".format(len(bli_sorted)))

    # --- test di arricchimento per amminoacido, contro la libreria iniziale ---
    top_decile = train[train["decile"] == N_DECILES - 1]
    tests = pd.concat([
        enrichment_tests(bli["Sequence"], np.ones(len(bli)), lib["Sequence"], lib["Count"], "BLI_all"),
        enrichment_tests(top_decile["Sequence"], top_decile["Count"], lib["Sequence"], lib["Count"],
                          "training_decile_9_top_enrichment"),
    ], ignore_index=True)
    tests.to_csv(os.path.join(OUT_DIR, "enrichment_test.csv"), index=False)
    n_sig = int((tests["pvalue_bh"] < 0.05).sum())
    print(f"enrichment_test.csv scritto — {n_sig}/{len(tests)} test con p_BH < 0.05")
    sig = tests[tests["pvalue_bh"] < 0.05].sort_values("pvalue_bh")
    if len(sig):
        print(sig[["population", "amino_acid", "odds_ratio", "pvalue_bh"]].to_string(index=False))

    # --- log di provenienza/scarti, per audit ---
    with open(os.path.join(OUT_DIR, "population_notes.json"), "w") as f:
        json.dump({
            "bli_n_sequences": len(bli),
            "bli_excluded_no_valid_kd": excluded_no_kd,
            "wt_discrepancy": wt_discrepancy,
            "library_round1_n_dropped_nonstandard": lib_dropped,
            "training_set_n_dropped_nonstandard": train_dropped,
            "enrichment_proxy_definition": "count_round3(seq) / total_reads_round3 (0 se assente da round3)",
        }, f, indent=2)

    # --- plot: frazione aromatica per popolazione ---
    plot_order = (["WT", "library_round1"] + [f"training_decile_{d}" for d in range(N_DECILES)]
                  + ["BLI_all"] + [f"BLI_{q}" for q in ["Q1_tightest", "Q2", "Q3", "Q4_weakest"]])
    plot_df = composition_table.set_index("population").loc[
        [p for p in plot_order if p in composition_table["population"].values]
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["gray", "black"] + ["steelblue"] * N_DECILES + ["darkorange"] + ["salmon"] * 4
    ax.bar(range(len(plot_df)), plot_df["aromatic_fraction"], color=colors[:len(plot_df)])
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df.index, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("frazione aromatica (F+W+Y)")
    ax.set_title("Frazione aromatica per popolazione: WT -> libreria -> decili di arricchimento -> BLI")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "aromatic_fraction_by_population.png"), dpi=150)
    plt.close(fig)

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
