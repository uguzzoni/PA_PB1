"""
run_0_8c_gap_composition.py

Fase 0.8.c del piano di design (Workplan_phase0_instantiated.md §0.8.c): i
design la cui traiettoria soft ha aperto un gap ampio (0.7) finiscono con una
sequenza discreta più aromatica di quelli con gap piccolo, o dei controlli
senza energia?

Limite ereditato da 0.7, non aggirabile con questi dati: a^(t) non è salvato,
quindi "composizione di a^(t*)" nel senso letterale del piano madre non è
calcolabile. Si usa la composizione della SEQUENZA DISCRETA FINALE del
blocco (colonna `final_seq`, aggiunta a summary_scalars_by_run.csv apposta
per questa sottofase), non la sequenza al passo t* — un proxy dichiarato,
non la quantità originale.

Usage:
    uv run python src/phase0_diagnostics/run_0_8c_gap_composition.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import REPO_ROOT

TRAJ_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_7_trajectory_gap")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_8_closing_checks")
NO_ENERGY_PROTOCOLS = {"opt_anneal_noenergy"}  # controllo vero (w_E=0 su tutti gli stadi)


def aromatic_fraction(seq: str) -> float:
    return sum(c in "FWY" for c in seq) / len(seq)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    summary = pd.read_csv(os.path.join(TRAJ_DIR, "summary_scalars_by_run.csv"))
    summary["aromatic_fraction"] = summary["final_seq"].apply(aromatic_fraction)

    with_gap = summary.dropna(subset=["gap_max"]).copy()
    print(f"Blocchi con gap_max definito (stadio soft presente, traccia energia disponibile): {len(with_gap)}")
    print(f"Blocchi totali (incl. senza traccia soft o senza energia): {len(summary)}\n")

    # --- 1. quartili POOLED su tutti i protocolli insieme (letteralmente come da piano) ---
    with_gap["gap_quartile_pooled"] = pd.qcut(with_gap["gap_max"], 4, labels=["Q1_min", "Q2", "Q3", "Q4_max"])
    pooled = with_gap.groupby("gap_quartile_pooled", observed=True).agg(
        n=("aromatic_fraction", "size"),
        aromatic_fraction_mean=("aromatic_fraction", "mean"),
        gap_max_range=("gap_max", lambda s: f"{s.min():.1f}-{s.max():.1f}"),
    ).reset_index()
    pooled.insert(0, "grouping", "pooled_all_protocols")
    pooled = pooled.rename(columns={"gap_quartile_pooled": "gap_quartile"})

    # --- 2. quartili PER PROTOCOLLO (0.7 ha mostrato che il pooled confonde i protocolli) ---
    per_proto_rows = []
    for proto, sub in with_gap.groupby("protocol"):
        sub = sub.copy()
        sub["gap_quartile_within"] = pd.qcut(sub["gap_max"], 4, labels=["Q1_min", "Q2", "Q3", "Q4_max"], duplicates="drop")
        agg = sub.groupby("gap_quartile_within", observed=True).agg(
            n=("aromatic_fraction", "size"),
            aromatic_fraction_mean=("aromatic_fraction", "mean"),
            gap_max_range=("gap_max", lambda s: f"{s.min():.1f}-{s.max():.1f}"),
        ).reset_index()
        agg.insert(0, "grouping", f"within_protocol[{proto}]")
        agg = agg.rename(columns={"gap_quartile_within": "gap_quartile"})
        per_proto_rows.append(agg)
    per_protocol = pd.concat(per_proto_rows, ignore_index=True)

    # --- 3. riferimento: controlli senza energia e senza-stadio-soft, nessun quartile (nessun gap) ---
    ref_rows = []
    for proto, sub in summary[~summary["protocol"].isin(with_gap["protocol"].unique())].groupby("protocol"):
        ref_rows.append({"grouping": "reference_no_gap_trace", "gap_quartile": proto,
                          "n": len(sub), "aromatic_fraction_mean": sub["aromatic_fraction"].mean(),
                          "gap_max_range": "n/d (nessuna traccia soft o w_E=0)"})
    reference = pd.DataFrame(ref_rows)

    result = pd.concat([pooled, per_protocol, reference], ignore_index=True)
    result.to_csv(os.path.join(OUT_DIR, "0_8c_aromatic_by_gap_quartile.csv"), index=False)

    print("Frazione aromatica media della sequenza finale, per quartile di gap_max (pooled):")
    print(pooled.to_string(index=False))
    print("\nStesso, entro ciascun protocollo (evita il confondimento fra protocolli visto in 0.7):")
    print(per_protocol.to_string(index=False))
    print("\nRiferimento — protocolli senza traccia di gap (nessun confronto per quartile possibile):")
    print(reference.to_string(index=False))

    q1 = pooled.loc[pooled["gap_quartile"] == "Q1_min", "aromatic_fraction_mean"].iloc[0]
    q4 = pooled.loc[pooled["gap_quartile"] == "Q4_max", "aromatic_fraction_mean"].iloc[0]
    print(f"\nConfronto diretto (pooled): Q1(gap minimo)={q1:.3f}  vs  Q4(gap massimo)={q4:.3f}  "
          f"({'Q4 > Q1' if q4 > q1 else 'Q4 <= Q1'})")

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
