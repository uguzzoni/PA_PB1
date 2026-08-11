"""
run_0_5_structural_context.py

Fase 0.5 del piano di design (Workplan_phase0_instantiated.md §0.5): quali
posizioni del binder sono interfaccia vs esposte, e se la composizione
amminoacidica dei design (bias aromatico già visto in 0.1/0.3'/0.4) è
concentrata all'interfaccia o distribuita ovunque.

Due classificazioni posizionali indipendenti, confrontate invece di fidarsi
di una sola (vedi Workplan §0.5 per il contesto):
  1. **2ZNL nativa**: contatto a soglia di distanza + SASA relativa sulla
     struttura reale del complesso PA-PB1 (data/pdbs/2ZNL.pdb), 1 sola
     osservazione (il binder canonico MDVNPTLLFLKVPAQ).
  2. **AF3-consensus**: contact_probs predetti da AF3 su 35 sequenze reali
     "promettenti" in af3_predictions/ (indicizzate da
     src/analysis/results/summary_best_candidates.csv), maggioranza sulle 35.
     Permette di verificare l'assunzione mai testata prima che ColabDesign
     mantenga il registro nativo del binder.

La classificazione finale usata per l'analisi di composizione (step 5) è
AF3-consensus (più diretta, basata su design reali invece che sul solo WT) —
2ZNL resta come cross-check indipendente, riportato ma non usato per il
raggruppamento finale.

Usage:
    uv run python src/phase0_diagnostics/run_0_5_structural_context.py
"""

import glob
import json
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import (AAS_JULIA, BINDER_LEN, REPO_ROOT, WT_SEQ, load_bli_population,
                           load_designed_sequences, load_round1_counts, load_training_counts,
                           try_encode_sequence)

PDB_PATH = os.path.join(REPO_ROOT, "data", "pdbs", "2ZNL.pdb")
AF3_PREDICTIONS_DIR = os.path.join(REPO_ROOT, "af3_predictions")
AF3_MANIFEST_CSV = os.path.join(REPO_ROOT, "src", "analysis", "results", "summary_best_candidates.csv")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_5_structural_context")

STANDARD_AA = list(AAS_JULIA[:20])
CONTACT_DISTANCE_A = 5.0     # soglia 2ZNL, Å heavy-atom (decisione aperta #2 nel piano)
AF3_CONTACT_PROB_THRESHOLD = 0.5  # soglia AF3 contact_probs (decisione aperta #2 nel piano)
SASA_EXPOSED_THRESHOLD = 0.5      # SASA_complesso / SASA_isolata


def classify_2znl():
    """Contatto a soglia di distanza (interfaccia) + SASA relativa (esposta),
    catena B (binder, 1-15) vs catena A (target, 257-716) in 2ZNL.pdb."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("2ZNL", PDB_PATH)
    model = structure[0]
    chain_a, chain_b = model["A"], model["B"]

    a_coords = np.array([a.coord for res in chain_a for a in res])
    min_dist = {}
    for res in chain_b:
        pos = res.id[1]
        b_coords = np.array([a.coord for a in res])
        min_dist[pos] = float(cdist(b_coords, a_coords).min())

    sr = ShrakeRupley()
    sr.compute(model, level="R")
    sasa_complex = {res.id[1]: res.sasa for res in chain_b}

    isolated = parser.get_structure("2ZNL_B_only", PDB_PATH)[0]
    for cid in list(isolated.child_dict):
        if cid != "B":
            isolated.detach_child(cid)
    sr.compute(isolated, level="R")
    sasa_isolated = {res.id[1]: res.sasa for res in isolated["B"]}

    rows = []
    for pos in range(1, BINDER_LEN + 1):
        rel_sasa = sasa_complex[pos] / sasa_isolated[pos] if sasa_isolated[pos] > 0 else 0.0
        rows.append({
            "position": pos, "znl_min_dist_to_target_A": min_dist[pos],
            "znl_interface": min_dist[pos] < CONTACT_DISTANCE_A,
            "znl_sasa_complex": sasa_complex[pos], "znl_sasa_isolated": sasa_isolated[pos],
            "znl_relative_sasa": rel_sasa, "znl_exposed": rel_sasa > SASA_EXPOSED_THRESHOLD,
        })
    return pd.DataFrame(rows)


def _best_model_files(folder_path: str):
    """Sceglie il modello a ranking_score più alto fra i _summary_confidences_N.json
    presenti (glob, non un pattern fisso: il batch 2026-04-08 usa nomi interni
    basati su timestamp, non sulla sequenza — vedi Workplan §0.5)."""
    summary_files = glob.glob(os.path.join(folder_path, "*_summary_confidences_*.json"))
    best_summary, best_file = None, None
    for sf in summary_files:
        with open(sf) as f:
            sc = json.load(f)
        if best_summary is None or sc["ranking_score"] > best_summary["ranking_score"]:
            best_summary, best_file = sc, sf
    full_data_file = best_file.replace("summary_confidences", "full_data")
    return best_summary, full_data_file


def classify_af3_consensus():
    """Interfaccia per posizione dai contact_probs predetti da AF3, maggioranza
    sulle 35 sequenze 'promettenti' in af3_predictions/ (indicizzate da
    summary_best_candidates.csv)."""
    manifest = pd.read_csv(AF3_MANIFEST_CSV)
    folders = manifest.drop_duplicates("af3_folder")[["af3_folder", "protocol"]]

    per_seq_rows = []
    for _, row in folders.iterrows():
        folder_path = os.path.join(AF3_PREDICTIONS_DIR, row["af3_folder"])
        best_summary, full_data_file = _best_model_files(folder_path)

        job_request_file = glob.glob(os.path.join(folder_path, "*_job_request.json"))[0]
        with open(job_request_file) as f:
            jr = json.load(f)
        binder_seq = jr[0]["sequences"][1]["proteinChain"]["sequence"]

        with open(full_data_file) as f:
            fd = json.load(f)
        contact_probs = np.array(fd["contact_probs"])
        chain_ids = np.array(fd["token_chain_ids"])
        binder_chain_id = min(Counter(chain_ids), key=lambda k: Counter(chain_ids)[k])
        b_idx = np.where(chain_ids == binder_chain_id)[0]
        a_idx = np.where(chain_ids != binder_chain_id)[0]
        max_contact = contact_probs[np.ix_(b_idx, a_idx)].max(axis=1)  # (15,)

        per_seq_rows.append({
            "af3_folder": row["af3_folder"], "seq": binder_seq, "protocol": row["protocol"],
            "ranking_score": best_summary["ranking_score"], "has_clash": best_summary["has_clash"],
            **{f"pos_{i+1}_max_contact": float(max_contact[i]) for i in range(BINDER_LEN)},
        })
    per_seq = pd.DataFrame(per_seq_rows)

    contact_cols = [f"pos_{i+1}_max_contact" for i in range(BINDER_LEN)]
    interface_bool = per_seq[contact_cols] > AF3_CONTACT_PROB_THRESHOLD
    consensus = pd.DataFrame({
        "position": range(1, BINDER_LEN + 1),
        "af3_frac_interface": interface_bool.mean(axis=0).to_numpy(),
        "af3_mean_max_contact": per_seq[contact_cols].mean(axis=0).to_numpy(),
    })
    consensus["af3_interface"] = consensus["af3_frac_interface"] > 0.5
    return consensus, per_seq


def aa_composition(seqs, weights=None) -> pd.Series:
    if weights is None:
        weights = np.ones(len(seqs))
    counts = pd.Series(0.0, index=STANDARD_AA)
    for seq, w in zip(seqs, weights):
        for c in seq:
            counts[c] += w
    return counts / counts.sum()


def aromatic_fraction(seqs, weights=None) -> float:
    freq = aa_composition(seqs, weights)
    return float(freq[["F", "W", "Y"]].sum())


def aromatic_by_class(seqs, position_class: dict, weights=None):
    """position_class: {1..15: 'interface'|'exposed'}. Ritorna {classe: frazione aromatica}
    su tutti i residui delle sequenze che cadono in quella classe."""
    if weights is None:
        weights = np.ones(len(seqs))
    letters = {"interface": [], "exposed": []}
    w_out = {"interface": [], "exposed": []}
    for seq, w in zip(seqs, weights):
        for pos_idx, c in enumerate(seq, start=1):
            cls = position_class.get(pos_idx)
            if cls is not None:
                letters[cls].append(c)
                w_out[cls].append(w)
    return {cls: aromatic_fraction(letters[cls], w_out[cls]) if letters[cls] else float("nan")
            for cls in ("interface", "exposed")}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Parsing 2ZNL.pdb (soglia contatto {CONTACT_DISTANCE_A}Å, soglia SASA relativa {SASA_EXPOSED_THRESHOLD})...")
    znl = classify_2znl()
    print(znl[["position", "znl_min_dist_to_target_A", "znl_interface", "znl_relative_sasa", "znl_exposed"]]
          .to_string(index=False))

    print(f"\nParsing af3_predictions/ (35 sequenze via {os.path.relpath(AF3_MANIFEST_CSV, REPO_ROOT)}, "
          f"soglia contact_prob {AF3_CONTACT_PROB_THRESHOLD})...")
    af3_consensus, af3_per_seq = classify_af3_consensus()
    print(f"  {len(af3_per_seq)} sequenze uniche, has_clash>0 su best model: {int((af3_per_seq['has_clash']>0).sum())}")
    print(af3_consensus.to_string(index=False))

    position_classification = znl.merge(af3_consensus, on="position")

    # znl_interface (contatto <5A) risulta VERO su tutte e 15 le posizioni (min_dist
    # osservato 2.58-3.75A ovunque, ben sotto qualsiasi soglia ragionevole per un
    # peptide di 15 residui incassato in un solco esteso) — è un risultato reale, non
    # un artefatto di soglia, ma rende il contatto binario non informativo per il
    # confronto con AF3. L'asse che discrimina è la sepoltura (SASA relativa): si
    # confronta quindi "sepolto" (non znl_exposed) con "interfaccia" AF3, non il
    # contatto binario 2ZNL (sempre vero, il confronto sarebbe degenere per
    # costruzione — vedi print sotto).
    position_classification["znl_buried"] = ~position_classification["znl_exposed"]
    position_classification["agree"] = (
        position_classification["znl_buried"] == position_classification["af3_interface"]
    )
    n_agree = int(position_classification["agree"].sum())
    n_znl_interface = int(position_classification["znl_interface"].sum())
    print(f"\nContatto 2ZNL a {CONTACT_DISTANCE_A}A: interfaccia su {n_znl_interface}/{BINDER_LEN} posizioni "
          f"({'degenere, non informativo — vedi min_dist sopra, tutte <{}A'.format(CONTACT_DISTANCE_A) if n_znl_interface == BINDER_LEN else 'informativo'})")
    print(f"Accordo sepoltura 2ZNL (SASA) vs interfaccia AF3-consensus: {n_agree}/{BINDER_LEN} posizioni")
    position_classification.to_csv(os.path.join(OUT_DIR, "position_classification.csv"), index=False)

    agreement_summary = {
        "n_positions_agree_burial_vs_af3_interface": n_agree, "n_positions_total": BINDER_LEN,
        "n_positions_znl_contact_interface": n_znl_interface,
        "znl_contact_criterion_degenerate": n_znl_interface == BINDER_LEN,
        "contact_distance_threshold_A": CONTACT_DISTANCE_A,
        "af3_contact_prob_threshold": AF3_CONTACT_PROB_THRESHOLD,
        "sasa_exposed_threshold": SASA_EXPOSED_THRESHOLD,
        "n_af3_sequences": len(af3_per_seq),
        "n_af3_sequences_with_clash_on_best_model": int((af3_per_seq["has_clash"] > 0).sum()),
        "per_position_agreement": position_classification[
            ["position", "znl_interface", "znl_buried", "af3_interface", "af3_frac_interface", "agree"]
        ].to_dict("records"),
    }
    with open(os.path.join(OUT_DIR, "af3_vs_2znl_agreement.csv"), "w") as f:
        position_classification[["position", "znl_interface", "znl_relative_sasa", "znl_buried", "af3_interface",
                                  "af3_frac_interface", "af3_mean_max_contact", "agree"]].to_csv(f, index=False)
    with open(os.path.join(OUT_DIR, "af3_vs_2znl_agreement_summary.json"), "w") as f:
        json.dump(agreement_summary, f, indent=2)

    # --- classificazione finale usata per l'analisi di composizione: AF3-consensus ---
    position_class = {
        int(r.position): ("interface" if r.af3_interface else "exposed")
        for r in position_classification.itertuples()
    }
    print(f"\nClassificazione finale (AF3-consensus): {position_class}")

    print("\nCaricamento popolazioni per il confronto (WT, libreria round1, training set, BLI, design)...")
    designs = load_designed_sequences()
    lib_raw = load_round1_counts()
    lib_valid = lib_raw[lib_raw["Sequence"].apply(lambda s: try_encode_sequence(s, allow_gap=False) is not None)]
    train_raw = load_training_counts()
    train_valid = train_raw[train_raw["Sequence"].apply(lambda s: try_encode_sequence(s, allow_gap=False) is not None)]
    bli, _, _ = load_bli_population()

    populations = {
        "WT": ([WT_SEQ], None),
        "library_round1": (lib_valid["Sequence"].tolist(), lib_valid["Count"].to_numpy()),
        "training_set": (train_valid["Sequence"].tolist(), train_valid["Count"].to_numpy()),
        "BLI_validated": (bli["Sequence"].tolist(), None),
        "designed_all_campaigns": (designs["Sequence"].tolist(), None),
    }

    rows = []
    for pop_name, (seqs, weights) in populations.items():
        by_class = aromatic_by_class(seqs, position_class, weights)
        rows.append({
            "population": pop_name, "n_sequences": len(seqs),
            "aromatic_fraction_interface": by_class["interface"],
            "aromatic_fraction_exposed": by_class["exposed"],
            "aromatic_fraction_overall": aromatic_fraction(seqs, weights),
        })
    aromatic_table = pd.DataFrame(rows)
    aromatic_table.to_csv(os.path.join(OUT_DIR, "aromatic_fraction_by_class.csv"), index=False)
    print("\n" + aromatic_table.to_string(index=False))

    # --- plot riassuntivo ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    x = np.arange(1, BINDER_LEN + 1)
    ax.bar(x - 0.2, position_classification["znl_relative_sasa"], width=0.4, label="2ZNL: SASA relativa")
    ax.bar(x + 0.2, position_classification["af3_frac_interface"], width=0.4, label="AF3: frazione interfaccia (35 seq)")
    ax.axhline(0.5, color="black", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xlabel("posizione nel binder")
    ax.set_ylabel("SASA relativa 2ZNL  /  frazione interfaccia AF3")
    ax.set_title("Classificazione posizionale: 2ZNL vs AF3-consensus")
    ax.legend(fontsize=7)

    ax = axes[1]
    pop_order = ["WT", "library_round1", "training_set", "BLI_validated", "designed_all_campaigns"]
    plot_df = aromatic_table.set_index("population").loc[pop_order]
    xx = np.arange(len(pop_order))
    ax.bar(xx - 0.2, plot_df["aromatic_fraction_interface"], width=0.4, color="firebrick", label="interfaccia (AF3)")
    ax.bar(xx + 0.2, plot_df["aromatic_fraction_exposed"], width=0.4, color="steelblue", label="esposta (AF3)")
    ax.set_xticks(xx)
    ax.set_xticklabels(pop_order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("frazione aromatica (F+W+Y)")
    ax.set_title("Bias aromatico: interfaccia vs esposta, per popolazione")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary.png"), dpi=150)
    plt.close(fig)

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
