"""
run_0_8b_hotspot_proximity.py

Estensione di 0.8.b (Workplan_phase0_instantiated.md §0.8.b), richiesta
dall'utente: `model_energy_guidance.py` (src/generative_protocols/, e
identico in src/config.py) fissa 28 residui di PA come `hotspot` passati a
`model.prep_inputs(...)` — sono le posizioni del target che ColabDesign è
esplicitamente istruito a contattare durante il design. Qui si verifica se
il bias aromatico dei binder si concentra sulle posizioni del peptide più
vicine a QUESTO sottoinsieme curato di residui (non a "chain A" per intero,
come in 0.5/0.8.b).

Usage:
    uv run python src/phase0_diagnostics/run_0_8b_hotspot_proximity.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import (BINDER_LEN, REPO_ROOT, WT_SEQ, load_bli_population, load_designed_sequences,
                           load_round1_counts, load_training_counts, try_encode_sequence)

# Identico a HOTSPOT_RESIDUES in src/config.py e
# src/generative_protocols/model_energy_guidance.py (non importato da lì per
# evitare la validazione delle env var di config.py, non necessaria qui).
HOTSPOT_RESIDUES = [408, 411, 412, 415, 594, 595, 599,
                    617, 618, 619, 620, 621, 622, 623,
                    629, 631, 639, 640, 643, 666, 667,
                    670, 673, 706, 709, 710, 713, 714]

PDB_PATH = os.path.join(REPO_ROOT, "data", "pdbs", "2ZNL.pdb")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_8_closing_checks")
CONTACT_DISTANCE_A = 4.0   # soglia conservativa (più stretta dei 5A usati in 0.5 per il contatto generico)
N_HOTSPOT_NEAR_THRESHOLD = 4  # >= 4 hotspot residues in contatto -> "near_hotspot" (vedi giustificazione sotto)


def hotspot_contact_profile():
    """Per ciascuna delle 15 posizioni di catena B: distanza minima al residuo
    hotspot più vicino, e numero di residui hotspot distinti (su 28) con
    almeno un atomo entro CONTACT_DISTANCE_A."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("2ZNL", PDB_PATH)
    model = structure[0]
    chain_a, chain_b = model["A"], model["B"]

    hotspot_set = set(HOTSPOT_RESIDUES)
    hotspot_res_coords = {res.id[1]: np.array([a.coord for a in res])
                           for res in chain_a if res.id[1] in hotspot_set}
    assert len(hotspot_res_coords) == len(HOTSPOT_RESIDUES), "hotspot mancanti in 2ZNL.pdb"

    rows = []
    for res in chain_b:
        pos = res.id[1]
        b_coords = np.array([a.coord for a in res])
        dists = {r: float(cdist(b_coords, coords).min()) for r, coords in hotspot_res_coords.items()}
        n_close = sum(d < CONTACT_DISTANCE_A for d in dists.values())
        closest_res = min(dists, key=dists.get)
        rows.append({"position": pos, "min_dist_to_hotspot": dists[closest_res],
                     "closest_hotspot_residue": closest_res, "n_hotspot_residues_in_contact": n_close})
    return pd.DataFrame(rows)


def aromatic_fraction(seqs, weights=None) -> float:
    if len(seqs) == 0:
        return float("nan")
    if weights is None:
        weights = np.ones(len(seqs))
    n_arom = sum(w for seq, w in zip(seqs, weights) for c in seq if c in "FWY")
    n_tot = sum(w * len(seq) for seq, w in zip(seqs, weights))
    return n_arom / n_tot


def aromatic_by_class(seqs, position_class: dict, weights=None) -> dict:
    if weights is None:
        weights = np.ones(len(seqs))
    letters = {"near_hotspot": [], "far_from_hotspot": []}
    w_out = {"near_hotspot": [], "far_from_hotspot": []}
    for seq, w in zip(seqs, weights):
        for pos_idx, c in enumerate(seq, start=1):
            cls = position_class.get(pos_idx)
            if cls is not None:
                letters[cls].append(c)
                w_out[cls].append(w)
    return {cls: aromatic_fraction(letters[cls], w_out[cls]) for cls in letters}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Profilo di contatto con i {len(HOTSPOT_RESIDUES)} residui hotspot (soglia {CONTACT_DISTANCE_A}A)...")
    profile = hotspot_contact_profile()
    print(profile.to_string(index=False))
    profile.to_csv(os.path.join(OUT_DIR, "0_8b_hotspot_contact_profile.csv"), index=False)

    degenerate = bool((profile["min_dist_to_hotspot"] < CONTACT_DISTANCE_A).all())
    print(f"\nDistanza minima al hotspot più vicino: tutte le 15 posizioni <{CONTACT_DISTANCE_A}A "
          f"({'SI' if degenerate else 'NO'}) — come in 0.5, il criterio di sola distanza minima è "
          f"degenere anche ristretto ai soli 28 hotspot: il binder WT è incassato per l'intera "
          f"lunghezza, ogni posizione tocca ALMENO un hotspot. Si usa quindi il NUMERO di hotspot "
          f"distinti in contatto (colonna n_hotspot_residues_in_contact) come metrica graduata, "
          f"non il semplice min_dist.")

    position_class = {
        int(r.position): ("near_hotspot" if r.n_hotspot_residues_in_contact >= N_HOTSPOT_NEAR_THRESHOLD
                           else "far_from_hotspot")
        for r in profile.itertuples()
    }
    near = sorted(p for p, c in position_class.items() if c == "near_hotspot")
    far = sorted(p for p, c in position_class.items() if c == "far_from_hotspot")
    print(f"\nClassificazione (soglia >= {N_HOTSPOT_NEAR_THRESHOLD} hotspot distinti entro {CONTACT_DISTANCE_A}A):")
    print(f"  near_hotspot  ({len(near)} pos.): {near}")
    print(f"  far_from_hotspot ({len(far)} pos.): {far}")

    print("\nCaricamento popolazioni...")
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
    # strati già usati in 0.8.b, per confronto diretto con la classificazione AF3-consensus
    for has_e, sub in designs.groupby("has_energy"):
        populations[f"designed_has_energy_{has_e}"] = (sub["Sequence"].tolist(), None)
    designs_valid_iptm = designs.dropna(subset=["i_ptm"])
    success = designs_valid_iptm["i_ptm"] >= 0.5
    populations["designed_success_iptm_ge_0.5"] = (designs_valid_iptm[success]["Sequence"].tolist(), None)
    populations["designed_failure_iptm_lt_0.5"] = (designs_valid_iptm[~success]["Sequence"].tolist(), None)

    rows = []
    for pop_name, (seqs, weights) in populations.items():
        by_class = aromatic_by_class(seqs, position_class, weights)
        ratio = (by_class["near_hotspot"] / by_class["far_from_hotspot"]
                 if by_class["far_from_hotspot"] else float("nan"))
        rows.append({"population": pop_name, "n_sequences": len(seqs),
                     "aromatic_fraction_near_hotspot": by_class["near_hotspot"],
                     "aromatic_fraction_far_from_hotspot": by_class["far_from_hotspot"],
                     "ratio": ratio})
    result_table = pd.DataFrame(rows)
    result_table.to_csv(os.path.join(OUT_DIR, "0_8b_aromatic_by_hotspot_proximity.csv"), index=False)
    print("\n" + result_table.to_string(index=False))

    # --- plot: profilo di contatto (giustifica la soglia) + frazione aromatica per popolazione ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    colors = ["firebrick" if position_class[p] == "near_hotspot" else "steelblue" for p in range(1, BINDER_LEN + 1)]
    ax.bar(range(1, BINDER_LEN + 1), profile.sort_values("position")["n_hotspot_residues_in_contact"], color=colors)
    ax.axhline(N_HOTSPOT_NEAR_THRESHOLD - 0.5, color="black", ls=":", lw=1)
    ax.set_xlabel("posizione nel binder")
    ax.set_ylabel(f"# hotspot distinti entro {CONTACT_DISTANCE_A}A")
    ax.set_title("Profilo di contatto con i 28 hotspot (rosso=near, blu=far)")

    ax = axes[1]
    plot_pops = ["WT", "library_round1", "training_set", "BLI_validated", "designed_all_campaigns"]
    plot_df = result_table.set_index("population").loc[plot_pops]
    xx = np.arange(len(plot_pops))
    ax.bar(xx - 0.2, plot_df["aromatic_fraction_near_hotspot"], width=0.4, color="firebrick", label="near_hotspot")
    ax.bar(xx + 0.2, plot_df["aromatic_fraction_far_from_hotspot"], width=0.4, color="steelblue", label="far_from_hotspot")
    ax.set_xticks(xx)
    ax.set_xticklabels(plot_pops, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("frazione aromatica (F+W+Y)")
    ax.set_title("Bias aromatico: vicino vs lontano dagli hotspot")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "0_8b_hotspot_proximity_summary.png"), dpi=150)
    plt.close(fig)

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
