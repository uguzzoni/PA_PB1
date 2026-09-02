"""
run_0_8b_design_stratification.py

Fase 0.8.b del piano di design (Workplan_phase0_instantiated.md §0.8.b): i
metadati esistenti distinguono i protocolli con termine di energia da quelli
senza — la popolazione di design (0.5, 900 sequenze) è quindi un esperimento
controllato già disponibile. Se il bias aromatico cresce con l'energia (o col
suo peso), la causa B/C (dati sperimentali / rilassamento continuo) è
dimostrata anziché solo inferita; se non varia, la causa A (obiettivo AF2)
torna in primo piano.

Riusa `classify_af3_consensus()` da run_0_5_structural_context.py (ora con
`threshold` esposto) invece di riscrivere la classificazione posizionale —
qui aggiunge solo l'analisi di sensitività alla soglia (0.5 la teneva fissa).

Usage:
    uv run python src/phase0_diagnostics/run_0_8b_design_stratification.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import AAS_JULIA, REPO_ROOT, load_designed_sequences
from run_0_5_structural_context import classify_af3_consensus

OUT_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_8_closing_checks")
STANDARD_AA = list(AAS_JULIA[:20])
IPTM_SUCCESS_THRESHOLD = 0.5  # soglia di accettabilità dichiarata nel piano madre §B
SENSITIVITY_THRESHOLDS = [0.3, 0.5, 0.7]


def aromatic_fraction(seqs) -> float:
    if len(seqs) == 0:
        return float("nan")
    letters = "".join(seqs)
    return sum(c in "FWY" for c in letters) / len(letters)


def aromatic_by_class(seqs, position_class: dict) -> dict:
    letters = {"interface": [], "exposed": []}
    for seq in seqs:
        for pos_idx, c in enumerate(seq, start=1):
            cls = position_class.get(pos_idx)
            if cls is not None:
                letters[cls].append(c)
    return {
        cls: (sum(c in "FWY" for c in chars) / len(chars) if chars else float("nan"))
        for cls, chars in letters.items()
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Caricamento popolazione disegnata (0.5, ora con energy_weight_stage3/has_energy)...")
    designs = load_designed_sequences()
    print(f"  {len(designs)} design, {designs['protocol_file'].nunique()} file sorgente")
    print(f"  con energia: {int(designs['has_energy'].sum())}  |  senza energia: {int((~designs['has_energy']).sum())}\n")

    # --- 1. frazione aromatica aggregata per strato ---
    rows = []
    for has_e, sub in designs.groupby("has_energy"):
        rows.append({"stratum_type": "has_energy", "stratum_value": str(has_e),
                     "n": len(sub), "aromatic_fraction": aromatic_fraction(sub["Sequence"])})
    for we, sub in designs.groupby("energy_weight_stage3"):
        rows.append({"stratum_type": "energy_weight_stage3", "stratum_value": str(we),
                     "n": len(sub), "aromatic_fraction": aromatic_fraction(sub["Sequence"])})
    designs_valid_iptm = designs.dropna(subset=["i_ptm"])
    success = designs_valid_iptm["i_ptm"] >= IPTM_SUCCESS_THRESHOLD
    for label, sub in [("success (i_ptm>=0.5)", designs_valid_iptm[success]),
                        ("failure (i_ptm<0.5)", designs_valid_iptm[~success])]:
        rows.append({"stratum_type": "iptm_outcome", "stratum_value": label,
                     "n": len(sub), "aromatic_fraction": aromatic_fraction(sub["Sequence"])})
    stratum_table = pd.DataFrame(rows)
    stratum_table.to_csv(os.path.join(OUT_DIR, "0_8b_aromatic_by_stratum.csv"), index=False)
    print("Frazione aromatica aggregata per strato:")
    print(stratum_table.to_string(index=False))

    # --- 2. decomposizione interfaccia/esposta per strato (soglia 0.5, come 0.5) ---
    print("\nRicalcolo classificazione AF3-consensus (soglia 0.5, come in 0.5)...")
    consensus_05, _ = classify_af3_consensus(threshold=0.5)
    position_class = {int(r.position): ("interface" if r.af3_interface else "exposed")
                       for r in consensus_05.itertuples()}

    rows = []
    for has_e, sub in designs.groupby("has_energy"):
        by_class = aromatic_by_class(sub["Sequence"], position_class)
        rows.append({"stratum_type": "has_energy", "stratum_value": str(has_e), "n": len(sub),
                     "aromatic_fraction_interface": by_class["interface"],
                     "aromatic_fraction_exposed": by_class["exposed"]})
    for label, sub in [("success (i_ptm>=0.5)", designs_valid_iptm[success]),
                        ("failure (i_ptm<0.5)", designs_valid_iptm[~success])]:
        by_class = aromatic_by_class(sub["Sequence"], position_class)
        rows.append({"stratum_type": "iptm_outcome", "stratum_value": label, "n": len(sub),
                     "aromatic_fraction_interface": by_class["interface"],
                     "aromatic_fraction_exposed": by_class["exposed"]})
    class_stratum_table = pd.DataFrame(rows)
    class_stratum_table.to_csv(os.path.join(OUT_DIR, "0_8b_aromatic_by_class_and_stratum.csv"), index=False)
    print("\nFrazione aromatica per classe posizionale (AF3-consensus, soglia 0.5), per strato:")
    print(class_stratum_table.to_string(index=False))

    # --- 3. analisi di sensitività sulla soglia contact_probs ---
    print("\nAnalisi di sensitività sulla soglia contact_probs (0.3 / 0.5 / 0.7)...")
    sens_rows = []
    for thr in SENSITIVITY_THRESHOLDS:
        consensus, _ = classify_af3_consensus(threshold=thr)
        pos_class_thr = {int(r.position): ("interface" if r.af3_interface else "exposed")
                          for r in consensus.itertuples()}
        n_interface = sum(v == "interface" for v in pos_class_thr.values())
        by_class = aromatic_by_class(designs["Sequence"], pos_class_thr)
        sens_rows.append({
            "threshold": thr, "n_interface_positions": n_interface,
            "interface_positions": ",".join(str(p) for p, c in pos_class_thr.items() if c == "interface"),
            "aromatic_fraction_interface": by_class["interface"],
            "aromatic_fraction_exposed": by_class["exposed"],
            "ratio": by_class["interface"] / by_class["exposed"] if by_class["exposed"] else float("nan"),
        })
    sensitivity_table = pd.DataFrame(sens_rows)
    sensitivity_table.to_csv(os.path.join(OUT_DIR, "0_8b_threshold_sensitivity.csv"), index=False)
    print(sensitivity_table.to_string(index=False))

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
