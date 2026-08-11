"""
data_loading.py — shared loaders and sequence encoder for Phase 0 diagnostics.

See ../../Workplan_phase0_instantiated.md for the rationale behind every choice
made here (alphabet, flatten order, which CSVs constitute the training set).
"""

import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PD_ENERGY_MODEL_DATA = os.path.join(REPO_ROOT, "PD_energy_model", "data")

BINDER_LEN = 15

# Training alphabet used by the Julia model (PD_energy_model/analysis/save_model_json.jl,
# AAs2 constant): 20 standard amino acids + a 21st "gap" channel. Stop codons ('*') in the
# raw NGS translations are treated as gap. Order matters: it fixes which column of the
# one-hot each weight row corresponds to.
AAS_JULIA = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_IDX = {c: i for i, c in enumerate(AAS_JULIA)}


def encode_sequence(seq: str, allow_gap: bool = False) -> np.ndarray:
    """One-hot encode a 15-residue sequence, Julia training order, shape (15, 21).

    Stop codons ('*') are mapped to the gap channel. Raises ValueError on wrong
    length or on any character outside the 20 AA (+ gap, if allow_gap) alphabet
    — callers are expected to catch this for noisy NGS data (see
    try_encode_sequence) rather than have it fail silently.
    """
    if len(seq) != BINDER_LEN:
        raise ValueError(f"expected length {BINDER_LEN}, got {len(seq)}: {seq!r}")
    seq = seq.replace("*", "-")
    if not allow_gap and "-" in seq:
        raise ValueError(f"gap not allowed in this context: {seq!r}")
    x = np.zeros((BINDER_LEN, len(AAS_JULIA)), dtype=np.float32)
    for i, c in enumerate(seq):
        idx = AA_TO_IDX.get(c)
        if idx is None:
            raise ValueError(f"unrecognized residue {c!r} in sequence {seq!r}")
        x[i, idx] = 1.0
    return x


def try_encode_sequence(seq: str, allow_gap: bool = True):
    """Like encode_sequence, but returns None instead of raising."""
    try:
        return encode_sequence(seq, allow_gap=allow_gap)
    except ValueError:
        return None


def _merge_counts(dir_path: str, files: list) -> pd.DataFrame:
    frames = [pd.read_csv(os.path.join(dir_path, f)) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df.groupby("Sequence", as_index=False)["Count"].sum()


def load_training_counts() -> pd.DataFrame:
    """Merge F2+F3+R2+R3 protein counts — the training set of the current
    3-layer energy model, per
    PD_energy_model/training/training_2rounds/model_training_7_PNB_negbinom_ll_3layers.ipynb
    (round 1 in protein_counts_123/ is NOT part of this training set).

    Returns one row per unique sequence, Count summed across the 4 files
    (mirrors the Julia notebook's merge_counts(F2,F3) / merge_counts(R2,R3)).
    """
    files = [
        "F_2_count_protein.csv",
        "F_3_count_protein.csv",
        "R_2_count_protein.csv",
        "R_3_count_protein.csv",
    ]
    return _merge_counts(os.path.join(PD_ENERGY_MODEL_DATA, "4_counts"), files)


def load_round2_counts() -> pd.DataFrame:
    """Round-2 counts (F_2 + R_2), used in 0.2 as the comparison round for
    the 'late round' clustering statistics (round 3 is the primary one)."""
    files = ["F_2_count_protein.csv", "R_2_count_protein.csv"]
    return _merge_counts(os.path.join(PD_ENERGY_MODEL_DATA, "4_counts"), files)


def load_round3_counts() -> pd.DataFrame:
    """Round-3 counts (F_3 + R_3) — the latest round available, used in 0.2
    as the primary 'late round' population for clustering / N_eff."""
    files = ["F_3_count_protein.csv", "R_3_count_protein.csv"]
    return _merge_counts(os.path.join(PD_ENERGY_MODEL_DATA, "4_counts"), files)


def load_training_sequences_onehot():
    """Encode the merged training set (load_training_counts) to one-hot (N, 15, 21).

    Sequences with characters outside the 20 AA + stop/gap alphabet (a handful of
    NGS reads with ambiguous calls, e.g. '_') are dropped. Returns
    (onehot: np.ndarray (N,15,21), sequences: list[str] kept, n_dropped: int).
    """
    df = load_training_counts()
    encoded, kept_seqs = [], []
    n_dropped = 0
    for seq in df["Sequence"]:
        x = try_encode_sequence(seq, allow_gap=True)
        if x is None:
            n_dropped += 1
            continue
        encoded.append(x)
        kept_seqs.append(seq)
    return np.stack(encoded), kept_seqs, n_dropped


AFFINITY_CSV = os.path.join(REPO_ROOT, "data", "affinity_measurements_PA-PB1_formatted.csv")
WT_SEQ = "MDVNPTLLFLKVPAQ"  # hardcoded nel piano (Workplan §0.3')


def load_bli_population():
    """Sequenze disegnate con Kd misurato in BLI, dedup per Sequence (non per
    Name: `AFFINITY_CSV` ha 39 righe ma solo 28 sequenze uniche, molte righe
    sono ri-misurazioni della stessa sequenza in round sperimentali diversi
    — vedi Workplan §0.3' per il dettaglio). Kd_nM aggregato come media
    geometrica sulle misure valide ripetute. Righe con Kd_nM=NA e righe "WT"
    escluse dalla popolazione (WT trattato a parte, vedi wt_discrepancy: il
    nome "WT" nel CSV è assegnato a 2 sequenze diverse, verosimile typo).

    Returns: (agg: DataFrame[Sequence, Kd_nM_geomean, n_measurements, Name,
    design_origin], excluded_no_kd: list[str] di Name senza alcun Kd valido,
    wt_discrepancy: dict).
    """
    df = pd.read_csv(AFFINITY_CSV)
    df["Kd_nM"] = pd.to_numeric(df["Kd_nM"], errors="coerce")

    excluded_no_kd = sorted(df.loc[df["Kd_nM"].isna(), "Name"].unique().tolist())
    valid = df.dropna(subset=["Kd_nM"]).copy()

    is_wt = valid["Name"] == "WT"
    wt_rows = valid[is_wt]
    designed = valid[~is_wt]

    agg = designed.groupby("Sequence").agg(
        Kd_nM_geomean=("Kd_nM", lambda x: float(np.exp(np.mean(np.log(x))))),
        n_measurements=("Kd_nM", "size"),
        Name=("Name", "first"),
    ).reset_index()
    agg["design_origin"] = np.where(agg["Name"].str.startswith("MPNN"), "MPNN", "candidate")

    wt_discrepancy = {
        "hardcoded_wt_seq": WT_SEQ,
        "sequences_labeled_WT_in_csv": sorted(wt_rows["Sequence"].unique().tolist()),
        "wt_rows": wt_rows[["Sequence", "Kd_nM"]].to_dict("records"),
        "note": ("2 sequenze diverse condividono il nome 'WT' nel CSV grezzo; "
                 "verosimile typo di trascrizione (pos. 12: L vs V), non due WT distinti. "
                 "Non risolto automaticamente: la composizione WT usa solo la sequenza "
                 "hardcoded dal piano, le altre righe 'WT' sono riportate qui per traccia."),
    }
    return agg, excluded_no_kd, wt_discrepancy


DESIGN_CAMPAIGN_DIRS = ["run1_protocols", "run2_wide", "run_multimer", "run_multimer2"]


def load_designed_sequences() -> pd.DataFrame:
    """Tutte le sequenze disegnate dalle campagne ColabDesign (0.5), unione dei
    JSON in `results/colabdesign/{run1_protocols,run2_wide,run_multimer,run_multimer2}/*.json`.
    Esclude deliberatamente `results/colabdesign/custom/` (le sequenze BLI/validazione
    usate in 0.3'/0.1, non output di una campagna di design).

    Returns: DataFrame[Sequence, protocol_file, energy, i_ptm, ptm, plddt,
    loss_total, loss_af, seed]. Nessun dedup: la stessa sequenza può comparire
    più volte fra protocolli/semi diversi, voluto (si contano i design, non le
    sequenze uniche).
    """
    import json

    rows = []
    for subdir in DESIGN_CAMPAIGN_DIRS:
        dir_path = os.path.join(REPO_ROOT, "results", "colabdesign", subdir)
        if not os.path.isdir(dir_path):
            continue
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(dir_path, fname)) as f:
                data = json.load(f)
            protocol_file = f"{subdir}/{fname[:-5]}"
            for r in data.get("results", []):
                rows.append({
                    "Sequence": r["seq"], "protocol_file": protocol_file,
                    "energy": r.get("energy"), "i_ptm": r.get("i_ptm"), "ptm": r.get("ptm"),
                    "plddt": r.get("plddt"), "loss_total": r.get("loss_total"),
                    "loss_af": r.get("loss_af"), "seed": r.get("seed"),
                })
    return pd.DataFrame(rows)


def load_round1_counts() -> pd.DataFrame:
    """Round-1 counts (F_1 + R_1), the closest available proxy for the 'initial
    library' composition in this dataset — NOT the naive pre-panning library,
    already post one round of selection. Only available in protein_counts_123/,
    not in 4_counts/.
    """
    files = ["F_1_count_protein.csv", "R_1_count_protein.csv"]
    return _merge_counts(os.path.join(PD_ENERGY_MODEL_DATA, "protein_counts_123"), files)
