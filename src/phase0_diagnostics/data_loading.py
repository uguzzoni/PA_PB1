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
    frames = [
        pd.read_csv(os.path.join(PD_ENERGY_MODEL_DATA, "4_counts", f)) for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.groupby("Sequence", as_index=False)["Count"].sum()


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


def load_round1_counts() -> pd.DataFrame:
    """Round-1 counts (F_1 + R_1), the closest available proxy for the 'initial
    library' composition in this dataset — NOT the naive pre-panning library,
    already post one round of selection. Only available in protein_counts_123/,
    not in 4_counts/.
    """
    files = ["F_1_count_protein.csv", "R_1_count_protein.csv"]
    frames = [
        pd.read_csv(os.path.join(PD_ENERGY_MODEL_DATA, "protein_counts_123", f))
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.groupby("Sequence", as_index=False)["Count"].sum()
