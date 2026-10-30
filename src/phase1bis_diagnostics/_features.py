"""
_features.py

Estrazione della mappa di feature phi(x) dell'ultimo strato nascosto e della
leva h(x) = phi(x)^T (Phi^T Phi + lam I)^-1 phi(x) -- condivise da §1b.1, §1b.2,
§1b.3 (Workplan_v2_phase1bis_instantiated.md). Pura algebra numpy sui pesi gia'
caricati con load_energy_model (dense1/dense2) -- nessuna modifica al fork,
nessuna dipendenza da mlp_forward/model_3layer_v2.
"""

import numpy as np


def extract_phi_batch(params, x_batch: np.ndarray) -> np.ndarray:
    """x_batch: (N, 15, 21) Julia-order (encode_sequence) -> phi: (N, 16), bias incluso.

    phi(x) = relu(relu(x_flat @ dense1.W + dense1.b) @ dense2.W + dense2.b), 15
    componenti + colonna di bias costante (Workplan_v2_fasi_1bis_2_3.md §1b.1).
    """
    x_flat = x_batch.reshape(x_batch.shape[0], -1)  # (N, 315)
    h1 = np.maximum(0.0, x_flat @ np.asarray(params["dense1"]["W"]) + np.asarray(params["dense1"]["b"]))
    phi = np.maximum(0.0, h1 @ np.asarray(params["dense2"]["W"]) + np.asarray(params["dense2"]["b"]))  # (N, 15)
    bias_col = np.ones((phi.shape[0], 1), dtype=phi.dtype)
    return np.concatenate([phi, bias_col], axis=1)  # (N, 16)


def compute_leverage(phi_batch: np.ndarray, phi_reference: np.ndarray, lam: float) -> np.ndarray:
    """h(x) per ciascuna riga di phi_batch, con la matrice 16x16 costruita da phi_reference
    (tipicamente Phi sul training set rilevante per l'esemplare in uso)."""
    p = phi_reference.shape[1]
    gram = phi_reference.T @ phi_reference + lam * np.eye(p)
    gram_inv = np.linalg.inv(gram)
    return np.einsum("ij,jk,ik->i", phi_batch, gram_inv, phi_batch)
