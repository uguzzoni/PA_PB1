"""
run_0_4_gradient_direction.py

Fase 0.4 del piano di design (Workplan_phase0_instantiated.md §0.4): direzione
del gradiente ∇_x E(x) valutato nei vertici one-hot del training set — quali
amminoacidi, in media, il gradiente spinge il design a favorire (E minore) o
evitare (E maggiore), e se questa direzione varia per posizione.

Convenzione di segno: il gradiente qui è quello di E rispetto a x (aumentare
la probabilità di un AA in una posizione). Il design minimizza
loss_total = loss_af + energy_weight * E (vedi CLAUDE.md), quindi un
amminoacido con gradiente NEGATIVO è quello verso cui il gradient descent
spinge (riduce E); un gradiente POSITIVO è penalizzato. aa_ranking.csv è
ordinato per gradiente crescente, cioè dal più "favorito" al più "evitato".

Il set BLI (0.3', 23 sequenze validate) è usato come secondo controllo
indipendente solo per il ranking medio (23*15=345 osservazioni AA, ai limiti
della robustezza ma indicativo) — non per la mappa 15x20 non mediata, troppo
rumorosa con questo N (nota esplicita nel piano).

Usage:
    uv run python src/phase0_diagnostics/run_0_4_gradient_direction.py
"""

import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import (AAS_JULIA, BINDER_LEN, REPO_ROOT, encode_sequence, load_bli_population,
                           load_training_sequences_onehot)

from colabdesign.energy_model.model_3layer import load_energy_model, mlp_forward

WEIGHTS_PATH = os.path.join(
    REPO_ROOT, "data", "energy_model_params", "PNB_2R_3lay_negbinom_energy_model_weights.json"
)
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_4_gradient_direction")
STANDARD_AA = list(AAS_JULIA[:20])

# Stesse sequenze/energie di validazione di 0.1 — condividono pesi ed encoder.
VALIDATION_SEQS = {
    "MDFNPWLLFLKVPAQ": -1.0169219970703125,
    "VDFNPWLLFLKVPAQ": -1.1298952102661133,
    "IDFNPYLLFLKVPAQ": 0.9208765625953674,
    "MDWNPLLLHLKRPAQ": 3.178508758544922,
    "VDYNPWLLFLRKPKQ": 1.1072289943695068,
    "IDYNPYLLFLKQPKQ": -1.1511486768722534,
}
VALIDATION_TOL = 1e-4


def validate_encoder(params) -> None:
    print("Validazione encoder contro le energie già calcolate da ColabDesign...")
    max_err = 0.0
    for seq, expected in VALIDATION_SEQS.items():
        x = encode_sequence(seq, allow_gap=False)[None]
        got = float(mlp_forward(params, jnp.asarray(x)))
        max_err = max(max_err, abs(got - expected))
    if max_err > VALIDATION_TOL:
        raise RuntimeError(f"Validazione FALLITA: |err| massimo={max_err:.2e} > {VALIDATION_TOL:.0e}")
    print(f"OK — |err| massimo = {max_err:.2e}\n")


def make_grad_batch(params):
    single = lambda xi: mlp_forward(params, xi[None])  # xi: (15,21) -> scalare
    grad_single = jax.grad(single)  # -> (15,21), stessa shape di xi
    return jax.jit(jax.vmap(grad_single))


def summarize_gradients(grads: np.ndarray):
    """grads: (N, 15, 21) -> ranking 20 AA (media su seq e posizioni) e
    mappa 15x20 (media solo su seq). Colonna gap (21esima) esclusa: sempre
    zero in produzione (vedi nota 0.1), il suo gradiente non è informativo
    per la direzione di design."""
    grads20 = grads[..., :20]  # (N, 15, 20)
    ranking = grads20.mean(axis=(0, 1))  # (20,)
    ranking_std = grads20.std(axis=(0, 1))
    position_map = grads20.mean(axis=0)  # (15, 20)
    return ranking, ranking_std, position_map


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    params = load_energy_model(WEIGHTS_PATH)
    validate_encoder(params)
    grad_batch = make_grad_batch(params)

    print("Caricamento training set (4_counts: F2+F3+R2+R3, stesso set di 0.1/0.2)...")
    train_onehot, train_seqs, n_dropped = load_training_sequences_onehot()
    print(f"  {len(train_seqs)} sequenze valide, {n_dropped} scartate\n")

    print(f"Calcolo gradiente ∇_x E in ciascuno dei {len(train_seqs)} vertici one-hot (jax.vmap(jax.grad))...")
    train_grads = np.asarray(grad_batch(jnp.asarray(train_onehot)))
    ranking, ranking_std, position_map = summarize_gradients(train_grads)

    print("Caricamento popolazione BLI (0.3') come secondo controllo indipendente...")
    bli, _, _ = load_bli_population()
    bli_onehot = np.stack([encode_sequence(s, allow_gap=False) for s in bli["Sequence"]])
    bli_grads = np.asarray(grad_batch(jnp.asarray(bli_onehot)))
    bli_ranking, bli_ranking_std, _ = summarize_gradients(bli_grads)
    print(f"  {len(bli)} sequenze — ranking calcolato solo su media seq+posizioni "
          f"(15x20 non mediata troppo rumorosa a questo N, per piano §0.4)\n")

    order = np.argsort(ranking)  # crescente: più "favorito" (E minore) prima
    aa_ranking = pd.DataFrame({
        "amino_acid": [STANDARD_AA[i] for i in order],
        "mean_gradient_training": ranking[order],
        "std_gradient_training": ranking_std[order],
        "mean_gradient_bli": bli_ranking[order],
        "std_gradient_bli": bli_ranking_std[order],
        "rank_training": np.arange(1, 21),
    })
    aa_ranking.to_csv(os.path.join(OUT_DIR, "aa_ranking.csv"), index=False)
    print("aa_ranking.csv (ordinato per gradiente training crescente = più favorito -> più evitato):")
    print(aa_ranking.to_string(index=False))

    spearman_num = np.corrcoef(np.argsort(np.argsort(ranking)), np.argsort(np.argsort(bli_ranking)))[0, 1]
    metrics = {
        "n_train_sequences": len(train_seqs),
        "n_bli_sequences": len(bli),
        "most_favored_aa_training": STANDARD_AA[int(np.argmin(ranking))],
        "most_disfavored_aa_training": STANDARD_AA[int(np.argmax(ranking))],
        "most_favored_aa_bli": STANDARD_AA[int(np.argmin(bli_ranking))],
        "most_disfavored_aa_bli": STANDARD_AA[int(np.argmax(bli_ranking))],
        "spearman_training_vs_bli_ranking": float(spearman_num),
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n" + json.dumps(metrics, indent=2))

    # --- ranking a barre, training vs BLI ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(20)
    ax.bar(x - 0.2, aa_ranking["mean_gradient_training"], width=0.4, label="training set (N={})".format(len(train_seqs)))
    ax.bar(x + 0.2, aa_ranking["mean_gradient_bli"], width=0.4, label="BLI validati (N={})".format(len(bli)))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(aa_ranking["amino_acid"])
    ax.set_ylabel(r"$\overline{\partial E/\partial x}$  (favorito < 0 < evitato)")
    ax.set_title("Direzione media del gradiente di E per amminoacido")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "aa_ranking.png"), dpi=150)
    plt.close(fig)

    # --- mappa 15x20 non mediata sulle posizioni (solo training set) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = np.abs(position_map).max()
    im = ax.imshow(position_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(20))
    ax.set_xticklabels(STANDARD_AA)
    ax.set_yticks(range(BINDER_LEN))
    ax.set_yticklabels(range(1, BINDER_LEN + 1))
    ax.set_xlabel("amminoacido")
    ax.set_ylabel("posizione nel binder")
    ax.set_title(f"Gradiente medio ∂E/∂x per posizione (training set, N={len(train_seqs)})\nblu=favorito (E minore), rosso=evitato (E maggiore)")
    plt.colorbar(im, label=r"$\overline{\partial E/\partial x}$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "position_map.png"), dpi=150)
    plt.close(fig)

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
