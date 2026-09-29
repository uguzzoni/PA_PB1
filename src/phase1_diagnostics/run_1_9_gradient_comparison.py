"""
run_1_9_gradient_comparison.py

Fase 1.9 del piano di design (Workplan_phase1_addendum.md §1.9): la direzione
aromatica del gradiente trovata in §0.4 (W unico AA con gradiente medio
negativo) sopravvive o scompare quando si esclude dal training il cluster
dominante (34% della massa di read in round 3, PSBinder-positivo, candidato
TUP di §0.6)? Se sopravvive, l'ipotesi "artefatto sperimentale" non e' piu'
disponibile come spiegazione del gradiente aromatico (ricaduta su §0.6/§1.11).

Passi 1-3 del metodo (export, refactor di run_0_4, riesecuzione sul modello
cluster0-escluso) sono gia' fatti altrove:
  - export: PD_energy_model/.../export_replica_weights.jl (Julia, manuale)
  - refactor + verifica di parita' col default: run_0_4_gradient_direction.py
    (regression-checked il 2026-09-14, risultati identici a prima del refactor)
Questo script fa il passo 4 (il confronto) piu' l'estensione della nota
"Attenzione alla popolazione di valutazione": valuta ANCHE il modello
cluster0-escluso sulle sequenze del cluster 0 (che non ha visto), ma la
riporta separatamente in summary.json, MAI mescolata nella media principale.

Prerequisiti:
  1. export_replica_weights.jl eseguito manualmente (passo A0 passato) ->
     PD_energy_model/.../exported_replicas/modelPNB_3layers_negbinom_ll_excluded_cluster0_weights.json
  2. Questo script invoca run_0_4_gradient_direction.py --weights-path <quel JSON>
     --out-dir results/phase1/1_9_gradient_cluster0/model_cluster0_excluded/
     (stessa popolazione di §0.4: training set completo incluso cluster0 + BLI)
     -- NON e' la stessa cosa del punto 4 sotto, che valuta separatamente solo
     le sequenze di cluster0.

Usage:
    JAX_PLATFORMS=cpu uv run python src/phase1_diagnostics/run_1_9_gradient_comparison.py
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORTED_DIR = os.path.join(REPO_ROOT, "PD_energy_model", "training", "training_2rounds",
                             "workplan_phase1", "exported_replicas")
CLUSTER0_WEIGHTS = os.path.join(EXPORTED_DIR, "modelPNB_3layers_negbinom_ll_excluded_cluster0_weights.json")
COMPLETE_MODEL_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_4_gradient_direction")
CLUSTER0_MODEL_DIR = os.path.join(REPO_ROOT, "results", "phase1", "1_9_gradient_cluster0", "model_cluster0_excluded")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase1", "1_9_gradient_cluster0")
RUN_0_4_SCRIPT = os.path.join(REPO_ROOT, "src", "phase0_diagnostics", "run_0_4_gradient_direction.py")
CLUSTER_ASSIGNMENT_CSV = os.path.join(REPO_ROOT, "results", "phase0", "0_2_clustering", "cluster_assignment.csv")
DOMINANT_CLUSTER_SEQ = "VDYNPWLLFLAQPWQ"  # identita' del cluster0 da §0.6/§0.8.b

sys.path.insert(0, os.path.join(REPO_ROOT, "src", "phase0_diagnostics"))
from data_loading import AAS_JULIA, BINDER_LEN, load_training_sequences_onehot  # noqa: E402

FORK_ROOT = "/home/guido/Projects/protein_design/methods/colabdesign_energy_guidance"
sys.path.insert(0, FORK_ROOT)
from colabdesign.energy_model import model_3layer_v2 as v2  # noqa: E402


def run_0_4_on_cluster0_excluded_model():
    if not os.path.exists(CLUSTER0_WEIGHTS):
        raise FileNotFoundError(
            f"{CLUSTER0_WEIGHTS} mancante -- eseguire prima export_replica_weights.jl (manuale, "
            "PD_energy_model/training/training_2rounds/workplan_phase1/)")
    os.makedirs(CLUSTER0_MODEL_DIR, exist_ok=True)
    print("Riesecuzione di run_0_4_gradient_direction.py sul modello cluster0-escluso "
          "(stessa popolazione di §0.4: training set completo + BLI)...")
    result = subprocess.run(
        [sys.executable, RUN_0_4_SCRIPT, "--weights-path", CLUSTER0_WEIGHTS, "--out-dir", CLUSTER0_MODEL_DIR],
        capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("run_0_4_gradient_direction.py e' fallito sul modello cluster0-escluso")


def load_ranking(model_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(model_dir, "aa_ranking.csv")).set_index("amino_acid")


def evaluate_cluster0_sequences_with_excluded_model():
    """Estensione della nota 'Attenzione alla popolazione di valutazione' (addendum §1.9):
    il modello cluster0-escluso valutato SOLO sulle sequenze del cluster che non ha visto
    -- riportato separatamente, mai mescolato nella media principale sopra."""
    if not os.path.exists(CLUSTER_ASSIGNMENT_CSV):
        print(f"  {CLUSTER_ASSIGNMENT_CSV} non trovato -- salto questa estensione.")
        return None
    cluster_assignment = pd.read_csv(CLUSTER_ASSIGNMENT_CSV)
    if DOMINANT_CLUSTER_SEQ not in set(cluster_assignment["Sequence"]):
        print(f"  sequenza rappresentativa del cluster0 ({DOMINANT_CLUSTER_SEQ}) non trovata in "
              "cluster_assignment.csv -- salto questa estensione.")
        return None
    cluster0_id = int(cluster_assignment.loc[cluster_assignment["Sequence"] == DOMINANT_CLUSTER_SEQ,
                                              "cluster_id"].iloc[0])
    cluster0_seqs_all = set(cluster_assignment.loc[cluster_assignment["cluster_id"] == cluster0_id, "Sequence"])

    train_onehot, train_seqs, _ = load_training_sequences_onehot()
    idx_cluster0 = [i for i, s in enumerate(train_seqs) if s in cluster0_seqs_all]
    if not idx_cluster0:
        print("  nessuna sequenza del training set risulta nel cluster0 -- salto questa estensione.")
        return None
    cluster0_onehot = train_onehot[idx_cluster0]

    params = v2.load_energy_model(CLUSTER0_WEIGHTS)
    single = lambda xi: v2.mlp_forward(params, xi[None])
    grad_batch = jax.jit(jax.vmap(jax.grad(single)))
    grads = np.asarray(grad_batch(jnp.asarray(cluster0_onehot)))[..., :20]
    ranking_cluster0 = grads.mean(axis=(0, 1))
    standard_aa = list(AAS_JULIA[:20])
    most_favored = standard_aa[int(np.argmin(ranking_cluster0))]
    most_disfavored = standard_aa[int(np.argmax(ranking_cluster0))]
    print(f"  N={len(idx_cluster0)} sequenze del cluster0 nel training set completo; "
          f"modello cluster0-escluso valutato SOLO su queste (mai visto in training): "
          f"AA piu' favorito={most_favored}, piu' evitato={most_disfavored}")
    return {
        "n_cluster0_sequences_in_full_training_set": len(idx_cluster0),
        "cluster0_id": cluster0_id,
        "most_favored_aa_on_cluster0_only": most_favored,
        "most_disfavored_aa_on_cluster0_only": most_disfavored,
        "note": ("valutazione del modello cluster0-escluso SOLO sulle sequenze che non ha visto "
                 "in training -- NON mescolata con la media principale sopra (addendum §1.9)"),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(COMPLETE_MODEL_DIR, "aa_ranking.csv")):
        raise FileNotFoundError(
            f"{COMPLETE_MODEL_DIR}/aa_ranking.csv mancante -- rieseguire prima "
            "run_0_4_gradient_direction.py senza argomenti (modello completo, §0.4).")
    run_0_4_on_cluster0_excluded_model()

    complete = load_ranking(COMPLETE_MODEL_DIR)
    excluded = load_ranking(CLUSTER0_MODEL_DIR)
    comparison = complete[["mean_gradient_training", "rank_training"]].join(
        excluded[["mean_gradient_training", "rank_training"]], lsuffix="_complete", rsuffix="_cluster0_excluded")
    comparison = comparison.sort_values("rank_training_complete")
    comparison.to_csv(os.path.join(OUT_DIR, "aa_ranking_comparison.csv"))
    print("\nConfronto ranking (modello completo vs cluster0-escluso):")
    print(comparison.to_string())

    spearman_rankings = np.corrcoef(comparison["rank_training_complete"],
                                     comparison["rank_training_cluster0_excluded"])[0, 1]
    w_negative_complete = comparison.loc["W", "mean_gradient_training_complete"] < 0
    w_negative_excluded = comparison.loc["W", "mean_gradient_training_cluster0_excluded"] < 0
    n_negative_complete = int((comparison["mean_gradient_training_complete"] < 0).sum())
    n_negative_excluded = int((comparison["mean_gradient_training_cluster0_excluded"] < 0).sum())
    # NOTA (scoperta durante l'esecuzione, 2026-09-14): Workplan_phase0_instantiated.md §0.4
    # scrive "W e' l'unico amminoacido con gradiente medio negativo (-0.17)", ma il valore
    # effettivamente calcolato E GIA' COMMESSO in git (commit 1a68083, aa_ranking.csv) e'
    # +0.1699515 -- POSITIVO. Verificato: questo script riproduce esattamente quel valore,
    # non e' un bug introdotto qui. Nessun amminoacido ha gradiente medio negativo in nessuno
    # dei due modelli (n_negative_complete/excluded = 0 in entrambi) -- il criterio letterale
    # "segno negativo" dell'addendum non puo' quindi discriminare nulla, e' un refuso nel testo
    # del piano madre (probabile errore di trascrizione, non un problema di codice o di dati).
    # Si usa percio' "W resta il piu' favorito (rank 1)" + correlazione di Spearman come
    # criterio operativo equivalente, coerente con l'intento sostanziale del piano.
    w_most_favored_complete = bool(comparison["mean_gradient_training_complete"].idxmin() == "W")
    w_most_favored_excluded = bool(comparison["mean_gradient_training_cluster0_excluded"].idxmin() == "W")

    # --- differenza mappa posizionale 15x20 ---
    pm_complete = np.load(os.path.join(COMPLETE_MODEL_DIR, "position_map.npy"))
    pm_excluded = np.load(os.path.join(CLUSTER0_MODEL_DIR, "position_map.npy"))
    pm_diff = pm_excluded - pm_complete
    standard_aa = list(AAS_JULIA[:20])
    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = np.abs(pm_diff).max()
    im = ax.imshow(pm_diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(20))
    ax.set_xticklabels(standard_aa)
    ax.set_yticks(range(BINDER_LEN))
    ax.set_yticklabels(range(1, BINDER_LEN + 1))
    ax.set_xlabel("amminoacido")
    ax.set_ylabel("posizione nel binder")
    ax.set_title("Differenza mappa posizionale: cluster0-escluso − completo\n"
                  "(vicino a zero ovunque = la mappa posizionale non dipende dal cluster0)")
    plt.colorbar(im, label=r"$\Delta \overline{\partial E/\partial x}$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "position_map_diff.png"), dpi=150)
    plt.close(fig)

    print(f"\nSpearman(ranking completo, ranking cluster0-escluso) = {spearman_rankings:.4f}")
    print(f"W ha gradiente medio negativo: completo={w_negative_complete}, cluster0-escluso={w_negative_excluded} "
          f"(NOTA: nessun AA e' negativo in nessuno dei due modelli -- vedi negative_gradient_criterion_note in summary.json)")
    print(f"W e' l'AA piu' favorito (rank 1): completo={w_most_favored_complete}, cluster0-escluso={w_most_favored_excluded}")
    print(f"# AA con gradiente medio negativo: completo={n_negative_complete}, cluster0-escluso={n_negative_excluded}")
    print(f"Differenza mappa posizionale: max|diff|={np.abs(pm_diff).max():.4f}")

    print("\nEstensione: valutazione del modello cluster0-escluso SOLO sulle sequenze del cluster0...")
    cluster0_only_eval = evaluate_cluster0_sequences_with_excluded_model()

    aromatic_direction_survives = bool(w_most_favored_complete and w_most_favored_excluded and spearman_rankings > 0.5)
    summary = {
        "spearman_ranking_complete_vs_cluster0_excluded": float(spearman_rankings),
        "w_negative_gradient_complete": bool(w_negative_complete),
        "w_negative_gradient_cluster0_excluded": bool(w_negative_excluded),
        "n_aa_negative_gradient_complete": n_negative_complete,
        "n_aa_negative_gradient_cluster0_excluded": n_negative_excluded,
        "negative_gradient_criterion_note": ("nessun AA ha gradiente medio negativo in nessuno dei due modelli -- "
                                              "Workplan_phase0_instantiated.md §0.4 riporta '-0.17' per W ma il "
                                              "valore committato in git (1a68083) e' +0.1699515: refuso nel testo "
                                              "del piano madre, non un bug di questo script (verificato). Criterio "
                                              "letterale non discriminante, sostituito da w_most_favored_* sotto."),
        "w_most_favored_complete": w_most_favored_complete,
        "w_most_favored_cluster0_excluded": w_most_favored_excluded,
        "position_map_diff_max_abs": float(np.abs(pm_diff).max()),
        "aromatic_direction_survives": aromatic_direction_survives,
        "reading": ("la direzione aromatica e' una proprieta' della popolazione selezionata, non del "
                    "contaminante -- §0.6 si ridimensiona a limite dichiarato (addendum §1.11.1)"
                    if aromatic_direction_survives else
                    "la direzione aromatica scompare/si attenua col cluster0 escluso -- §0.4/§0.6 "
                    "descrivevano lo stesso fenomeno, il modello cluster0-escluso diventa candidato "
                    "naturale per la Fase 2 (esito da trattare come importante, non come dettaglio)"),
        "cluster0_only_evaluation": cluster0_only_eval,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nLettura: {summary['reading']}")
    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
