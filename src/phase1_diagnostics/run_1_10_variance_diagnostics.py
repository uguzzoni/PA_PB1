"""
run_1_10_variance_diagnostics.py

Fase 1.10 del piano di design (Workplan_phase1_addendum.md §1.10): verifica
esplicita che le due varianze del piano madre (§1.3, "Distinzione da
mantenere") siano effettivamente distinte, e che quella che entra in kappa*sigma
(Workplan_peptidi_PA-PB1.md §2.3) sia quella giusta:

  Var_x~a[E(x)]      aleatoria rispetto al rilassamento, decresce durante
                     l'annealing, si annulla allo stadio hard.
  Var_repliche[E(a)] epistemica, cresce allontanandosi dai dati -- questa
                     e' quella che make_energy_fn/make_energy_aux (§1.6)
                     usa in kappa*sigma.

Nota sulle unita': in questo script "sigma_repliche" e' gia' calcolato sui
valori z standardizzati PER REPLICA (§1.6(c), stessa convenzione del bundle),
quindi e' gia' nella stessa scala unitaria in cui l'energia standardizzata
z_ensemble varia di ordine 1 sul training set -- non serve una conversione
aggiuntiva "in unita' di sigma_train" per confrontare le due grandezze, la
domanda del punto 5 dell'addendum si riduce a confrontare sigma_repliche con
1 (l'ordine di grandezza per costruzione di z sul training).

Lavora interamente in ordine Julia (encode_sequence di data_loading.py,
(15,21) con colonna gap sempre zero), NON in ordine AF/ColabDesign -- stesso
approccio di run_0_1/run_0_4, non serve passare da make_energy_fn/aux qui
perche' non c'e' alcuna interazione con ColabDesign in questo script.

Prerequisito: bundle M=10 di §1.7 (build_replica_bundle.py) gia' costruito e
salvato in data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_bundle_M10.npz.

Usage:
    JAX_PLATFORMS=cpu uv run python src/phase1_diagnostics/run_1_10_variance_diagnostics.py
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORK_ROOT = "/home/guido/Projects/protein_design/methods/colabdesign_energy_guidance"
BUNDLE_PATH = os.path.join(REPO_ROOT, "data", "energy_model_params",
                            "PNB_2R_3lay_negbinom_energy_model_bundle_M10.npz")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase1", "1_10_variance_diagnostics")
CLUSTER_ASSIGNMENT_CSV = os.path.join(REPO_ROOT, "results", "phase0", "0_2_clustering", "cluster_assignment.csv")
K_MC = 64
HAMMING_CHUNK = 200  # query alla volta contro l'intero training set (memoria)

sys.path.insert(0, os.path.join(REPO_ROOT, "src", "phase0_diagnostics"))
from data_loading import (BINDER_LEN, encode_sequence, load_bli_population,  # noqa: E402
                           load_training_sequences_onehot)
from run_0_1_simplex_variance import sample_interior_points  # noqa: E402  (stesso seed/griglia di §0.1)

sys.path.insert(0, FORK_ROOT)
from colabdesign.energy_model import model_3layer_v2 as v2  # noqa: E402


def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(
            f"{BUNDLE_PATH} mancante -- eseguire prima build_replica_bundle.py (§1.7 passo B), "
            "che a sua volta richiede export_replica_weights.jl (manuale, Julia).")
    return v2.load_bundle(BUNDLE_PATH)


def make_batched_z_fn(bundle):
    """(N,15,21) Julia-order -> (M,N) z standardizzati per replica (§1.6(c)). Batch su N,
    vmap sulle M repliche del bundle (pytree stacked_params, dim principale M)."""
    W, mus, sds = bundle["stacked_params"], bundle["mu"], bundle["sigma"]

    def mlp_single(params, x):
        return v2.mlp_forward(params, x[None])

    per_replica_batch = jax.vmap(mlp_single, in_axes=(None, 0))       # (N,15,21) -> (N,)
    all_replicas_batch = jax.jit(jax.vmap(per_replica_batch, in_axes=(0, None)))  # -> (M,N)

    def z_fn(x_batch):
        E = np.asarray(all_replicas_batch(W, jnp.asarray(x_batch)))  # (M,N)
        return (E - np.asarray(mus)[:, None]) / np.asarray(sds)[:, None]

    return z_fn


def hamming_distance_to_training(query_idx: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """query_idx: (N,15) indici AA (0-19, gap escluso qui: solo vertici veri).
    train_idx: (N_train,15). Ritorna (N,) distanza di Hamming minima dal training set,
    a chunk per contenere la memoria (N_train tipicamente ~27000)."""
    out = np.empty(len(query_idx), dtype=np.int32)
    for start in range(0, len(query_idx), HAMMING_CHUNK):
        chunk = query_idx[start:start + HAMMING_CHUNK]  # (c,15)
        d = (chunk[:, None, :] != train_idx[None, :, :]).sum(axis=-1)  # (c, N_train)
        out[start:start + len(chunk)] = d.min(axis=1)
    return out


def onehot_to_idx(onehot: np.ndarray) -> np.ndarray:
    """(...,15,21) -> (...,15) indice dell'AA (argmax sulle 20 colonne reali, gap escluso)."""
    return onehot[..., :20].argmax(axis=-1)


def load_cluster_representatives_90pct():
    """19 rappresentanti di cluster che coprono il 90.1% della massa di read in round 3
    (§0.6/§0.8, cluster_read_mass decrescente) -- ricostruito da cluster_assignment.csv,
    non hardcoded."""
    df = pd.read_csv(CLUSTER_ASSIGNMENT_CSV)
    reps = df[df["is_representative"]].sort_values("cluster_read_mass", ascending=False).reset_index(drop=True)
    total_mass = df.groupby("cluster_id")["cluster_read_mass"].first().sum()
    cum = reps["cluster_read_mass"].cumsum() / total_mass
    n_reps = int(np.searchsorted(cum.to_numpy(), 0.901) + 1)
    chosen = reps.iloc[:n_reps]
    print(f"  {len(chosen)} rappresentanti di cluster coprono {cum.iloc[n_reps - 1] * 100:.1f}% della massa totale "
          f"(atteso ~19 / 90.1%, §0.6)")
    return chosen["Sequence"].tolist()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bundle = load_bundle()
    print(f"Bundle caricato: M={bundle['M']}, binder_len={bundle['binder_len']}")
    z_fn = make_batched_z_fn(bundle)

    print("Caricamento training set (per la distanza di Hamming, riferimento)...")
    train_onehot, train_seqs, _ = load_training_sequences_onehot()
    train_idx = onehot_to_idx(train_onehot)

    print("Caricamento popolazione BLI (b, 23 sequenze) e rappresentanti di cluster (c, ~19)...")
    bli, _, _ = load_bli_population()
    bli_onehot = np.stack([encode_sequence(s, allow_gap=False) for s in bli["Sequence"]])
    cluster_reps = load_cluster_representatives_90pct()
    cluster_onehot = np.stack([encode_sequence(s, allow_gap=False) for s in cluster_reps])

    print(f"Campionamento punti Dirichlet (a) -- stesso seed/griglia di §0.1...")
    rng = np.random.default_rng(0)  # SEED di run_0_1_simplex_variance.py
    dirichlet_points, dirichlet_alphas = sample_interior_points(rng)  # (N,15,21), (N,)
    print(f"  {len(dirichlet_points)} punti")

    rows = []

    # --- (b) BLI, (c) cluster reps: solo sigma_repliche + distanza di Hamming ---
    for label, onehot, seqs in [("bli", bli_onehot, bli["Sequence"].tolist()),
                                 ("cluster_representative", cluster_onehot, cluster_reps)]:
        z = z_fn(onehot)  # (M,N)
        sigma_repliche = z.std(axis=0)
        z_mean = z.mean(axis=0)
        dist = hamming_distance_to_training(onehot_to_idx(onehot), train_idx)
        for i, seq in enumerate(seqs):
            rows.append({"population": label, "sequence": seq, "sigma_repliche": float(sigma_repliche[i]),
                         "z_ensemble_mean": float(z_mean[i]), "hamming_dist_to_training": int(dist[i]),
                         "var_x_given_a": np.nan})

    # --- (a) Dirichlet: sigma_repliche + Var_x~a[E(x)] via MC (K=64) + distanza (dal vertice piu' vicino) ---
    print(f"Punti Dirichlet: sigma_repliche + Var_x~a[E(x)] (MC, K={K_MC})...")
    z_soft = z_fn(dirichlet_points)  # (M,N) -- forward diretto sul punto soft (mode 'soft', coerente col bundle)
    sigma_repliche_dirichlet = z_soft.std(axis=0)
    z_mean_dirichlet = z_soft.mean(axis=0)

    vertex_idx = onehot_to_idx(dirichlet_points)
    dist_dirichlet = hamming_distance_to_training(vertex_idx, train_idx)

    rng_mc = np.random.default_rng(1)
    var_x_given_a = np.empty(len(dirichlet_points), dtype=np.float64)
    for i, a in enumerate(dirichlet_points):
        probs20 = a[:, :20]  # (15,20)
        samples = np.zeros((K_MC, BINDER_LEN, 21), dtype=np.float32)
        for pos in range(BINDER_LEN):
            idx = rng_mc.choice(20, size=K_MC, p=probs20[pos])
            samples[np.arange(K_MC), pos, idx] = 1.0
        z_samples = z_fn(samples)  # (M,K_MC)
        e_ensemble_per_sample = z_samples.mean(axis=0)  # (K_MC,) -- E(x) = media ensemble, standardizzata
        var_x_given_a[i] = e_ensemble_per_sample.var()

    for i in range(len(dirichlet_points)):
        rows.append({"population": "dirichlet", "sequence": None, "alpha": float(dirichlet_alphas[i]),
                     "sigma_repliche": float(sigma_repliche_dirichlet[i]),
                     "z_ensemble_mean": float(z_mean_dirichlet[i]),
                     "hamming_dist_to_training": int(dist_dirichlet[i]),
                     "var_x_given_a": float(var_x_given_a[i])})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "variance_by_population.csv"), index=False)

    # --- verifica 1: sigma_repliche cresce allontanandosi dal training set ---
    dirichlet_df = df[df["population"] == "dirichlet"]
    corr_sigma_dist = dirichlet_df[["sigma_repliche", "hamming_dist_to_training"]].corr().iloc[0, 1]
    print(f"\n[verifica 1] corr(sigma_repliche, distanza Hamming) sui punti Dirichlet = {corr_sigma_dist:.3f} "
          f"(atteso: chiaramente positivo)")
    # Diagnostica supplementare: la distanza di Hamming e' calcolata sul VERTICE argmax del
    # punto soft, che per alpha grande (quasi uniforme) e' quasi un tie-break arbitrario fra
    # 20 AA equiprobabili per posizione -- puo' essere un proxy rumoroso della distanza "vera"
    # del punto soft dal training set. log(alpha) e' una misura di distanza diretta e priva di
    # questo rumore (alpha piccolo=vicino ai dati reali/vertici, alpha grande=lontano/baricentro).
    corr_sigma_log_alpha = np.corrcoef(np.log(dirichlet_df["alpha"]), dirichlet_df["sigma_repliche"])[0, 1]
    print(f"             corr(sigma_repliche, log(alpha)) sugli stessi punti = {corr_sigma_log_alpha:.3f} "
          f"(proxy di distanza alternativo, non rumoroso dal tie-break dell'argmax -- "
          f"atteso positivo se sigma_repliche cresce verso il baricentro)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, color in [("bli", "steelblue"), ("cluster_representative", "darkorange"), ("dirichlet", "gray")]:
        sub = df[df["population"] == label]
        ax.scatter(sub["hamming_dist_to_training"], sub["sigma_repliche"], s=10, alpha=0.4, label=label, color=color)
    ax.set_xlabel("distanza di Hamming minima dal training set")
    ax.set_ylabel(r"$\sigma_{\rm repliche}$ (unita' standardizzate per replica)")
    ax.legend(fontsize=8)
    ax.set_title(f"sigma_repliche vs distanza dal training (corr Dirichlet={corr_sigma_dist:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sigma_vs_distance.png"), dpi=150)
    plt.close(fig)

    # --- verifica 2: Var_x~a[E(x)] -> 0 ai vertici, cresce verso il baricentro.
    # ATTENZIONE alla direzione di alpha per Dirichlet(alpha*1_20): alpha PICCOLO produce
    # draw sparsi/concentrati su un singolo angolo (vicino ai VERTICI), alpha GRANDE concentra
    # attorno al punto uniforme (1/20,...,1/20), cioe' il BARICENTRO -- esattamente l'opposto
    # di quel che l'intuizione suggerirebbe leggendo "concentrazione" come "concentrato in un
    # posto piccolo". Verificato: e' cosi' che run_0_1_simplex_variance.py genera i punti
    # (ALPHA_MIN=0.05 -> quasi one-hot, ALPHA_MAX=50 -> quasi uniforme).
    alpha_grid = sorted(dirichlet_df["alpha"].unique())
    var_by_alpha = [dirichlet_df.loc[dirichlet_df["alpha"] == a, "var_x_given_a"].mean() for a in alpha_grid]
    sigma_by_alpha = [dirichlet_df.loc[dirichlet_df["alpha"] == a, "sigma_repliche"].mean() for a in alpha_grid]
    monotone_ok = var_by_alpha[0] < var_by_alpha[-1]  # alpha piccolo (vertice, var bassa) -> alpha grande (baricentro, var alta)
    print(f"[verifica 2] Var_x~a[E(x)] medio ad alpha minimo (vertice)={var_by_alpha[0]:.4f} vs "
          f"alpha massimo (baricentro)={var_by_alpha[-1]:.4f} -- "
          f"{'OK (cresce verso il baricentro)' if monotone_ok else 'ANOMALO -- controllare l implementazione del MC'}")

    # --- verifica 3: le due varianze sono poco correlate fra loro ---
    corr_two_variances = dirichlet_df[["sigma_repliche", "var_x_given_a"]].corr().iloc[0, 1]
    print(f"[verifica 3] corr(sigma_repliche, Var_x~a[E(x)]) sui punti Dirichlet = {corr_two_variances:.3f} "
          f"(atteso: bassa in valore assoluto -- se alta, kappa*sigma starebbe penalizzando "
          f"l'indecisione dell'annealing, non l'ignoranza del modello)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(alpha_grid, var_by_alpha, marker="o", color="firebrick", label=r"Var$_{x\sim a}[E(x)]$ (aleatoria)")
    axes[0].plot(alpha_grid, sigma_by_alpha, marker="o", color="steelblue", label=r"$\sigma_{\rm repliche}$ (epistemica)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("alpha (concentrazione Dirichlet, piccolo=vertice, grande=baricentro)")
    axes[0].set_ylabel("varianza / dispersione")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Le due varianze lungo il rilassamento soft->hard")

    axes[1].scatter(dirichlet_df["sigma_repliche"], dirichlet_df["var_x_given_a"], s=6, alpha=0.3, color="purple")
    axes[1].set_xlabel(r"$\sigma_{\rm repliche}$ (epistemica)")
    axes[1].set_ylabel(r"Var$_{x\sim a}[E(x)]$ (aleatoria)")
    axes[1].set_title(f"corr={corr_two_variances:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "aleatoric_vs_epistemic.png"), dpi=150)
    plt.close(fig)

    summary = {
        "n_dirichlet_points": len(dirichlet_points),
        "n_bli": len(bli_onehot),
        "n_cluster_representatives": len(cluster_reps),
        "corr_sigma_repliche_vs_hamming_distance": float(corr_sigma_dist),
        "corr_sigma_repliche_vs_log_alpha": float(corr_sigma_log_alpha),
        "sigma_grows_with_distance": bool(corr_sigma_dist > 0.2),
        "sigma_grows_with_log_alpha": bool(corr_sigma_log_alpha > 0.2),
        "var_x_given_a_decreases_toward_vertices": bool(monotone_ok),
        "corr_two_variances": float(corr_two_variances),
        "two_variances_weakly_correlated": bool(abs(corr_two_variances) < 0.3),
        "sigma_repliche_scale_note": ("gia' in unita' standardizzate per replica (ordine 1 sul training "
                                       "per costruzione, §1.6(c)) -- nessuna conversione aggiuntiva "
                                       "necessaria per il confronto con la griglia kappa di §2.3"),
        "sigma_repliche_median_bli": float(df.loc[df["population"] == "bli", "sigma_repliche"].median()),
        "sigma_repliche_median_cluster_representative": float(
            df.loc[df["population"] == "cluster_representative", "sigma_repliche"].median()),
        "sigma_repliche_median_dirichlet": float(dirichlet_df["sigma_repliche"].median()),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
