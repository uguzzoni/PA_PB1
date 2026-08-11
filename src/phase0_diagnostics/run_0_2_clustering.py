"""
run_0_2_clustering.py

Fase 0.2 del piano di design (Workplan_phase0_instantiated.md §0.2): taglia
effettiva del dataset NGS dopo aver raggruppato le sequenze quasi-duplicate
(≥70% identità), per capire quante sequenze "distinte" ci sono davvero dietro
i conteggi grezzi. Round 3 (F_3+R_3) è il round tardo primario; round 2
(F_2+R_2) è calcolato in parallelo solo per confronto.

MMseqs2/CD-HIT non sono installati su questa macchina (verificato di nuovo:
`which mmseqs cd-hit` non trova nulla) — si usa il fallback puro Python
descritto nel piano: clustering greedy stile CD-HIT (non hierarchical
clustering scipy, che per round 2 richiederebbe una matrice di distanza
condensata O(N^2) troppo pesante in memoria — vedi commento in
greedy_cluster_70). Decisione aperta #1 del piano risolta così per questo
run; da rivedere se MMseqs2 diventa disponibile sul cluster AAR.

Usage:
    uv run python src/phase0_diagnostics/run_0_2_clustering.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import REPO_ROOT, load_round2_counts, load_round3_counts

OUT_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_2_clustering")

IDENTITY_THRESHOLD = 0.70
SEQ_LEN = 15
# floor((1 - 0.70) * 15) = 4: fino a 4 mismatch su 15 posizioni dà identità
# reale (15-4)/15 = 73.3% >= 70% (5 mismatch darebbe 66.7% < 70%, escluso).
MAX_MISMATCHES = int(np.floor((1 - IDENTITY_THRESHOLD) * SEQ_LEN))


def greedy_cluster_70(seqs: list, counts: np.ndarray, max_mismatches: int = MAX_MISMATCHES):
    """Clustering greedy stile CD-HIT a soglia di identità fissa (Hamming,
    tutte le sequenze hanno la stessa lunghezza qui, niente indel/allineamento
    necessario).

    Ordina le sequenze per abbondanza (Count) decrescente, cosicché il
    rappresentante di ogni cluster sia la sua sequenza più abbondante — poi
    per ciascuna sequenza, nell'ordine, cerca il rappresentante esistente più
    vicino (Hamming); se a distanza <= max_mismatches la assegna a quel
    cluster, altrimenti apre un nuovo cluster con se stessa come
    rappresentante.

    Evita deliberatamente di materializzare la matrice di distanza N x N
    completa (scipy.cluster.hierarchy la richiederebbe): per round 2
    (N=23.677) una condensed distance matrix in float64 pesa ~2.2GB, al
    limite degli 8GB liberi su questa macchina (vedi `free -h` nel workplan).
    Il confronto qui è vettorizzato ma solo contro i rappresentanti esistenti
    (tipicamente << N), quindi resta rapido: ~3.5s per round 2 su questa
    macchina.

    Returns: cluster_id (N,) int, indice del rappresentante nell'ordine di
    creazione; rep_source_idx (n_clusters,) int, indice in `seqs` del
    rappresentante di ciascun cluster.
    """
    n = len(seqs)
    order = np.argsort(-counts, kind="stable")
    seq_arr = np.array([list(s) for s in seqs])  # (N, 15) array di caratteri
    reps_buffer = np.empty((n, seq_arr.shape[1]), dtype=seq_arr.dtype)
    rep_source_idx = np.empty(n, dtype=np.int64)
    cluster_id = np.full(n, -1, dtype=np.int64)
    n_reps = 0
    for idx in order:
        row = seq_arr[idx]
        if n_reps > 0:
            mismatches = (reps_buffer[:n_reps] != row).sum(axis=1)
            best = int(np.argmin(mismatches))
            if mismatches[best] <= max_mismatches:
                cluster_id[idx] = best
                continue
        reps_buffer[n_reps] = row
        rep_source_idx[n_reps] = idx
        cluster_id[idx] = n_reps
        n_reps += 1
    return cluster_id, rep_source_idx[:n_reps]


def cluster_stats(df: pd.DataFrame, cluster_id: np.ndarray) -> dict:
    """N_eff, distribuzione dimensioni cluster, punti di copertura 50%/90%.

    N_eff = sum_s 1/m_s (piano §0.2), con s che varia sulle sequenze uniche e
    m_s la dimensione (in # sequenze uniche) del cluster a cui s appartiene:
    ogni cluster di dimensione m contribuisce m * (1/m) = 1, quindi la somma
    collassa esattamente al numero di cluster. È la lettura più diretta della
    formula e coincide con l'intuizione: "taglia effettiva" = quante
    sequenze davvero distinte restano dopo aver collassato i quasi-duplicati
    sotto il 70% di identità.
    """
    counts = df["Count"].to_numpy()
    n_clusters = int(cluster_id.max()) + 1
    cluster_size = np.bincount(cluster_id, minlength=n_clusters)  # # seq uniche/cluster
    cluster_mass = np.bincount(cluster_id, weights=counts, minlength=n_clusters)  # # reads/cluster

    n_eff = float(n_clusters)  # vedi docstring: sum_s 1/m_s == n_clusters

    order_desc = np.argsort(-cluster_mass)
    cum_mass_frac = np.cumsum(cluster_mass[order_desc]) / cluster_mass.sum()
    n_for_50 = int(np.searchsorted(cum_mass_frac, 0.50) + 1)
    n_for_90 = int(np.searchsorted(cum_mass_frac, 0.90) + 1)

    return {
        "n_unique_sequences": int(len(df)),
        "n_total_reads": int(counts.sum()),
        "n_clusters": n_clusters,
        "N_eff": n_eff,
        "cluster_size_mean": float(cluster_size.mean()),
        "cluster_size_median": float(np.median(cluster_size)),
        "cluster_size_max": int(cluster_size.max()),
        "n_singleton_clusters": int((cluster_size == 1).sum()),
        "n_clusters_for_50pct_read_mass": n_for_50,
        "n_clusters_for_90pct_read_mass": n_for_90,
    }, cluster_size, cluster_mass


def lorenz_curve(cluster_mass: np.ndarray):
    """Punti (frazione cumulata di cluster, frazione cumulata di massa read),
    cluster ordinati per massa CRESCENTE (convenzione standard curva di
    Lorenz: la diagonale è l'uguaglianza perfetta, la curva sotto la
    diagonale mostra quanto pochi cluster grandi concentrino la massa)."""
    sorted_mass = np.sort(cluster_mass)
    cum_mass = np.cumsum(sorted_mass) / sorted_mass.sum()
    frac_clusters = np.arange(1, len(sorted_mass) + 1) / len(sorted_mass)
    return frac_clusters, cum_mass


def save_assignment(df: pd.DataFrame, cluster_id: np.ndarray, rep_source_idx: np.ndarray,
                     cluster_size: np.ndarray, cluster_mass: np.ndarray, path: str) -> None:
    is_rep = np.zeros(len(df), dtype=bool)
    is_rep[rep_source_idx] = True
    out = pd.DataFrame({
        "Sequence": df["Sequence"].to_numpy(),
        "Count": df["Count"].to_numpy(),
        "cluster_id": cluster_id,
        "cluster_size": cluster_size[cluster_id],
        "cluster_read_mass": cluster_mass[cluster_id].astype(int),
        "is_representative": is_rep,
    })
    out.to_csv(path, index=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Soglia identità {IDENTITY_THRESHOLD:.0%} -> max {MAX_MISMATCHES} mismatch su {SEQ_LEN} posizioni (Hamming).\n")

    results = {}
    cluster_data = {}
    for name, loader in [("round3", load_round3_counts), ("round2", load_round2_counts)]:
        print(f"Caricamento {name}...")
        df = loader()
        print(f"  {len(df)} sequenze uniche, {int(df['Count'].sum())} reads totali")

        cluster_id, rep_source_idx = greedy_cluster_70(df["Sequence"].tolist(), df["Count"].to_numpy())
        stats, cluster_size, cluster_mass = cluster_stats(df, cluster_id)
        results[name] = stats
        cluster_data[name] = (df, cluster_id, rep_source_idx, cluster_size, cluster_mass)
        print(f"  N_eff = {stats['n_clusters']} cluster ({stats['n_singleton_clusters']} singleton)")
        print(f"  copertura: {stats['n_clusters_for_50pct_read_mass']} cluster -> 50% massa, "
              f"{stats['n_clusters_for_90pct_read_mass']} cluster -> 90% massa\n")

    metrics = {
        "identity_threshold": IDENTITY_THRESHOLD,
        "max_mismatches": MAX_MISMATCHES,
        "clustering_method": "greedy CD-HIT-style (fallback Python, MMseqs2/CD-HIT non installati)",
        "primary_late_round": "round3",
        **results,
    }
    with open(os.path.join(OUT_DIR, "n_eff.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # --- cluster_assignment.csv: round3 è l'output primario (prerequisito 2.3/2.5) ---
    df3, cid3, rep3, csize3, cmass3 = cluster_data["round3"]
    save_assignment(df3, cid3, rep3, csize3, cmass3, os.path.join(OUT_DIR, "cluster_assignment.csv"))
    # round2 solo per confronto, non un prerequisito dichiarato di fasi successive
    df2, cid2, rep2, csize2, cmass2 = cluster_data["round2"]
    save_assignment(df2, cid2, rep2, csize2, cmass2, os.path.join(OUT_DIR, "cluster_assignment_round2.csv"))

    # --- istogramma dimensioni cluster ---
    # Distribuzione fortemente right-skewed (pochi cluster enormi, code di
    # singleton) -> bin log-spaziati + assi log-log, altrimenti quasi tutte
    # le barre sono invisibili su scala lineare.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name in zip(axes, ["round3", "round2"]):
        _, _, _, csize, _ = cluster_data[name]
        max_size = int(csize.max())
        bins = np.unique(np.round(np.geomspace(1, max_size + 1, 40)).astype(int))
        ax.hist(csize, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("dimensione cluster (# sequenze uniche, scala log)")
        ax.set_title(f"{name} (N_eff={cluster_data[name][3].size})")
    axes[0].set_ylabel("# cluster (scala log)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cluster_sizes.png"), dpi=150)
    plt.close(fig)

    # --- curva di Lorenz della massa di read, con annotazione copertura 50%/90% ---
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"round3": "darkorange", "round2": "steelblue"}
    markers = {0.50: "o", 0.90: "s"}
    for name in ["round3", "round2"]:
        _, _, _, _, cmass = cluster_data[name]
        frac_clusters, cum_mass = lorenz_curve(cmass)
        ax.plot(frac_clusters, cum_mass, color=colors[name], label=name)
        n_cl = len(cmass)
        for pct in (0.50, 0.90):
            n_needed = results[name][f"n_clusters_for_{int(pct*100)}pct_read_mass"]
            x_top = 1 - n_needed / n_cl  # top n_needed cluster (massa più alta) = coda destra della curva
            ax.plot(x_top, 1 - pct, markers[pct], color=colors[name], ms=6)
            ax.annotate(f"{n_needed}", (x_top, 1 - pct), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=colors[name])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="uguaglianza perfetta")
    ax.set_xlabel("frazione cumulata di cluster (ordinati per massa crescente)")
    ax.set_ylabel("frazione cumulata di massa read")
    ax.set_title("Curva di Lorenz — concentrazione della massa di read nei cluster\n(marker: soglie 50%=cerchio / 90%=quadrato, letti dall'alto)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lorenz_curve.png"), dpi=150)
    plt.close(fig)

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
