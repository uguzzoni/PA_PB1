"""
build_replica_bundle.py

Fase 1.7 passo B del piano di design (Workplan_phase1_addendum.md §1.7):
costruisce il bundle multi-replica M=10 (famiglia F-only/10%, solo i 10 split
-- decisioni confermate con l'utente il 2026-09-14) a partire dai pesi e dalle
sequenze di training esportati dal passo A (export_replica_weights.jl,
esecuzione manuale, Julia).

Per ciascuna replica calcola mu_m/sigma_m sul training set PROPRIO di quella
replica (§1.6(c)) -- NON riusa mu=3.398753/sigma=2.087558 di §1.2 (statistiche
del modello di produzione sul training set completo): applicarle a tutte le
repliche reintrodurrebbe esattamente l'offset di nuisance che la
standardizzazione per replica esiste per rimuovere.

Prerequisito: eseguire manualmente
PD_energy_model/training/training_2rounds/workplan_phase1/export_replica_weights.jl
(deve passare il passo A0 prima di proseguire).

Usage:
    JAX_PLATFORMS=cpu uv run python src/phase1_diagnostics/build_replica_bundle.py
"""

import csv
import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORK_ROOT = "/home/guido/Projects/protein_design/methods/colabdesign_energy_guidance"
EXPORTED_DIR = os.path.join(REPO_ROOT, "PD_energy_model", "training", "training_2rounds",
                             "workplan_phase1", "exported_replicas")
BUNDLE_PATH = os.path.join(REPO_ROOT, "data", "energy_model_params",
                            "PNB_2R_3lay_negbinom_energy_model_bundle_M10.npz")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase1", "1_7_replica_bundle")
N_REPLICAS = 10
REPLICA_FAMILY = "Fonly_10pct"

sys.path.insert(0, os.path.join(REPO_ROOT, "src", "phase0_diagnostics"))
from data_loading import load_bli_population, try_encode_sequence  # noqa: E402

sys.path.insert(0, FORK_ROOT)
from colabdesign.energy_model import model_3layer_v2 as v2  # noqa: E402

AAS_AF = "ARNDCQEGHILKMFPSTWYV"
AF_IDX = {c: i for i, c in enumerate(AAS_AF)}


def af_probs(seq: str):
    x = np.zeros((1, 15, 20), dtype="float32")
    for i, c in enumerate(seq):
        x[0, i, AF_IDX[c]] = 1.0
    return jnp.asarray(x)


def replica_name(i: int) -> str:
    return f"random_split_{REPLICA_FAMILY}_{i}"


def load_kept_sequences(name: str):
    path = os.path.join(EXPORTED_DIR, f"{name}_kept_sequences.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} mancante -- eseguire prima export_replica_weights.jl (manuale, "
            "PD_energy_model/training/training_2rounds/workplan_phase1/)")
    with open(path) as f:
        raw_seqs = [line.strip() for line in f if line.strip()]
    encoded, kept, n_dropped = [], [], 0
    for s in raw_seqs:
        x = try_encode_sequence(s, allow_gap=True)
        if x is None:
            n_dropped += 1
            continue
        encoded.append(x)
        kept.append(s)
    return np.stack(encoded), kept, n_dropped


def compute_mu_sigma(weights_path: str, kept_onehot: np.ndarray):
    params = v2.load_energy_model(weights_path)
    forward_batch = jax.jit(jax.vmap(lambda xi: v2.mlp_forward(params, xi[None])))
    E = np.asarray(forward_batch(jnp.asarray(kept_onehot)))
    return float(E.mean()), float(E.std())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    replica_specs, replica_rows, kept_onehot_by_replica = [], [], {}
    print(f"Caricamento {N_REPLICAS} repliche (famiglia {REPLICA_FAMILY})...")
    for i in range(1, N_REPLICAS + 1):
        name = replica_name(i)
        weights_path = os.path.join(EXPORTED_DIR, f"{name}_weights.json")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"{weights_path} mancante -- eseguire prima export_replica_weights.jl "
                "(manuale, PD_energy_model/training/training_2rounds/workplan_phase1/)")
        kept_onehot, kept_seqs, n_dropped = load_kept_sequences(name)
        mu, sigma = compute_mu_sigma(weights_path, kept_onehot)
        print(f"  {name}: N_kept={len(kept_seqs)} (scartate {n_dropped})  mu={mu:.6f}  sigma={sigma:.6f}")
        replica_specs.append({"weights_path": weights_path, "mu": mu, "sigma": sigma})
        replica_rows.append({"replica": name, "n_kept_sequences": len(kept_seqs),
                              "n_dropped": n_dropped, "mu": mu, "sigma": sigma})
        kept_onehot_by_replica[name] = kept_onehot

    with open(os.path.join(OUT_DIR, "replica_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(replica_rows[0].keys()))
        writer.writeheader()
        writer.writerows(replica_rows)

    metadata = {
        "replica_family": REPLICA_FAMILY,
        "bundle_composition": ("solo i 10 split (nessun modello completo aggiunto) -- "
                                "decisione confermata con l'utente 2026-09-14"),
        "n_replicas": N_REPLICAS,
        "source_addendum": "Workplan_phase1_addendum.md §1.7",
    }
    print(f"\nScrittura bundle in {BUNDLE_PATH}...")
    v2.save_bundle(replica_specs, BUNDLE_PATH, metadata=metadata)

    # --- verifiche di accettazione (addendum §1.7, tabella "Verifiche di accettazione") ---
    print("\nVerifiche di accettazione...")
    bundle = v2.load_bundle(BUNDLE_PATH)

    # test 2: shape
    shape_ok = bundle["M"] == N_REPLICAS
    print(f"  [2] shape: M={bundle['M']} (atteso {N_REPLICAS}) -- {'OK' if shape_ok else 'FALLITO'}")

    # test 3: standardizzazione per replica, su ciascun training set PROPRIO
    std_check = []
    for i in range(1, N_REPLICAS + 1):
        name = replica_name(i)
        weights_path = replica_specs[i - 1]["weights_path"]
        params = v2.load_energy_model(weights_path)
        forward_batch = jax.jit(jax.vmap(lambda xi: v2.mlp_forward(params, xi[None])))
        E = np.asarray(forward_batch(jnp.asarray(kept_onehot_by_replica[name])))
        mu_m, sigma_m = replica_specs[i - 1]["mu"], replica_specs[i - 1]["sigma"]
        z = (E - mu_m) / sigma_m
        std_check.append({"replica": name, "mean_z": float(z.mean()), "std_z": float(z.std())})
    mean_z_ok = all(abs(r["mean_z"]) < 1e-4 for r in std_check)
    std_z_ok = all(abs(r["std_z"] - 1.0) < 1e-4 for r in std_check)
    print(f"  [3] standardizzazione per replica: mean(z)~0 {'OK' if mean_z_ok else 'FALLITO'}, "
          f"std(z)~1 {'OK' if std_z_ok else 'FALLITO'}")

    # test 4: dispersione fra repliche non degenere sulle 23 sequenze BLI
    bli, _, _ = load_bli_population()
    aux_bundle_soft = v2.make_energy_aux(bundle, mode="soft")
    sigmas_bli = np.array([float(aux_bundle_soft(af_probs(s))["sigma"]) for s in bli["Sequence"]])
    sigma_degenerate = bool(np.allclose(sigmas_bli, sigmas_bli[0], atol=1e-6))
    print(f"  [4] sigma_repliche sulle 23 BLI: min={sigmas_bli.min():.4f} max={sigmas_bli.max():.4f} -- "
          f"{'DEGENERE (indagare, vedi §1.6(c))' if sigma_degenerate else 'OK, non degenere'}")

    # test 5: gap identicamente nullo in modalita' st, anche con M=10
    aux_bundle_st = v2.make_energy_aux(bundle, mode="st")
    gaps = np.array([float(aux_bundle_st(af_probs(s))["gap"]) for s in bli["Sequence"]])
    gap_ok = bool(np.max(np.abs(gaps)) < 1e-5)
    print(f"  [5] gap in modalita' st (M=10): max|gap|={np.max(np.abs(gaps)):.2e} -- {'OK' if gap_ok else 'FALLITO'}")

    # test 6: retrocompatibilita' -- make_energy_fn con path singolo continua a funzionare
    single_path = os.path.join(REPO_ROOT, "data", "energy_model_params",
                                "PNB_2R_3lay_negbinom_energy_model_weights.json")
    energy_fn_single = v2.make_energy_fn(single_path, mode="soft")
    seq, raw_expected = "MDFNPWLLFLKVPAQ", -1.0169219970703125
    got_single = float(energy_fn_single(af_probs(seq)))
    backward_compat_ok = abs(got_single - raw_expected) < 1e-4
    print(f"  [6] retrocompatibilita' path singolo: {got_single:.6f} (atteso {raw_expected:.6f}) -- "
          f"{'OK' if backward_compat_ok else 'FALLITO'}")

    validation_report = {
        "bundle_path": BUNDLE_PATH,
        "replica_family": REPLICA_FAMILY,
        "n_replicas": N_REPLICAS,
        "test_1_regression_export": "verificato manualmente in export_replica_weights.jl passo A0 (Julia, fuori da questo script)",
        "test_2_shape_ok": shape_ok,
        "test_3_standardization_per_replica": std_check,
        "test_3_mean_z_ok": mean_z_ok,
        "test_3_std_z_ok": std_z_ok,
        "test_4_sigma_replicas_bli": sigmas_bli.tolist(),
        "test_4_sigma_degenerate": sigma_degenerate,
        "test_5_gap_st_max_abs": float(np.max(np.abs(gaps))),
        "test_5_gap_ok": gap_ok,
        "test_6_backward_compat_ok": backward_compat_ok,
    }
    with open(os.path.join(OUT_DIR, "bundle_validation.json"), "w") as f:
        json.dump(validation_report, f, indent=2)

    all_ok = shape_ok and mean_z_ok and std_z_ok and (not sigma_degenerate) and gap_ok and backward_compat_ok
    print(f"\nTutte le verifiche: {'OK' if all_ok else 'ALMENO UNA FALLITA -- vedi bundle_validation.json'}")
    print(f"Output scritto in {OUT_DIR}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
