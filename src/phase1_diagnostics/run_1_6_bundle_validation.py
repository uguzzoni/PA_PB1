"""
run_1_6_bundle_validation.py

Fase 1.6 del piano di design (Workplan_phase1_instantiated.md §1.6): verifica
lato PA_PB1, indipendente dal codice del fork, dell'artefatto multi-replica
(save_bundle/load_bundle/make_energy_fn/make_energy_aux in
methods/colabdesign_energy_guidance/colabdesign/energy_model/model_3layer_v2.py).

I test numerici veri e propri sono in energy_guidance/test_model_3layer_v2.py
(nel fork, eseguiti come parte di questo script per un unico report). Qui si
aggiunge: (a) la costruzione di un bundle "placeholder" a 1 replica con le
statistiche di standardizzazione reali di 1.2 (nessuna replica aggiuntiva
esiste ancora, dipende da 1.3), salvato in data/energy_model_params/ come da
piano; (b) il report dei 7 criteri di accettazione del piano madre, inclusi
quelli NON verificabili in questo ambiente (richiedono AfDesign/PDB/GPU).

Usage:
    JAX_PLATFORMS=cpu uv run python src/phase1_diagnostics/run_1_6_bundle_validation.py
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORK_ROOT = "/home/guido/Projects/protein_design/methods/colabdesign_energy_guidance"
WEIGHTS_PATH = os.path.join(REPO_ROOT, "data", "energy_model_params",
                             "PNB_2R_3lay_negbinom_energy_model_weights.json")
BUNDLE_PATH = os.path.join(REPO_ROOT, "data", "energy_model_params",
                            "PNB_2R_3lay_negbinom_energy_model_bundle_v1.npz")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase1", "1_6_bundle_packaging")

with open(os.path.join(REPO_ROOT, "results", "phase1", "1_2_standardization", "standardization_params.json")) as f:
    STD_PARAMS = json.load(f)

sys.path.insert(0, FORK_ROOT)
from colabdesign.energy_model import model_3layer_v2 as v2  # noqa: E402


def run_fork_unit_tests():
    script = os.path.join(FORK_ROOT, "energy_guidance", "test_model_3layer_v2.py")
    env = dict(os.environ, COLABDESIGN_ENERGY_WEIGHTS_PATH=WEIGHTS_PATH,
               JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
    result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def build_placeholder_bundle():
    """Bundle a 1 replica (M=1) con mu/sigma reali di 1.2 — placeholder in attesa
    delle repliche vere di 1.3/1.3.a. Rifatto ad ogni esecuzione (deterministico)."""
    replica_specs = [{
        "weights_path": WEIGHTS_PATH,
        "mu": STD_PARAMS["mu_train"],
        "sigma": STD_PARAMS["sigma_train"],
    }]
    v2.save_bundle(replica_specs, BUNDLE_PATH, metadata={
        "note": "placeholder M=1 in attesa delle repliche di §1.3/1.3.a — non usare come ensemble reale",
        "phase0_metrics_source": "results/phase0/0_1_simplex_variance/metrics.json",
    })
    bundle = v2.load_bundle(BUNDLE_PATH)
    print(f"Bundle placeholder scritto in {BUNDLE_PATH} (M={bundle['M']}, binder_len={bundle['binder_len']})")
    return bundle


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Esecuzione test unitari (energy_guidance/test_model_3layer_v2.py, nel fork)...")
    tests_ok = run_fork_unit_tests()

    print("\nCostruzione bundle placeholder (M=1, mu/sigma reali di §1.2)...")
    bundle = build_placeholder_bundle()

    print("\nVerifica indipendente del bundle appena scritto (round-trip)...")
    energy_fn = v2.make_energy_fn(bundle, mode="st", kappa=0.0)
    aux = v2.make_energy_aux(bundle, mode="st")
    AAS_AF = "ARNDCQEGHILKMFPSTWYV"

    def af_probs(seq):
        import numpy as np
        idx = {c: i for i, c in enumerate(AAS_AF)}
        x = np.zeros((1, 15, 20), dtype="float32")
        for i, c in enumerate(seq):
            x[0, i, idx[c]] = 1.0
        return jnp.asarray(x)

    seq = "MDFNPWLLFLKVPAQ"  # energia nota da Fase 0: -1.0169219970703125 (raw)
    raw_expected = -1.0169219970703125
    z_expected = (raw_expected - bundle["mu"][0]) / bundle["sigma"][0]
    got = float(energy_fn(af_probs(seq)))
    bundle_roundtrip_ok = abs(got - float(z_expected)) < 1e-4
    print(f"  {seq}: bundle(st,kappa=0)={got:.6f}  atteso (raw-mu)/sigma={float(z_expected):.6f}  "
          f"{'OK' if bundle_roundtrip_ok else 'FALLITO'}")
    aux_out = aux(af_probs(seq))
    gap_ok = abs(float(aux_out["gap"])) < 1e-5
    print(f"  gap (modalita' st, bundle appena caricato da disco) = {float(aux_out['gap']):.2e} "
          f"({'OK' if gap_ok else 'FALLITO'})")

    # --- report dei 7 criteri di accettazione del piano madre (Workplan_peptidi_PA-PB1.md §1.6) ---
    criteria = [
        {"id": 1, "criterio": "Coerenza ai vertici (kappa=0 riproduce l'energia standardizzata)",
         "esito": "verificato", "dettaglio": "test 'media ensemble (kappa=0) coerente col calcolo a mano', energy_guidance/test_model_3layer_v2.py"},
        {"id": 2, "criterio": "Gap identicamente nullo in modalita' st",
         "esito": "verificato", "dettaglio": "single replica e bundle M=3 (test), bundle M=1 reale su disco (questo script)"},
        {"id": 3, "criterio": "Gradiente non nullo",
         "esito": "verificato", "dettaglio": "mlp_forward_st, 20/20 punti campionati"},
        {"id": 4, "criterio": "Gradiente di kappa*sigma finito e non nullo dove le repliche concordano",
         "esito": "verificato parzialmente", "dettaglio": "finito e non-NaN verificato; non-nullita' non isolata esplicitamente dal termine kappa=0"},
        {"id": 5, "criterio": "Retrocompatibilita' (energy_fn=None riproduce la traiettoria originale a parita' di seed)",
         "esito": "NON verificabile in questo ambiente", "dettaglio": "richiede un run AfDesign reale (PDB + parametri AF2) — perimetro CPU-only di Fase 1, da fare in Fase 2 §2.1"},
        {"id": 6, "criterio": "Modalita' legacy (mode='soft', kappa=0 riproduce i run precedenti)",
         "esito": "verificato", "dettaglio": "make_energy_fn(path,...) invariato dopo il refactor a bundle, errore 0.00e+00"},
        {"id": 7, "criterio": "Test placeholder (glicina) esteso a kappa>0",
         "esito": "NON fatto", "dettaglio": "test_energy_guidance.py (integrazione AfDesign/PDB) non modificato in questa sessione — richiede GPU/PDB per essere eseguito comunque"},
    ]
    n_ok = sum(c["esito"] == "verificato" for c in criteria)
    report = {
        "unit_tests_passed": tests_ok,
        "bundle_roundtrip_ok": bundle_roundtrip_ok,
        "bundle_gap_ok": gap_ok,
        "bundle_path": BUNDLE_PATH,
        "bundle_metadata": bundle["metadata"],
        "acceptance_criteria": criteria,
        "n_criteria_fully_verified": n_ok,
        "n_criteria_total": len(criteria),
    }
    with open(os.path.join(OUT_DIR, "acceptance_test_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nCriteri di accettazione verificati: {n_ok}/{len(criteria)} (2 richiedono AfDesign/PDB/GPU, fuori dal perimetro di Fase 1)")
    print(f"Output scritto in {OUT_DIR}")
    sys.exit(0 if (tests_ok and bundle_roundtrip_ok and gap_ok) else 1)


if __name__ == "__main__":
    main()
