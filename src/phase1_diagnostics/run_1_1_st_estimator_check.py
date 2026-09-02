"""
run_1_1_st_estimator_check.py

Fase 1.1 del piano di design (Workplan_phase1_instantiated.md §1.1): verifica
indipendente, lato PA_PB1, delle modifiche fatte in
methods/colabdesign_energy_guidance/colabdesign/energy_model/model_3layer_v2.py
(mlp_forward_st, make_energy_aux). I test numerici puntuali (coerenza ai
vertici, gradiente non nullo, retrocompatibilita' con model_3layer legacy)
sono in energy_guidance/test_model_3layer_v2.py nel fork — qui si aggiunge
solo la verifica visiva lungo una traiettoria, che quel file non produce.

Nota sulla traiettoria: e' **sintetica** (interpolazione soft->hard verso un
vertice fisso via temperatura decrescente), non un run reale di AfDesign — non
richiede GPU/PDB/parametri AF2, a differenza del piano madre che nel testo
originale immagina di rigenerare un run reale minimale. Una verifica su
traiettoria REALE resta comunque utile e va fatta in Fase 2 (§2.1), quando
sara' comunque necessario eseguire ColabDesign per il confronto soft/ST.

Usage:
    JAX_PLATFORMS=cpu uv run python src/phase1_diagnostics/run_1_1_st_estimator_check.py
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORK_ROOT = "/home/guido/Projects/protein_design/methods/colabdesign_energy_guidance"
WEIGHTS_PATH = os.path.join(REPO_ROOT, "data", "energy_model_params",
                             "PNB_2R_3lay_negbinom_energy_model_weights.json")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase1", "1_1_st_estimator")

with open(os.path.join(REPO_ROOT, "results", "phase0", "0_1_simplex_variance", "metrics.json")) as f:
    SIGMA_TRAIN = json.load(f)["sigma_train"]

sys.path.insert(0, FORK_ROOT)
from colabdesign.energy_model import model_3layer_v2 as v2  # noqa: E402

BINDER_LEN = 15
N_AA = 20


def run_unit_tests():
    """Esegue energy_guidance/test_model_3layer_v2.py come subprocesso (stesso
    interprete/venv di questo script) e ne cattura l'esito — evita di duplicare
    qui la logica dei test, che vive nel fork insieme al codice che testano."""
    script = os.path.join(FORK_ROOT, "energy_guidance", "test_model_3layer_v2.py")
    env = dict(os.environ, COLABDESIGN_ENERGY_WEIGHTS_PATH=WEIGHTS_PATH,
               JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
    result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0, result.stdout


def synthetic_annealing_trajectory(params, rng, n_steps=100, temp_min=0.02):
    """Interpolazione soft->hard verso un vertice fisso x_target, temp da 1 a
    temp_min (schema simile allo stadio di annealing di design_3stage, T: 1->0).
    L'argmax di softmax(logit_base/temp) e' uguale a x_target per costruzione,
    per ogni temp>0 (logit_base ha gia' il suo massimo in x_target) — quindi
    e_hard_finale = mlp_forward(x_target) e' il riferimento esatto lungo tutta
    la traiettoria, non un valore stimato a posteriori come in 0.7.
    """
    target_idx = rng.integers(0, N_AA, size=BINDER_LEN)
    x_target = np.zeros((BINDER_LEN, N_AA + 1), dtype=np.float32)
    x_target[np.arange(BINDER_LEN), target_idx] = 1.0
    logit_base = x_target[:, :N_AA] * 4.0  # margine netto, argmax stabile a qualunque temp>0

    temps = np.geomspace(1.0, temp_min, n_steps)
    e_hard_final = float(v2.mlp_forward(params, jnp.asarray(x_target[None])))

    gap_soft, gap_st = [], []
    for temp in temps:
        logits = logit_base / temp
        soft20 = np.exp(logits - logits.max(axis=-1, keepdims=True))
        soft20 /= soft20.sum(axis=-1, keepdims=True)
        soft = np.concatenate([soft20, np.zeros((BINDER_LEN, 1), dtype=np.float32)], axis=-1)
        x = jnp.asarray(soft[None])

        e_soft = float(v2.mlp_forward(params, x))
        e_st = float(v2.mlp_forward_st(params, x))
        gap_soft.append(e_soft - e_hard_final)
        gap_st.append(e_st - e_hard_final)

    return temps, np.array(gap_soft), np.array(gap_st)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Esecuzione test unitari (energy_guidance/test_model_3layer_v2.py, nel fork)...")
    tests_ok, tests_stdout = run_unit_tests()

    print("\nGenerazione traiettoria sintetica di annealing (gap soft vs straight-through)...")
    params = v2.load_energy_model(WEIGHTS_PATH)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    max_gap_st_seen = 0.0
    for ax_idx in range(3):
        temps, gap_soft, gap_st = synthetic_annealing_trajectory(params, rng, n_steps=150)
        max_gap_st_seen = max(max_gap_st_seen, np.abs(gap_st).max())
        ax = axes[ax_idx]
        ax.plot(temps, gap_soft / SIGMA_TRAIN, label="mode='soft' (legacy)", color="firebrick")
        ax.plot(temps, gap_st / SIGMA_TRAIN, label="mode='st' (§1.1)", color="steelblue")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xscale("log")
        ax.invert_xaxis()  # temp decrescente verso destra, come l'andamento reale dell'annealing
        ax.set_xlabel("temperatura (scala log, decrescente →)")
        ax.set_ylabel(r"gap(t) / $\sigma_{\rm train}$")
        ax.set_title(f"traiettoria sintetica #{ax_idx+1}")
        if ax_idx == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Verifica visiva §1.1: gap identicamente nullo in modalità straight-through,\n"
                 "aperto in modalità soft — traiettoria sintetica (non un run AfDesign reale, vedi docstring)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gap_zero_verification.png"), dpi=150)
    plt.close(fig)

    report = {
        "unit_tests_passed": tests_ok,
        "unit_tests_source": "methods/colabdesign_energy_guidance/energy_guidance/test_model_3layer_v2.py",
        "synthetic_trajectory_max_abs_gap_st": float(max_gap_st_seen),
        "synthetic_trajectory_note": "traiettoria sintetica (interpolazione verso un vertice fisso), non un run AfDesign reale — vedi run_1_1_st_estimator_check.py docstring",
    }
    with open(os.path.join(OUT_DIR, "test_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTest unitari: {'OK' if tests_ok else 'FALLITI'}")
    print(f"Gap massimo |st| su traiettorie sintetiche: {max_gap_st_seen:.2e} (atteso: 0)")
    print(f"\nOutput scritto in {OUT_DIR}")
    sys.exit(0 if tests_ok else 1)


if __name__ == "__main__":
    main()
