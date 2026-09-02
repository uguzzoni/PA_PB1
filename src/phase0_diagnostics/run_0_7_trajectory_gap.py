"""
run_0_7_trajectory_gap.py

Fase 0.7 del piano di design (Workplan_phase0_instantiated.md §0.7): quanto il
valore di energia effettivamente ottimizzato durante gli stadi soft diverge
dall'energia della sequenza discreta corrispondente, e come questa divergenza
dipende dal protocollo e da w_E.

Dati disponibili: 9 file .out in src/slurm_subs/ (campagna run2_wide, 2 job
array SLURM parziali) — nessun'altra campagna ha traiettorie salvate. Non è
salvato a^(t)/i logit, solo scalari per-step (loss, i_con, plddt, ptm, i_ptm,
energy). Il Delta(t) = E(a^(t)) - E(onehot(argmax a^(t))) del piano madre non
è quindi calcolabile alla lettera: si usa il proxy

    delta_proxy(t) = E_soft(t) - E_hard,finale

dove E_hard,finale è il valore di energia a cui lo stadio 3 converge per quel
blocco (verificato costante entro traiettoria, non un singolo campione
rumoroso). Vedi Workplan §0.7 per la motivazione e i limiti di questa scelta
(confermata dall'utente 2026-08-12).

Convenzione di segno: delta_proxy è tipicamente <= 0 nello stadio soft
(l'energia soft scende artificialmente sotto il valore reale della sequenza
discreta). "gap(t) = -delta_proxy(t)" è quindi >= 0 e cresce con l'ampiezza
del guadagno illusorio — è la quantità riportata come Delta_max negli
scalari riassuntivi (il piano madre la chiama "massimo di Delta(t)": qui si
riporta la sua ampiezza, non il valore signed, per evitare l'ambiguità di
segno insita nella formula letterale).

Usage:
    uv run python src/phase0_diagnostics/run_0_7_trajectory_gap.py
"""

import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SLURM_SUBS_DIR = os.path.join(REPO_ROOT, "src", "slurm_subs")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase0", "0_7_trajectory_gap")

sys.path.insert(0, SLURM_SUBS_DIR)
from parse_slurm_out import PROTOCOL_CONFIGS, RE_GEN_BLOCK, RE_OPT_BLOCK, RE_PROTOCOL  # noqa: E402

with open(os.path.join(REPO_ROOT, "results", "phase0", "0_1_simplex_variance", "metrics.json")) as f:
    SIGMA_TRAIN = json.load(f)["sigma_train"]

FLOAT = r'(-?[\d]+\.?[\d]*(?:[eE][-+]?\d+)?)'
# Il campo "energy" è assente dalla riga quando w_E=0 su tutti gli stadi (verificato:
# opt_anneal_noenergy non lo stampa affatto per-step, solo nella riga finale "-> seq=...")
# — reso opzionale, non un bug del regex precedente.
RE_STEP = re.compile(
    rf'^(\d+) models \[\d+\] recycles \d+ hard (\d) soft (\d) temp {FLOAT}'
    rf' loss {FLOAT} i_con {FLOAT} plddt {FLOAT} ptm {FLOAT} i_ptm {FLOAT}(?: energy {FLOAT})?'
)
RE_GEN_SEQ = re.compile(rf'^\s+seq=(\S+)\s+loss={FLOAT}\s+loss_af={FLOAT}\s+energy={FLOAT}\s+i_ptm={FLOAT}')
RE_OPT_SEQ = re.compile(rf'^\s+→\s+seq=(\S+)\s+loss={FLOAT}\s+loss_af={FLOAT}\s+energy={FLOAT}\s+i_ptm={FLOAT}')
HARD_STABILITY_TOL = 1e-3  # E_hard deve essere ~costante entro un blocco (vedi Workplan §0.7)


def stage_boundaries(protocol_cfg: dict) -> dict:
    """{stage_label: (first_step, last_step)} cumulativo, stadi con iters=0 esclusi."""
    order = [k for k in ("stage_1a", "stage_1b", "stage_2", "stage_3") if k in protocol_cfg]
    bounds, cum = {}, 0
    for k in order:
        iters = protocol_cfg[k]["iters"]
        if iters:
            bounds[k] = (cum + 1, cum + iters)
            cum += iters
    return bounds


def stage_of_step(step: int, bounds: dict):
    for stage, (lo, hi) in bounds.items():
        if lo <= step <= hi:
            return stage
    return None


def parse_out_file(path: str):
    lines = open(path, errors="replace").read().splitlines()
    proto_name = None
    for line in lines[:10]:
        m = RE_PROTOCOL.match(line)
        if m:
            proto_name = m.group(1)
            break
    if proto_name is None or proto_name not in PROTOCOL_CONFIGS:
        raise ValueError(f"{path}: protocollo non riconosciuto ({proto_name!r})")

    cfg = PROTOCOL_CONFIGS[proto_name]
    ptype = cfg["type"]
    bounds = stage_boundaries(cfg["protocol"])
    w_E = {stage: cfg["protocol"][stage]["energy_weight"] for stage in bounds}
    re_block = RE_OPT_BLOCK if ptype == "opt" else RE_GEN_BLOCK
    re_seq = RE_OPT_SEQ if ptype == "opt" else RE_GEN_SEQ

    blocks, current = [], None
    for line in lines:
        m = re_block.match(line)
        if m:
            if current is not None and (current["steps"] or current["final_seq_energy"] is not None):
                blocks.append(current)
            current = {
                "seed": int(m.group(4) if ptype == "opt" else m.group(3)),
                "input_seq": m.group(3) if ptype == "opt" else None,
                "steps": [], "final_seq_energy": None, "final_seq": None,
            }
            continue
        if current is None:
            continue
        sm = RE_STEP.match(line)
        if sm:
            step = int(sm.group(1))
            energy_str = sm.group(10)
            current["steps"].append({
                "step": step, "hard": int(sm.group(2)), "temp": float(sm.group(4)),
                "i_ptm": float(sm.group(9)),
                "energy": float(energy_str) if energy_str is not None else np.nan,
                "stage": stage_of_step(step, bounds),
            })
            continue
        qm = re_seq.match(line)
        if qm:
            current["final_seq_energy"] = float(qm.group(4))
            current["final_seq"] = qm.group(1)
    if current is not None and (current["steps"] or current["final_seq_energy"] is not None):
        blocks.append(current)
    return proto_name, ptype, w_E, blocks


def summarize_block(proto_name, ptype, w_E, block_id, block):
    steps = pd.DataFrame(block["steps"]) if block["steps"] else pd.DataFrame(
        columns=["step", "hard", "temp", "i_ptm", "energy", "stage"])
    hard_steps = steps[steps["hard"] == 1]
    soft_steps = steps[steps["hard"] == 0]
    hard_energy = hard_steps["energy"].dropna()

    row = {
        "protocol": proto_name, "block_id": block_id, "seed": block["seed"],
        "input_seq": block["input_seq"], "final_seq": block["final_seq"], "n_steps": len(steps),
        "n_hard_steps": len(hard_steps), "n_soft_steps": len(soft_steps),
        "w_E_stage3": w_E.get("stage_3"),
    }

    # w_E=0 su tutti gli stadi (opt_anneal_noenergy): "energy" non è stampata per-step,
    # solo nella riga finale "-> seq=..." — un solo valore per blocco, nessuna traccia.
    if len(hard_energy) == 0:
        if block["final_seq_energy"] is not None:
            row["E_hard_final"] = block["final_seq_energy"]
            row["E_hard_final_source"] = "final_seq_line (nessuna traccia per-step: w_E=0)"
        else:
            row["E_hard_final"] = np.nan
            row["E_hard_final_source"] = None
        row.update({"hard_stable": None, "final_i_ptm": float(hard_steps["i_ptm"].mean()) if len(hard_steps) else np.nan,
                     "gap_max": np.nan, "t_star": np.nan, "stage_star": None,
                     "gap_integral_soft": np.nan, "gap_fine_temp": np.nan, "E_soft_min": np.nan})
        return row, steps

    E_hard_final = float(hard_energy.median())
    hard_stable = bool((hard_energy.max() - hard_energy.min()) < HARD_STABILITY_TOL)
    row["E_hard_final"] = E_hard_final
    row["E_hard_final_source"] = "hard_steps_median"
    row["hard_stable"] = hard_stable
    row["final_i_ptm"] = float(hard_steps["i_ptm"].mean())

    soft_energy = soft_steps.dropna(subset=["energy"])
    if len(soft_energy) == 0:
        row.update({"gap_max": np.nan, "t_star": np.nan, "stage_star": None,
                     "gap_integral_soft": np.nan, "gap_fine_temp": np.nan, "E_soft_min": np.nan})
        return row, steps
    soft_steps = soft_energy

    delta_proxy = (soft_steps["energy"] - E_hard_final) / SIGMA_TRAIN  # <=0 tipicamente
    gap = -delta_proxy  # >=0, ampiezza del guadagno illusorio
    i_max = gap.idxmax()
    row["gap_max"] = float(gap.loc[i_max])
    row["t_star"] = int(soft_steps.loc[i_max, "step"])
    row["stage_star"] = soft_steps.loc[i_max, "stage"]
    row["gap_integral_soft"] = float(gap.sum())  # passo=1, somma == integrale discreto
    row["E_soft_min"] = float(soft_steps["energy"].min())

    stage2 = soft_steps[soft_steps["stage"] == "stage_2"]
    if len(stage2):
        last = stage2.loc[stage2["step"].idxmax()]
        row["gap_fine_temp"] = float(-(last["energy"] - E_hard_final) / SIGMA_TRAIN)
    else:
        row["gap_fine_temp"] = np.nan

    return row, steps


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_files = sorted(glob.glob(os.path.join(SLURM_SUBS_DIR, "*.out")))
    print(f"{len(out_files)} file .out trovati in {SLURM_SUBS_DIR}\n")

    all_traj, all_summary = [], []
    for path in out_files:
        proto_name, ptype, w_E, blocks = parse_out_file(path)
        print(f"{os.path.basename(path)}: protocollo={proto_name} tipo={ptype} blocchi={len(blocks)}")
        for i, block in enumerate(blocks):
            block_id = f"{proto_name}__{os.path.basename(path)}__{i}"
            row, steps = summarize_block(proto_name, ptype, w_E, block_id, block)
            all_summary.append(row)
            steps["block_id"] = block_id
            steps["protocol"] = proto_name
            all_traj.append(steps)

    summary = pd.DataFrame(all_summary)
    traj = pd.concat(all_traj, ignore_index=True)

    n_unstable = int((summary["hard_stable"] == False).sum())  # noqa: E712
    print(f"\nBlocchi totali: {len(summary)}  |  E_hard instabile in {n_unstable} blocchi "
          f"(sequenza ancora in cambiamento nello stadio hard — verificare)")

    traj.to_csv(os.path.join(OUT_DIR, "trajectories_normalized.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "summary_scalars_by_run.csv"), index=False)

    # --- aggregazione per protocollo ---
    agg = summary.dropna(subset=["gap_max"]).groupby("protocol").agg(
        n_blocks=("gap_max", "size"),
        w_E_stage3=("w_E_stage3", "first"),
        gap_max_median=("gap_max", "median"),
        gap_max_q25=("gap_max", lambda s: s.quantile(0.25)),
        gap_max_q75=("gap_max", lambda s: s.quantile(0.75)),
        gap_integral_median=("gap_integral_soft", "median"),
        final_iptm_median=("final_i_ptm", "median"),
    ).reset_index().sort_values("w_E_stage3")
    agg.to_csv(os.path.join(OUT_DIR, "summary_by_protocol.csv"), index=False)
    print("\nAggregato per protocollo (mediana, IQR):")
    print(agg.to_string(index=False))

    # --- correlazione gap_max vs i_ptm finale ---
    valid = summary.dropna(subset=["gap_max", "final_i_ptm"])
    if len(valid) > 2:
        corr = np.corrcoef(valid["gap_max"], valid["final_i_ptm"])[0, 1]
        print(f"\nCorrelazione (Pearson) gap_max vs i_ptm finale, tutti i protocolli: {corr:.3f}")
    else:
        corr = float("nan")

    fig, ax = plt.subplots(figsize=(6, 5))
    for proto, sub in valid.groupby("protocol"):
        ax.scatter(sub["gap_max"], sub["final_i_ptm"], s=12, alpha=0.6, label=proto)
    ax.set_xlabel(r"gap$_{\max}$  ($\sigma_{\rm train}$)")
    ax.set_ylabel("i_ptm finale (media stadio hard)")
    ax.set_title(f"gap_max vs i_ptm finale (r={corr:.2f})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "delta_max_vs_iptm.png"), dpi=150)
    plt.close(fig)

    # --- gap_max vs w_E ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for proto, sub in valid.groupby("protocol"):
        we = sub["w_E_stage3"].iloc[0]
        ax.scatter([we] * len(sub), sub["gap_max"], s=12, alpha=0.5)
        ax.scatter([we], [sub["gap_max"].median()], color="black", marker="D", s=40, zorder=5)
    ax.set_xlabel(r"$w_E$ (stadio 3)")
    ax.set_ylabel(r"gap$_{\max}$  ($\sigma_{\rm train}$)")
    ax.set_title("gap_max vs peso dell'energia (diamanti = mediana per protocollo)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "delta_max_vs_wE.png"), dpi=150)
    plt.close(fig)

    # --- curve gap(t) per protocollo, small multiples ---
    protocols = sorted(traj["protocol"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    for ax, proto in zip(axes.flat, protocols):
        sub = traj[(traj["protocol"] == proto) & (traj["hard"] == 0)]
        for block_id, block_steps in sub.groupby("block_id"):
            row = summary[summary["block_id"] == block_id]
            if row.empty or pd.isna(row["E_hard_final"].iloc[0]):
                continue
            e_hard = row["E_hard_final"].iloc[0]
            gap_t = -(block_steps["energy"] - e_hard) / SIGMA_TRAIN
            ax.plot(block_steps["step"], gap_t, alpha=0.15, color="steelblue", lw=0.8)
        ax.set_title(proto, fontsize=9)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("step (locale al blocco)")
        ax.set_ylabel(r"gap(t)  ($\sigma_{\rm train}$)")
    for ax in axes.flat[len(protocols):]:
        ax.axis("off")
    fig.suptitle("Traiettorie gap(t) = -(E_soft(t) - E_hard,finale) / sigma_train, per protocollo")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "delta_curves_by_protocol.png"), dpi=150)
    plt.close(fig)

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
