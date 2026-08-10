# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computational binder design campaign targeting the influenza PA–PB1 polymerase interface. The workflow combines AF2-based sequence hallucination (ColabDesign) with ML energy guidance (PhageNegBinom MLP), followed by structural scoring with Boltz-1 and AlphaFold3.

## Folder Structure

```
PA_PB1/
├── PD_energy_model/       # Energy model: Julia training/analysis + trained params (nested git repo, git-ignored here)
├── src/
│   ├── config.py                # Env var validation, hotspot residues
│   ├── check_gen_seq.py         # Sanity-checks generated sequences
│   ├── collect_af3_results.py   # Aggregates AF3 outputs (imports the root-level config.py, run from repo root)
│   ├── generative_protocols/    # All generate_/optimize_/score_ scripts + model_energy_guidance.py
│   ├── analysis/                # Notebooks analyzing generated sequences (+ results/ CSV exports)
│   └── slurm_subs/              # SLURM launchers, prediction runner, log parsers
├── data/                   # Inputs: pdbs/, energy_model_params/, seed & measured sequences
└── results/                # colabdesign/ (tracked), af3/ + boltz/ (git-ignored, large)
```

See `README.md` for the fully annotated tree.

## Environment Setup

Scripts require these environment variables (set via `.env` at the repo root — copy `.env.example`):

```bash
export PA_PB1_PDB=/path/to/2ZNL_chainA_400plus.pdb
export COLABDESIGN_PARAMS_DIR=/path/to/af2_params/
export ENERGY_WEIGHTS_PATH=/path/to/PNB_2R_3lay_negbinom_energy_model_weights.json
export COLABDESIGN_RESULTS_DIR=/path/to/colabdesign_results
export BOLTZ_RESULTS_DIR=/path/to/boltz_results
export AF3_RESULTS_DIR=/path/to/af3_results
```

`src/config.py` validates these at import time via `_require()` and exposes `PDB`, `PARAMS_DIR`, `WEIGHTS_PATH`, `RESULTS_DIR`, `BOLTZ_RESULTS_DIR`, `AF3_RESULTS_DIR`, and the hardcoded list of 27 PA hotspot residues used in `prep_inputs()`.

There is a second, near-duplicate `config.py` at the repo root (same variables plus `AF3_SKIP`) used only by `src/collect_af3_results.py`, which does a bare `import config` and must be run from the repo root. Scripts under `src/generative_protocols/` do `from config import ...` against `src/config.py` and need `src/` on `PYTHONPATH` (SLURM scripts set this — see below).

SLURM scripts source `.env` and set `PYTHONPATH` before calling Python; the launcher scripts currently point `SCRIPT_DIR` at pre-reorganization paths (`$CODEBASE/prg/PA_PB1`) and reference `generate_sequences.py` etc. directly — update these to `src/generative_protocols/` when next touching them.

## Running Jobs

**Design campaign (SLURM array, 12 protocols, 1 GPU each):**
```bash
cd src/slurm_subs
sbatch --array=0-11 submit_protocols.sh
```

**Wide search (6 protocols, higher sequence counts):**
```bash
sbatch --array=0-5 submit_wide_search.sh
```

**Score sequences with AF2 + Boltz (array per sequence, then collect):**
```bash
sbatch submit_predictions.sh <seq_file> <output_dir>
# collect job is auto-submitted with --dependency=afterok
```

**Run a single design script directly (requires GPU + env vars):**
```bash
source .env
export PYTHONPATH=src:$PYTHONPATH
python src/generative_protocols/generate_sequences.py -n 5 --ew_1a 0.02 --ew_1b 0.05 --ew_2 0.02 --ew_3 0.05
python src/generative_protocols/optimize_sequences.py --seq_file data/input_sequences.txt -n 4 --ew_2 0.05 --ew_3 0.20
python src/generative_protocols/score_sequences.py --seq ACDEFGHIKLMNPQR
```

## Code Architecture

### Design workflow (two modes)

All of the following live in `src/generative_protocols/`.

**Generate** (`generate_sequences.py`): 4-stage protocol, starts from random logits.
1. Stage 1a (50 iter): soft, dropout, T=1.0, energy weight `ew_1a`
2. Stage 1b (50 iter): soft, dropout, T=1.0, energy weight `ew_1b`
3. Stage 2 (50 iter): soft, dropout, T anneals 1→0, weight `ew_2`
4. Stage 3 (30 iter): hard (discrete), no dropout, T≈0, weight `ew_3`

**Optimize** (`optimize_sequences.py`): 2-stage protocol, starts from input sequences.
- Stage 2 (annealing) is optional — skip with `--iters_2 0` for deterministic hard-only mode.
- `-n` seeds control stochasticity; stage 3 alone is fully deterministic.

Multimer variants (`generate_sequences_multimer.py`, `optimize_sequences_multimer.py`) use AF2-Multimer instead of pTM models. `generate_sequences_baseline.py` is the ColabDesign-only (no energy) baseline, and `generate_sequences_2enw.py` targets a 2ENW-derived template.

### Energy guidance

`src/generative_protocols/model_energy_guidance.py` wraps ColabDesign's `AfDesign` and injects a 3-layer MLP energy term into the loss. The MLP weights are trained in `PD_energy_model/` (Julia) and exported to `data/energy_model_params/`. The MLP operates on soft logits (sequence probability distributions), outputting a scalar energy. The weight balances AF2 structural confidence vs. predicted binding affinity:

```
loss_total = loss_af + energy_weight * energy
```

Energy is always computed and logged regardless of weight; `energy_weight=0` gives pure ColabDesign.

### Output format

All design scripts write a single JSON:
```json
{
  "protocol": { "stage_1a": {"iters": 50, "soft": true, "energy_weight": 0.05, ...}, ... },
  "results": [
    {"seed": 0, "seq": "ACDE...", "loss_total": 5.1, "loss_af": 4.8, "energy": 0.32, "i_ptm": 0.75, "ptm": 0.71, "plddt": 83.1},
    ...
  ]
}
```

Prediction scripts write per-sequence JSONs then merge into `predictions_af2.json` / `predictions_boltz.json`.

### SLURM job structure

`submit_protocols.sh` maps `$SLURM_ARRAY_TASK_ID` 0–11 to named protocols (e.g., `gen_energy_C`, `opt_anneal_energy_B`) via a case statement, sets energy weights, and calls the appropriate Python script.

`submit_predictions.sh` launches an array job (one task per sequence) then auto-submits `collect_predictions.py` with `--dependency=afterok`.

### Parsing logs

`parse_slurm_out.py` and `parse_multimer_out.py` extract per-step loss/metric tables from `.out` files for debugging and analysis.

## Key Files

| File | Role |
|---|---|
| `src/config.py` | Env var validation, hotspot residues (used by `generative_protocols/`) |
| `config.py` (repo root) | Near-duplicate config used only by `src/collect_af3_results.py` |
| `src/generative_protocols/model_energy_guidance.py` | MLP energy model + ColabDesign integration |
| `src/generative_protocols/generate_sequences.py` | De novo generation (pTM) |
| `src/generative_protocols/optimize_sequences.py` | Sequence optimization from inputs (pTM) |
| `src/generative_protocols/score_sequences.py` | AF2 single forward-pass scoring |
| `src/analysis/*.ipynb` | Notebooks ranking/comparing candidates across campaigns |
| `src/slurm_subs/submit_protocols.sh` | Main campaign launcher |
| `src/slurm_subs/run_predictions.sh` | Per-sequence AF2 + Boltz scoring |
| `src/slurm_subs/collect_predictions.py` | Merge per-sequence prediction JSONs |
| `data/input_sequences.txt` | Seed sequences for optimization runs |
| `data/energy_model_params/*.json` | Pre-trained MLP weights (exported from `PD_energy_model/`) |
| `PD_energy_model/` | Nested git repo: Julia training/analysis for the energy model |

## Notes

- Binder length is hardcoded to 15 AA; target is always chain A of 2ZNL.
- Boltz-1 scoring uses a YAML template written per-sequence; confidence parsed from the output JSON.
- AF3 results in `results/af3/` are pre-computed externally; not called by scripts here.
- Large results (`results/af3/`, `results/boltz/`) and `PD_energy_model/` are excluded from git — sync via rsync (see `../../SYNC.md`).
- `src/slurm_subs/*.sh` launchers were not yet updated for this reorganization: they still point `SCRIPT_DIR`/`GEN_SCRIPT`/`OPT_SCRIPT` at pre-move paths (e.g. `$CODEBASE/prg/PA_PB1/generate_sequences.py`). Fix these paths before relying on `sbatch` runs.
