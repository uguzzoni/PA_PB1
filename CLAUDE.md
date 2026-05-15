# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computational binder design campaign targeting the influenza PA–PB1 polymerase interface. The workflow combines AF2-based sequence hallucination (ColabDesign) with ML energy guidance (PhageNegBinom MLP), followed by structural scoring with Boltz-1 and AlphaFold3.

## Environment Setup

Scripts require these environment variables (set via `src/slurm_subs/.env`):

```bash
export COLABDESIGN_PDB=/path/to/2ZNL_chainA_400plus.pdb
export COLABDESIGN_PARAMS_DIR=/path/to/af2_params/
export COLABDESIGN_ENERGY_WEIGHTS_PATH=/path/to/PNB_2R_3lay_negbinom_energy_model_weights.json
```

`config.py` validates these at import time via `_require()` and exposes `PDB_PATH`, `PARAMS_DIR`, `ENERGY_WEIGHTS_PATH`, and the hardcoded list of 27 PA hotspot residues used in `prep_inputs()`.

SLURM scripts source `.env` and set `PYTHONPATH` to the colabdesign fork before calling Python.

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
source src/slurm_subs/.env
python src/generate_sequences.py -n 5 --ew_1a 0.02 --ew_1b 0.05 --ew_2 0.02 --ew_3 0.05
python src/optimize_sequences.py --seq_file data/input_sequences.txt -n 4 --ew_2 0.05 --ew_3 0.20
python src/score_sequences.py --seq ACDEFGHIKLMNPQR
```

## Code Architecture

### Design workflow (two modes)

**Generate** (`generate_sequences.py`): 4-stage protocol, starts from random logits.
1. Stage 1a (50 iter): soft, dropout, T=1.0, energy weight `ew_1a`
2. Stage 1b (50 iter): soft, dropout, T=1.0, energy weight `ew_1b`
3. Stage 2 (50 iter): soft, dropout, T anneals 1→0, weight `ew_2`
4. Stage 3 (30 iter): hard (discrete), no dropout, T≈0, weight `ew_3`

**Optimize** (`optimize_sequences.py`): 2-stage protocol, starts from input sequences.
- Stage 2 (annealing) is optional — skip with `--iters_2 0` for deterministic hard-only mode.
- `-n` seeds control stochasticity; stage 3 alone is fully deterministic.

Multimer variants (`generate_sequences_multimer.py`, `optimize_sequences_multimer.py`) use AF2-Multimer instead of pTM models.

### Energy guidance

`model_energy_guidance.py` wraps ColabDesign's `AfDesign` and injects a 3-layer MLP energy term into the loss. The MLP operates on soft logits (sequence probability distributions), outputting a scalar energy. The weight balances AF2 structural confidence vs. predicted binding affinity:

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
| `src/config.py` | Env var validation, hotspot residues |
| `src/model_energy_guidance.py` | MLP energy model + ColabDesign integration |
| `src/generate_sequences.py` | De novo generation (pTM) |
| `src/optimize_sequences.py` | Sequence optimization from inputs (pTM) |
| `src/score_sequences.py` | AF2 single forward-pass scoring |
| `src/slurm_subs/submit_protocols.sh` | Main campaign launcher |
| `src/slurm_subs/run_predictions.sh` | Per-sequence AF2 + Boltz scoring |
| `src/slurm_subs/collect_predictions.py` | Merge per-sequence prediction JSONs |
| `data/input_sequences.txt` | Seed sequences for optimization runs |
| `data/energy_model_params/*.json` | Pre-trained MLP weights |

## Notes

- Binder length is hardcoded to 15 AA; target is always chain A of 2ZNL.
- Boltz-1 scoring uses a YAML template written per-sequence; confidence parsed from the output JSON.
- AF3 results in `results/af3/` are pre-computed externally; not called by scripts here.
- Large results (`results/af3/`, `results/boltz/`) are excluded from git — sync via rsync (see `../../SYNC.md`).
