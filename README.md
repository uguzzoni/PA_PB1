# PA_PB1 — Binder Design at the Influenza PA–PB1 Interface

Design campaign targeting the interaction between the influenza polymerase subunits
**PA** (endonuclease/C-terminal domain) and **PB1** (N-terminal domain). The goal is
to design peptides or mini-proteins that mimic or disrupt the PA–PB1 interface.

## Biological Target

- **PA**: PA C-terminal domain of influenza RNA-dependent RNA polymerase
- **PB1**: PB1 N-terminal fragment that binds the PA C-terminal groove
- Key structures: `2ZNL` (PA–PB1 complex), `7QVM` (used for interface residue analysis)

## Folder Structure

```
PA_PB1/
├── PD_energy_model/                 # Energy model (nested git repo, ignored by this repo's git)
│   ├── training/                    # Julia training scripts (training_2rounds/, training_3rounds/)
│   ├── analysis/                    # Julia/notebook analysis of the trained model
│   ├── model_params/                # Trained weights (Julia-side source of data/energy_model_params/)
│   └── data/                        # Affinity measurements & sequencing counts used to train the model
│
├── src/
│   ├── config.py                    # Env var validation, hotspot residues (used by generative_protocols/)
│   ├── check_gen_seq.py             # Sanity-checks generated sequences against the design constraints
│   ├── collect_af3_results.py       # Aggregates AlphaFold3 outputs into results/af3/af3_summary.json
│   │
│   ├── generative_protocols/        # Sequence generation: AF2 (ColabDesign) + energy-guided protocols
│   │   ├── generate_sequences.py            # De novo generation (pTM), 4-stage protocol
│   │   ├── generate_sequences_2enw.py       # Variant targeting 2ENW-derived template
│   │   ├── generate_sequences_baseline.py   # ColabDesign-only baseline (no energy term)
│   │   ├── generate_sequences_multimer.py   # AF2-Multimer version of generate
│   │   ├── optimize_sequences.py            # Sequence optimization from seed inputs (pTM), 2-stage
│   │   ├── optimize_sequence.py             # Single-sequence optimization variant
│   │   ├── optimize_sequences_multimer.py   # AF2-Multimer version of optimize
│   │   ├── model_energy_guidance.py         # MLP energy term wrapped around ColabDesign's AfDesign
│   │   ├── scan_energy_weight.py            # Sweeps energy_weight to characterize its effect
│   │   ├── score_sequences.py               # AF2 single forward-pass scoring
│   │   └── score_custom_sequences.py        # Scores an arbitrary list of user-supplied sequences
│   │
│   ├── analysis/                     # Notebooks analyzing generated sequences & comparing runs
│   │   ├── analyze_results.ipynb            # Main analysis of a design campaign
│   │   ├── analyze_results_1.ipynb          # Extended analysis (run 1)
│   │   ├── analyze_results_2.ipynb          # Extended analysis (run 2)
│   │   ├── analyze_results_wide.ipynb       # Analysis of the wide-search campaign
│   │   ├── candidates_report.ipynb          # Candidate shortlist report
│   │   ├── compare_predictions_models.ipynb # AF2 vs Boltz-1 vs AF3 comparison
│   │   ├── summary_analysis_results.ipynb   # Cross-campaign summary
│   │   └── results/                         # CSV exports from the notebooks (candidate tables)
│   │
│   └── slurm_subs/                   # SLURM launchers, per-sequence prediction runner, log parsers
│       ├── submit_protocols.sh / submit_protocols_2.sh
│       ├── submit_wide_search.sh
│       ├── submit_multimer_search.sh / submit_multimer_search_2.sh
│       ├── submit_predictions.sh / run_predictions.sh
│       ├── collect_predictions.py
│       └── parse_slurm_out.py / parse_multimer_out.py
│
├── data/                             # Inputs: structures, seeds, experimental measurements
│   ├── pdbs/                         # Input PDB/CIF structures
│   │   ├── 2ZNL.pdb / .cif / 2ZNL_chainA.pdb / 2ZNL_chainA_400plus.pdb
│   │   ├── 7QVM.pdb                  # Additional reference structure
│   │   └── 7QVM_interface_seqs.txt   # Interface residue sequences
│   ├── energy_model_params/          # Trained energy model weights (PhageNegBinom), copied from PD_energy_model/
│   ├── input_sequences.txt           # Seed/wild-type sequences for optimize_sequences.py
│   ├── seqs_aff_alberto_darren*.txt  # Sequences with experimentally measured affinity
│   └── affinity_measurements_PA-PB1_formatted.csv
│
└── results/                           # Generated sequences and structure predictions
    ├── colabdesign/                   # Generation/optimization run outputs (JSON) + candidate CSVs
    │   ├── run1_protocols/ run2_wide/ run_multimer/ run_multimer2/ custom/
    │   └── top_candidates.csv, top_candidates_wide.csv
    ├── af3/                            # AlphaFold3 structure predictions (git-ignored, ~GB scale)
    └── boltz/                          # Boltz-1 structure predictions (git-ignored, ~GB scale)
```

## Design Workflow

1. **Energy model** — trained in `PD_energy_model/` (Julia); exported weights land in `data/energy_model_params/`
2. **Sequence generation** — `src/generative_protocols/` runs AF2-based hallucination (ColabDesign) guided by the energy model
3. **Structural scoring** — AlphaFold3 (`results/af3/`) and Boltz-1 (`results/boltz/`) predictions
4. **Analysis** — `src/analysis/` notebooks rank candidates by predicted binding affinity and structural quality

## Git & Sync Notes

- **Nested git repo**: `PD_energy_model/` is its own git repository and is entirely git-ignored from this repo (see `.gitignore`). Manage it independently.
- **Tracked in git**: `src/`, `data/pdbs/`, `data/energy_model_params/`, `data/input_sequences.txt`, CSVs in `results/colabdesign/`
- **Excluded from git** (too large): `results/af3/`, `results/boltz/`, `data/to_share/`, `PD_energy_model/`
- Large results synced via `rsync` — see [../../SYNC.md](../../SYNC.md)
