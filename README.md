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
├── data/
│   ├── pdbs/                        # Input PDB/CIF structures (~1.8 MB)
│   │   ├── 2ZNL.pdb / .cif          # PA–PB1 complex
│   │   ├── 2ZNL_chainA.pdb          # PA chain only
│   │   ├── 7QVM.pdb                 # Additional reference structure
│   │   └── 7QVM_interface_seqs.txt  # Interface residue sequences
│   ├── energy_model_params/         # Trained energy model weights (PhageNegBinom)
│   │   └── PNB_2R_3lay_negbinom_energy_model_weights.json
│   └── input_sequences.txt          # Seed/wild-type sequences for design
│
├── notebooks/
│   ├── analyze_results.ipynb        # Main analysis of design campaigns
│   ├── analyze_results_1.ipynb      # Extended analysis (run 1)
│   ├── analyze_results_wide.ipynb   # Analysis with wider sequence space
│   └── compare_predictions_models.ipynb  # Comparison across scoring models
│
└── results/                         # Generated sequences and structure predictions
    ├── colabdesign/                 # ColabDesign hallucination runs (~232 KB)
    │   ├── run1_protocols/
    │   ├── run2_wide/
    │   ├── run_multimer/
    │   ├── top_candidates.csv
    │   └── top_candidates_wide.csv
    ├── af3/                         # AlphaFold3 structure predictions (~1.7 GB)
    └── boltz/                       # Boltz-1 structure predictions (~779 MB)
```

## Design Workflow

1. **Sequence generation** — ColabDesign (AF2-based hallucination) in `methods/colabdesign_fork/`
2. **Structural scoring** — AlphaFold3 (`results/af3/`) and Boltz-1 (`results/boltz/`) predictions
3. **Sequence scoring** — PhageNegBinom energy model (`data/energy_model_params/`)
4. **Analysis** — notebooks rank candidates by predicted binding affinity and structural quality

## Git & Sync Notes

- **Tracked in git**: `notebooks/`, `data/pdbs/`, `data/energy_model_params/`, `data/input_sequences.txt`, CSVs in `results/colabdesign/`
- **Excluded from git** (too large): `results/af3/` (~1.7 GB), `results/boltz/` (~779 MB)
- Large results synced via `rsync` — see [../../SYNC.md](../../SYNC.md)
