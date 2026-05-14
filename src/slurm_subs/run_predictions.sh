#!/bin/bash
#SBATCH --job-name=pa_pb1_predictions
#SBATCH --partition=default1_gpu
#SBATCH --output=pa_pb1_predictions_%A_%a.out
#SBATCH --error=pa_pb1_predictions_%A_%a.err
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=aar183,aar184,aar185,aar031,aar032,aar019
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-03:00:00
# NOTE: --array is set dynamically by submit_predictions.sh; no default here.

# =============================================================================
# run_predictions.sh  — SLURM array worker
#
# Each array task scores ONE sequence with:
#   1. AF2 (colabdesign, model_3/4/5, single forward pass)     → seq_N_af2.json
#   2. Boltz (structure prediction via yaml input)              → seq_N_boltz.json
#
# Arguments (passed by submit_predictions.sh):
#   $1  SEQ_FILE   : text file with one sequence per line (# = comment)
#   $2  PRED_DIR   : directory where per-sequence JSONs are written
#
# The collect_predictions.py script merges all per-seq JSONs into two final
# output files: predictions_af2.json and predictions_boltz.json
# =============================================================================

set -euo pipefail

SEQ_FILE="$1"
PRED_DIR="$2"
mkdir -p "${PRED_DIR}"

# ── Paths ─────────────────────────────────────────────────────────────────────
SHARED=/home/share_nfs/342-Projets_BGE/342.3-Gen-Chem/342.3.2-BioInfo/342.3.2.8-SeqModels
CODEBASE=${SHARED}/protein_design/colabdesign_energy_guidance
SCRIPT_DIR=${CODEBASE}/prg/PA_PB1

AF2_SCRIPT=${SCRIPT_DIR}/score_sequences.py

BOLTZ_BASE=${SHARED}/protein_design/boltz
BOLTZ_TEMPLATE=${BOLTZ_BASE}/inputs/pa_pb1_wt.yaml   # chain A template (PA + MSA + CIF)
BOLTZ_OUTDIR=${BOLTZ_BASE}/results
BOLTZ_YAML_DIR=${BOLTZ_BASE}/inputs/tmp_yamls
mkdir -p "${BOLTZ_YAML_DIR}"

# Colabdesign environment (AF2 scoring)
COLABDESIGN_ENV=colabdesign
# Boltz environment  (set same as colabdesign if boltz is installed there)
BOLTZ_ENV=boltz

# ── Read sequence for this task ───────────────────────────────────────────────
mapfile -t SEQS < <(grep -v '^#' "${SEQ_FILE}" | awk 'NF {print $1}')
SEQ="${SEQS[$SLURM_ARRAY_TASK_ID]}"
IDX="${SLURM_ARRAY_TASK_ID}"
NAME="seq_${IDX}"

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Array task : $SLURM_ARRAY_TASK_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "Sequence   : $SEQ  (idx=${IDX})"
echo "---------------------------------------------"
nvidia-smi | head -10
echo "---------------------------------------------"

# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — AF2 scoring  (colabdesign / JAX)
# ═════════════════════════════════════════════════════════════════════════════
AF2_OUT="${PRED_DIR}/${NAME}_af2.json"

echo "[AF2] Starting ..."
source "${SHARED}/miniforge3/bin/activate" "${COLABDESIGN_ENV}"
source "${CODEBASE}/.env"

CUDNN_LIB="${SHARED}/miniforge3/envs/${COLABDESIGN_ENV}/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${CODEBASE}:${PYTHONPATH:-}"

# GPU check
python3 - << 'PYEOF'
import jax
devices = jax.devices()
if not any(d.platform == "gpu" for d in devices):
    raise SystemExit("No GPU found — aborting AF2 step.")
print(f"GPU OK: {devices[0]}")
PYEOF

cd "${SCRIPT_DIR}"
python3 "${AF2_SCRIPT}" --seq "${SEQ}" -o "${AF2_OUT}"
echo "[AF2] Done → ${AF2_OUT}"
conda deactivate

# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — Boltz prediction
# ═════════════════════════════════════════════════════════════════════════════
BOLTZ_OUT="${PRED_DIR}/${NAME}_boltz.json"
YAML_PATH="${BOLTZ_YAML_DIR}/pa_${NAME}.yaml"

echo "[Boltz] Creating YAML → ${YAML_PATH} ..."

# Build per-sequence YAML by reading chain A from the wt template and
# inserting the binder sequence as chain B.
python3 - "${BOLTZ_TEMPLATE}" "${SEQ}" "${YAML_PATH}" << 'PYEOF'
import sys

template_path = sys.argv[1]
binder_seq    = sys.argv[2]
out_path      = sys.argv[3]

# Read the wt template and replace the last 'sequence:' entry (chain B).
with open(template_path) as f:
    lines = f.readlines()

# Find line containing "id: B" and the following "sequence:" line to replace.
in_chain_b = False
new_lines = []
for line in lines:
    if "id: B" in line:
        in_chain_b = True
    if in_chain_b and line.strip().startswith("sequence:"):
        line = line[:line.index("sequence:")] + f"sequence: {binder_seq}\n"
        in_chain_b = False   # only replace once
    new_lines.append(line)

with open(out_path, "w") as f:
    f.writelines(new_lines)
print(f"YAML written: {out_path}")
PYEOF

echo "[Boltz] Running boltz predict ..."
source "${SHARED}/miniforge3/bin/activate" "${BOLTZ_ENV}"

boltz predict "${YAML_PATH}" \
    --output_path "${BOLTZ_OUTDIR}" \
    --use_msa_server False \
    --recycling_steps 3 \
    --sampling_steps 200

conda deactivate

# Parse confidence JSON and write per-sequence boltz result
CONF_JSON=$(find "${BOLTZ_OUTDIR}/boltz_results_pa_${NAME}" \
                 -name "confidence_*.json" 2>/dev/null | head -1)

if [[ -z "${CONF_JSON}" ]]; then
    echo "[Boltz] WARNING: confidence JSON not found, writing empty result."
    python3 -c "
import json, sys
result = {'idx': ${IDX}, 'seq': '${SEQ}', 'error': 'boltz output not found'}
with open('${BOLTZ_OUT}', 'w') as f:
    json.dump(result, f, indent=2)
"
else
    python3 - "${CONF_JSON}" "${SEQ}" "${IDX}" "${BOLTZ_OUT}" << 'PYEOF'
import json, sys
conf_path = sys.argv[1]
seq       = sys.argv[2]
idx       = int(sys.argv[3])
out_path  = sys.argv[4]

with open(conf_path) as f:
    d = json.load(f)

result = {
    "idx":              idx,
    "seq":              seq,
    "iptm":             round(d.get("iptm",             0.0), 6),
    "ptm":              round(d.get("ptm",              0.0), 6),
    "confidence_score": round(d.get("confidence_score", 0.0), 6),
    "complex_plddt":    round(d.get("complex_plddt",    0.0), 6),
    "complex_iplddt":   round(d.get("complex_iplddt",   0.0), 6),
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Boltz result saved: {out_path}")
PYEOF
fi

echo "[Boltz] Done → ${BOLTZ_OUT}"

# ── Final ─────────────────────────────────────────────────────────────────────
echo "---------------------------------------------"
echo "End time : $(date)"
