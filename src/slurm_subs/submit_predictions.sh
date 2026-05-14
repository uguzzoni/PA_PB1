#!/bin/bash
# =============================================================================
# submit_predictions.sh
#
# Given a file of sequences, submits:
#   1. A SLURM array job (one task per sequence) that runs both:
#        - AF2 scoring  (colabdesign forward pass → i_ptm, ptm, plddt, energy)
#        - Boltz prediction                       → iptm, ptm, confidence_score,
#                                                    complex_plddt, complex_iplddt
#      Each task writes  <pred_dir>/seq_N_af2.json  and  seq_N_boltz.json
#
#   2. A collect job (dependency: afterok on the array) that merges all
#      per-sequence JSONs into two final files:
#        <pred_dir>/predictions_af2.json
#        <pred_dir>/predictions_boltz.json
#
# Usage:
#   bash submit_predictions.sh <seq_file> [pred_dir]
#
# Examples:
#   bash submit_predictions.sh ../input_sequences.txt
#   bash submit_predictions.sh top_candidates.txt results/predictions
# =============================================================================

set -euo pipefail

SEQ_FILE="${1:-}"
if [[ -z "${SEQ_FILE}" ]]; then
    echo "Usage: bash submit_predictions.sh <seq_file> [pred_dir]"
    exit 1
fi
if [[ ! -f "${SEQ_FILE}" ]]; then
    echo "ERROR: sequence file not found: ${SEQ_FILE}"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED=/home/share_nfs/342-Projets_BGE/342.3-Gen-Chem/342.3.2-BioInfo/342.3.2.8-SeqModels
CODEBASE=${SHARED}/protein_design/colabdesign_energy_guidance

PRED_DIR="${2:-${SLURM_DIR}/predictions}"
mkdir -p "${PRED_DIR}"

# Resolve to absolute paths (SLURM workers need them)
SEQ_FILE="$(realpath "${SEQ_FILE}")"
PRED_DIR="$(realpath "${PRED_DIR}")"

RUN_SCRIPT="${SLURM_DIR}/run_predictions.sh"
COLLECT_SCRIPT="${SLURM_DIR}/collect_predictions.py"

# ── Count sequences ───────────────────────────────────────────────────────────
N=$(grep -cv '^#' "${SEQ_FILE}" || true)
N=$(grep -c '[^[:space:]]' <(grep -v '^#' "${SEQ_FILE}") || true)
if [[ "${N}" -eq 0 ]]; then
    echo "ERROR: no sequences found in ${SEQ_FILE}"
    exit 1
fi
echo "Sequence file : ${SEQ_FILE}"
echo "Sequences     : ${N}"
echo "Output dir    : ${PRED_DIR}"
echo ""

# ── Submit prediction array job ───────────────────────────────────────────────
ARRAY_JOB_ID=$(sbatch \
    --array=0-$((N - 1)) \
    --parsable \
    "${RUN_SCRIPT}" "${SEQ_FILE}" "${PRED_DIR}")

echo "Submitted prediction array : job ${ARRAY_JOB_ID}  (${N} tasks)"
echo "  Logs: ${SLURM_DIR}/pa_pb1_predictions_${ARRAY_JOB_ID}_*.out"

# ── Submit collect job (runs after ALL array tasks succeed) ───────────────────
COLLECT_JOB_ID=$(sbatch \
    --job-name=pa_pb1_collect \
    --partition=default1_gpu \
    --cpus-per-task=2 \
    --mem=4G \
    --time=0-00:30:00 \
    --dependency=afterok:"${ARRAY_JOB_ID}" \
    --output="${SLURM_DIR}/pa_pb1_collect_%j.out" \
    --error="${SLURM_DIR}/pa_pb1_collect_%j.err" \
    --parsable \
    --wrap="source ${SHARED}/miniforge3/bin/activate colabdesign && \
            python3 ${COLLECT_SCRIPT} \
                --pred_dir ${PRED_DIR} \
                --source_file ${SEQ_FILE} \
                --af2_out ${PRED_DIR}/predictions_af2.json \
                --boltz_out ${PRED_DIR}/predictions_boltz.json")

echo "Submitted collect job      : job ${COLLECT_JOB_ID}  (runs after array completes)"
echo ""
echo "Final outputs (after collect job):"
echo "  ${PRED_DIR}/predictions_af2.json"
echo "  ${PRED_DIR}/predictions_boltz.json"
