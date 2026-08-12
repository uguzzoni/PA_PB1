#!/bin/bash
#SBATCH --job-name=pa_pb1_protocols_2
#SBATCH --partition=default1_gpu
#SBATCH --output=pa_pb1_protocols_%A_%a.out
#SBATCH --error=pa_pb1_protocols_%A_%a.err
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=aar183,aar184,aar185,aar031,aar032,aar019
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=0-06:00:00
#SBATCH --array=4-11

# =============================================================================
# Resubmission of protocols 4-11 (opt_* group), which failed in the first run.
# Protocols 0-3 (gen_*) completed successfully.
#
#  Group B – optimize from input_sequences.txt (optimize_sequences.py)
#    4  opt_hard_noenergy    : stage 3 only,          no energy,  n=1/seq (deterministic)
#    5  opt_anneal_noenergy  : stage 2+3,             no energy,  n=4/seq (stochastic)
#    6  opt_hard_energy_A    : stage 3 only,          ew3=0.05,   n=1/seq
#    7  opt_hard_energy_B    : stage 3 only,          ew3=0.20,   n=1/seq
#    8  opt_hard_energy_C    : stage 3 only,          ew3=0.50,   n=1/seq
#    9  opt_anneal_energy_A  : stage 2+3, ew2=0.02,  ew3=0.05,   n=4/seq
#   10  opt_anneal_energy_B  : stage 2+3, ew2=0.02,  ew3=0.20,   n=4/seq
#   11  opt_anneal_energy_C  : stage 2+3, ew2=0.05,  ew3=0.50,   n=4/seq
# =============================================================================

PROTO_NAMES=(
    "gen_colabonly"
    "gen_energy_A"
    "gen_energy_B"
    "gen_energy_C"
    "opt_hard_noenergy"
    "opt_anneal_noenergy"
    "opt_hard_energy_A"
    "opt_hard_energy_B"
    "opt_hard_energy_C"
    "opt_anneal_energy_A"
    "opt_anneal_energy_B"
    "opt_anneal_energy_C"
)

PROTO=${PROTO_NAMES[$SLURM_ARRAY_TASK_ID]}

# ── Paths ─────────────────────────────────────────────────────────────────────
SHARED=/home/share_nfs/342-Projets_BGE/342.3-Gen-Chem/342.3.2-BioInfo/342.3.2.8-SeqModels
CODEBASE=${SHARED}/protein_design/colabdesign_energy_guidance
SCRIPT_DIR=${CODEBASE}/prg/PA_PB1

OPT_SCRIPT=${SCRIPT_DIR}/optimize_sequences.py
SEQ_FILE=${SCRIPT_DIR}/input_sequences.txt
OUTDIR=${SCRIPT_DIR}/results
mkdir -p ${OUTDIR}

# ── Environment ───────────────────────────────────────────────────────────────
source ${SHARED}/miniforge3/bin/activate colabdesign
source ${CODEBASE}/.env

CUDNN_LIB=${SHARED}/miniforge3/envs/colabdesign/lib/python3.10/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$CUDNN_LIB:$LD_LIBRARY_PATH
export PYTHONPATH=${CODEBASE}:$PYTHONPATH

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "Job ID       : $SLURM_JOB_ID"
echo "Array task   : $SLURM_ARRAY_TASK_ID  →  protocol: $PROTO"
echo "Node         : $SLURMD_NODENAME"
echo "Start time   : $(date)"
echo "---------------------------------------------"
nvidia-smi | head -15
echo "---------------------------------------------"

# ── GPU check ─────────────────────────────────────────────────────────────────
python3 - << 'PYEOF'
import jax
devices = jax.devices()
print(f"JAX devices: {devices}")
if not any(d.platform == "gpu" for d in devices):
    raise SystemExit("WARNING: No GPU found — aborting.")
print(f"GPU check passed: {devices[0]}")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: GPU check failed. Exiting."
    exit 1
fi

# ── Run protocol ──────────────────────────────────────────────────────────────
cd ${SCRIPT_DIR}
OUTPUT=${OUTDIR}/${PROTO}.json
echo "Output file  : $OUTPUT"
echo "Running protocol: $PROTO ..."

case "$PROTO" in

    # Hard-only (no annealing): deterministic → 1 seed/seq × 5 seqs = 5 results
    opt_hard_noenergy)
        python3 $OPT_SCRIPT --seq_file $SEQ_FILE \
            --iters_2 0 --ew_3 0.00 -n 1 \
            -o $OUTPUT
        ;;

    # Annealing+hard: stochastic (dropout) → 4 seeds/seq × 5 seqs = 20 results
    opt_anneal_noenergy)
        python3 $OPT_SCRIPT --seq_file $SEQ_FILE \
            --ew_2 0.00 --ew_3 0.00 -n 4 \
            -o $OUTPUT
        ;;

    opt_hard_energy_A)
        python3 $OPT_SCRIPT --seq_file $SEQ_FILE \
            --iters_2 0 --ew_3 0.05 -n 1 \
            -o $OUTPUT
        ;;

    opt_hard_energy_B)
        python3 $OPT_SCRIPT --seq_file $SEQ_FILE \
            --iters_2 0 --ew_3 0.20 -n 1 \
            -o $OUTPUT
        ;;

    opt_hard_energy_C)
        python3 $OPT_SCRIPT --seq_file $SEQ_FILE \
            --iters_2 0 --ew_3 0.50 -n 1 \
            -o $OUTPUT
        ;;

    opt_anneal_energy_A)
        python3 $OPT_SCRIPT --seq_file $SEQ_FILE \
            --ew_2 0.02 --ew_3 0.05 -n 4 \
            -o $OUTPUT
        ;;

    opt_anneal_energy_B)
        python3 $OPT_SCRIPT --seq_file $SEQ_FILE \
            --ew_2 0.02 --ew_3 0.20 -n 4 \
            -o $OUTPUT
        ;;

    opt_anneal_energy_C)
        python3 $OPT_SCRIPT --seq_file $SEQ_FILE \
            --ew_2 0.05 --ew_3 0.50 -n 4 \
            -o $OUTPUT
        ;;

    *)
        echo "ERROR: unknown protocol '$PROTO'"
        exit 1
        ;;
esac

echo "---------------------------------------------"
echo "End time : $(date)"
