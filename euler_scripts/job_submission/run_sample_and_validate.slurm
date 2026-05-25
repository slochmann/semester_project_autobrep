#!/bin/bash
#SBATCH --job-name=sample_and_validate
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G                   # Total 32G (ar.ckpt is ~12GB)
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/slochmann/logs/sample_and_validate_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/sample_and_validate_%j.err

#==============================================================================
# CONFIGURATION
#==============================================================================
SAMPLE_BATCH_SIZE=4
TEMPERATURE=0.7
TOP_P=0.9
COMPLEXITY=17                              # 14=easy, 15=medium, 16=hard, 17=random
PREFIX_RATIO=0.5
HF_SPLIT="validation"
HF_SAMPLE_SKIP=0
SEED=42
VERTEX_THRESHOLD=0.002
SEWING_TOLERANCE=0.002
Z_THRESHOLD=0.002
#==============================================================================

mkdir -p "$SCRATCH/logs"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.8.0
module load eth_proxy

#------------------------------------------------------------------------------
# Conda / Python
#------------------------------------------------------------------------------
CONDA_DIR="$HOME/miniconda3"
PYTHON="$CONDA_DIR/envs/autobrep/bin/python"
unset PYTHONPATH

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------
AUTOBREP_DIR="$SCRATCH/AutoBrep"
CKPT_DIR="$SCRATCH/AutoBrep/checkpoints"
OUTPUT_DIR="$SCRATCH/AutoBrep/samples/sample_and_validate"

SCRIPT_PATH="$HOME/semester_project_euler/euler_scripts/sample_and_validate_ar.py"

if [ ! -f "$CKPT_DIR/ar.ckpt" ]; then
    echo "ERROR: Base checkpoints not found at $CKPT_DIR"
    echo "       Run download_autobrep_ckpt.slurm first."
    exit 1
fi

cd "$AUTOBREP_DIR/core" || { echo "ERROR: Could not cd into $AUTOBREP_DIR/core"; exit 1; }

echo "Starting sample + validate run..."
"$PYTHON" "$SCRIPT_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --hf_split "$HF_SPLIT" \
    --hf_sample_skip $HF_SAMPLE_SKIP \
    --sample_batch_size $SAMPLE_BATCH_SIZE \
    --sample_temperature $TEMPERATURE \
    --sample_threshold $TOP_P \
    --sample_complexity $COMPLEXITY \
    --prefix_ratio $PREFIX_RATIO \
    --z_threshold $Z_THRESHOLD \
    --vertex_threshold $VERTEX_THRESHOLD \
    --sewing_tolerance $SEWING_TOLERANCE \
    --seed $SEED

echo "================================================================"
echo " sample + validate complete!"
echo " Results in: $OUTPUT_DIR"
echo "================================================================"
