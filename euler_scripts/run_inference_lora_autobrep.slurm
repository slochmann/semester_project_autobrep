#!/bin/bash
#SBATCH --job-name=100_cylinders
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G                   # Total 32G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/slochmann/logs/autobrep_lora_inference_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/autobrep_lora_inference_%j.err

NAME="${SLURM_JOB_NAME}"

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/AutoBrep/lora_samples"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.8.0
module load eth_proxy

#------------------------------------------------------------------------------
# Set up Conda environment Python
#------------------------------------------------------------------------------
CONDA_DIR="$HOME/miniconda3"
PYTHON="$CONDA_DIR/envs/autobrep/bin/python"

# Clear PYTHONPATH to prevent system shadowing
unset PYTHONPATH

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------
AUTOBREP_DIR="$SCRATCH/AutoBrep"
CKPT_DIR="$SCRATCH/AutoBrep/checkpoints"
OUTPUT_DIR="$SCRATCH/AutoBrep/lora_samples/$NAME"

# Script and adapter path definitions
# By default we map the repo to scratch on Euler; if running this from the repository directly mapped to Euler:
LORA_ADAPTER_PATH="$CKPT_DIR/$NAME"
SCRIPT_PATH="$HOME/semester_project_euler/euler_scripts/sample_lora_ar.py"

# Verify checkpoints exist
if [ ! -f "$CKPT_DIR/ar.ckpt" ]; then
    echo "ERROR: Base checkpoints not found at $CKPT_DIR"
    exit 1
fi

if [ ! -d "$LORA_ADAPTER_PATH" ]; then
    echo "ERROR: LoRA adapter not found at $LORA_ADAPTER_PATH"
    echo "Please ensure the adapter was copied to this location."
    exit 1
fi

#------------------------------------------------------------------------------
# Run inference from core/ (required for autobrep package import)
#------------------------------------------------------------------------------
cd "$AUTOBREP_DIR/core" || { echo "ERROR: Could not cd into $AUTOBREP_DIR/core"; exit 1; }

echo "Starting LoRA Inference Headless Generation..."
"$PYTHON" "$SCRIPT_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --lora_adapter_path "$LORA_ADAPTER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 200 \
    --temperature 1.0 \
    --threshold 0.95 \
    --complexity 14 \
    --batch_size 10 \
    --seed 42
# complexity: 14=easy, 15=medium, 16=hard, 17=random/uncond
# Lower temperature (e.g. 0.8) — reduces randomness, makes the model follow the grammar more strictly.
#                               But an overfitted model already has extremely peaked logits, reducing temperature further concentrates probability on degenerate tokens
# Lower top-p threshold (e.g. 0.90 instead of 0.95) — cuts off low-probability tokens that lead to grammar violations.
echo "================================================================"
echo " LoRA Inference complete!"
echo " Samples saved to: $OUTPUT_DIR"
echo "================================================================"
