#!/bin/bash
#SBATCH --job-name=autobrep_lora_inference
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G                   # Total 32G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/slochmann/logs/autobrep_lora_inference_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/autobrep_lora_inference_%j.err

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
OUTPUT_DIR="$SCRATCH/AutoBrep/lora_samples"

# Script and adapter path definitions
# By default we map the repo to scratch on Euler; if running this from the repository directly mapped to Euler:
LORA_ADAPTER_PATH="$CKPT_DIR/furniture-lora-adapter"
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
    --num_samples 10 \
    --temperature 1.0 \
    --threshold 0.95 \
    --complexity 16 \
    --batch_size 10
# complexity: 14=easy, 15=medium, 16=hard, 17=random/uncond
echo "================================================================"
echo " LoRA Inference complete!"
echo " Samples saved to: $OUTPUT_DIR"
echo "================================================================"
