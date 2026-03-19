#!/bin/bash
#SBATCH --job-name=autobrep_lora_train
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G                   # Total 32G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/slochmann/logs/autobrep_lora_train_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/autobrep_lora_train_%j.err

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/AutoBrep/checkpoints/furniture-lora-adapter"

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
DATASET_PATH="$SCRATCH/AutoBrep/data/parquet"
OUTPUT_DIR="$CKPT_DIR/furniture-lora-adapter"

SCRIPT_PATH="$HOME/semester_project_euler/euler_scripts/train_lora_ar.py"

if [ ! -f "$CKPT_DIR/ar.ckpt" ]; then
    echo "ERROR: Base checkpoints not found at $CKPT_DIR"
    exit 1
fi

cd "$AUTOBREP_DIR/core" || { echo "ERROR: Could not cd into $AUTOBREP_DIR/core"; exit 1; }

echo "Starting LoRA Fine-Tuning..."
"$PYTHON" "$SCRIPT_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 1e-4

echo "================================================================"
echo " LoRA Training complete!"
echo " Adapter saved to: $OUTPUT_DIR"
echo "================================================================"
