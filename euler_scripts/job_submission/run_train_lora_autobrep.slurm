#!/bin/bash
#SBATCH --job-name=600_cylinders
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G                   # Total 32G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/slochmann/logs/autobrep_lora_train_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/autobrep_lora_train_%j.err

# Use SLURM job name; override with sbatch --job-name=autobrep_lora_train_<descriptor>
NAME="${SLURM_JOB_NAME}"
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/AutoBrep/checkpoints/$NAME"

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
# Set PyTorch environment variables
#------------------------------------------------------------------------------
export PYTORCH_ALLOC_CONF=expandable_segments:True

#------------------------------------------------------------------------------
# Set WANDB environment variables
#------------------------------------------------------------------------------
export WANDB_API_KEY="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"
export WANDB_PROJECT="autobrep"
export WANDB_ENTITY="slochmann-ethz"

#------------------------------------------------------------------------------
# Set Hugging Face environment variables
#------------------------------------------------------------------------------
export HF_TOKEN="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"

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------
AUTOBREP_DIR="$SCRATCH/AutoBrep"
CKPT_DIR="$SCRATCH/AutoBrep/checkpoints"
DATASET_PATH="$SCRATCH/AutoBrep/data/parquet/$NAME"
OUTPUT_DIR="$CKPT_DIR/$NAME"

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
    --num_epochs 2 \
    --learning_rate 5e-5 \
    --sample_temperature 1 \
    --sample_complexity 17 \
    --sample_interval_epochs 1 \
    --sample_interval_batches 0 \
    --lora_r 32 \
    --lora_alpha 128 \
    --wandb_project "lora-autobrep" \
    --wandb_entity "slochmann-ethz"
# complexity: 14=easy, 15=medium, 16=hard, 17=random/uncond
echo "================================================================"
echo " LoRA Training complete!"
echo " Adapter saved to: $OUTPUT_DIR"
echo "================================================================"
