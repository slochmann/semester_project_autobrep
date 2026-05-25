#!/bin/bash
#SBATCH --job-name=600_cylinders
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G                   # Total 32G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/slochmann/logs/autobrep_lora_train_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/autobrep_lora_train_%j.err

#------------------------------------------------------------------------------
# Hyperparameter Configuration (edit here to change training params)
#------------------------------------------------------------------------------
BATCH_SIZE=4
NUM_EPOCHS=10
LEARNING_RATE="1e-5"
SAMPLE_TEMPERATURE=0.6
SAMPLE_COMPLEXITY=17
LORA_R=32
LORA_ALPHA=128
VAL_INTERVAL_BATCHES=33
Z_THRESHOLD=0
VERTEX_THRESHOLD=0.002
SEWING_TOLERANCE=0.002

#------------------------------------------------------------------------------
# Generate unique checkpoint name with hyperparameters and timestamp
#------------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="${SLURM_JOB_NAME}"
RUNNAME="${SLURM_JOB_NAME}_T${SAMPLE_TEMPERATURE}_comp${SAMPLE_COMPLEXITY}_lr${LEARNING_RATE}_e${NUM_EPOCHS}_r${LORA_R}_a${LORA_ALPHA}_${TIMESTAMP}"

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/AutoBrep/checkpoints/$RUNNAME"

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
# Paths
#------------------------------------------------------------------------------
AUTOBREP_DIR="$SCRATCH/AutoBrep"
CKPT_DIR="$SCRATCH/AutoBrep/checkpoints"
DATASET_PATH="$SCRATCH/AutoBrep/data/parquet/$NAME"
OUTPUT_DIR="$CKPT_DIR/$RUNNAME"

SCRIPT_PATH="$HOME/semester_project_euler/euler_scripts/train_lora_ar.py"

if [ ! -f "$CKPT_DIR/ar.ckpt" ]; then
    echo "ERROR: Base checkpoints not found at $CKPT_DIR"
    exit 1
fi

cd "$AUTOBREP_DIR/core" || { echo "ERROR: Could not cd into $AUTOBREP_DIR/core"; exit 1; }

echo "Starting LoRA Fine-Tuning..."
echo "Configuration:"
echo "  Learning Rate: $LEARNING_RATE"
echo "  LoRA R: $LORA_R, Alpha: $LORA_ALPHA"
echo "  Temperature: $SAMPLE_TEMPERATURE, Complexity: $SAMPLE_COMPLEXITY"
echo "  Checkpoint: $OUTPUT_DIR"
echo ""

"$PYTHON" "$SCRIPT_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --sample_temperature $SAMPLE_TEMPERATURE \
    --sample_complexity $SAMPLE_COMPLEXITY \
    --sample_interval_epochs 1 \
    --sample_only_last_n_epochs 4 \
    --sample_interval_batches 0 \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --wandb_project "lora-autobrep" \
    --wandb_entity "slochmann-ethz" \
    --val_interval_batches $VAL_INTERVAL_BATCHES \
    --z_threshold $Z_THRESHOLD \
    --vertex_threshold $VERTEX_THRESHOLD \
    --sewing_tolerance $SEWING_TOLERANCE
# complexity: 14=easy, 15=medium, 16=hard, 17=random/uncond
echo "================================================================"
echo " LoRA Training complete!"
echo " Adapter saved to: $OUTPUT_DIR"
echo "================================================================"
