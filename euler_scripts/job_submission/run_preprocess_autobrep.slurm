#!/bin/bash
#SBATCH --job-name=1_pipe
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G                   # Total 32G
#SBATCH --output=/cluster/scratch/slochmann/logs/autobrep_preprocess_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/autobrep_preprocess_%j.err

mkdir -p "$SCRATCH/logs"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0

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
NAME="${SLURM_JOB_NAME}"
INPUT_DIR="$SCRATCH/AutoBrep/data/step/$NAME"
OUTPUT_DIR="$SCRATCH/AutoBrep/data/parquet/$NAME"

SCRIPT_PATH="$HOME/semester_project_euler/euler_scripts/preprocess_step_to_parquet.py"

echo "Starting Preprocessing from STEP to Parquet with Train/Val Split..."
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR (train/ and val/ subdirectories)"
echo ""
"$PYTHON" "$SCRIPT_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --train_val_split 1 \
    --seed 42

echo ""
echo "================================================================"
echo " Preprocessing complete!"
echo " Train data saved to:  $OUTPUT_DIR/train/data.parquet"
echo " Val data saved to:    $OUTPUT_DIR/val/data.parquet"
echo "================================================================"
