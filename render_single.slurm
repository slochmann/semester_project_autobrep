#!/bin/bash
#SBATCH --job-name=autobrep_render
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/slochmann/logs/render_%a.out
#SBATCH --error=/cluster/scratch/slochmann/logs/render_%a.err

module purge
module load stack/2024-06
module load gcc/12.2.0
module load eth_proxy
module load mesa

CONDA_DIR="$HOME/miniconda3"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate autobrep

SAMPLES_DIR="$SCRATCH/AutoBrep/samples"
RENDER_DIR="$SCRATCH/AutoBrep/images"
mkdir -p "$RENDER_DIR"

# Get the STEP file for this array task
STEP_FILES=($(ls $SAMPLES_DIR/*.step | sort))
STEP_FILE=${STEP_FILES[$SLURM_ARRAY_TASK_ID - 1]}

# Create subdirectory for this file's output using the step filename
FILENAME=$(basename "$STEP_FILE" .step)
FILE_OUTPUT_DIR="$RENDER_DIR/$FILENAME"
mkdir -p "$FILE_OUTPUT_DIR"

echo "=== Task $SLURM_ARRAY_TASK_ID ==="
echo "STEP file: $STEP_FILE"
echo "Output dir: $FILE_OUTPUT_DIR"

# Process single file
xvfb-run -a python /cluster/home/slochmann/semester_project_euler/render_images.py "$STEP_FILE" "$FILE_OUTPUT_DIR"
