#!/bin/bash
#SBATCH --job-name=600_cubes
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
OUTPUT_FILE="$SCRATCH/AutoBrep/data/parquet/train/$NAME.parquet"

SCRIPT_PATH="$HOME/semester_project_euler/euler_scripts/preprocess_step_to_parquet.py"

echo "Starting Preprocessing from STEP to Parquet..."
"$PYTHON" "$SCRIPT_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_file "$OUTPUT_FILE"

echo "================================================================"
echo " Preprocessing complete!"
echo " Output saved to: $OUTPUT_FILE"
echo "================================================================"
