#!/bin/bash
#SBATCH --job-name=create_collages
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/slochmann/logs/create_collages_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/create_collages_%j.err

#==============================================================================
# CONFIGURATION - Modify these parameters
#==============================================================================
SAMPLES_SUBDIR="samples"                  # Subdirectory under $SCRATCH/AutoBrep/
DRY_RUN=false                             # Set to true to preview without creating
#==============================================================================

mkdir -p "$SCRATCH/logs"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load eth_proxy

#------------------------------------------------------------------------------
# Set up Conda environment Python
#------------------------------------------------------------------------------
CONDA_DIR="$HOME/miniconda3"
PYTHON="$CONDA_DIR/envs/autobrep/bin/python"

# Clear any PYTHONPATH injected by the module system
unset PYTHONPATH

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------
AUTOBREP_DIR="$SCRATCH/AutoBrep"
SAMPLES_DIR="$AUTOBREP_DIR/$SAMPLES_SUBDIR"
SCRIPT_DIR="$HOME/semester_project_euler/euler_scripts"

#------------------------------------------------------------------------------
# Run collage creation script
#------------------------------------------------------------------------------
echo "================================================================"
echo " Creating collages from samples in: $SAMPLES_DIR"
echo "================================================================"

if [ ! -d "$SAMPLES_DIR" ]; then
    echo "ERROR: Samples directory not found: $SAMPLES_DIR"
    exit 1
fi

DRY_RUN_FLAG=""
if [ "$DRY_RUN" = "true" ]; then
    DRY_RUN_FLAG="--dry-run"
fi

"$PYTHON" "$SCRIPT_DIR/create_inference_collages.py" "$SAMPLES_DIR" $DRY_RUN_FLAG

echo "================================================================"
echo " Collage creation complete!"
echo "================================================================"
