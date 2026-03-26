#!/bin/bash
#SBATCH --job-name=autobrep_inference
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G                   # Total 32G (ar.ckpt is 12GB)
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/slochmann/logs/autobrep_inference_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/autobrep_inference_%j.err


mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/AutoBrep/samples"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.8.0
module load eth_proxy

#------------------------------------------------------------------------------
# Set up Conda environment Python (explicit path — conda activate is unreliable
# in non-interactive SLURM jobs)
#------------------------------------------------------------------------------
CONDA_DIR="$HOME/miniconda3"
PYTHON="$CONDA_DIR/envs/autobrep/bin/python"

# Clear any PYTHONPATH injected by the module system — it causes conda env
# packages (e.g. setuptools/pkg_resources) to be shadowed by system versions
unset PYTHONPATH

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------
AUTOBREP_DIR="$SCRATCH/AutoBrep"
CKPT_DIR="$SCRATCH/AutoBrep/checkpoints"
SAMPLE_CONFIG="$AUTOBREP_DIR/configs/sample.json"
OUTPUT_DIR="$SCRATCH/AutoBrep/samples/easy_untuned"

# Verify checkpoints exist
if [ ! -f "$CKPT_DIR/ar.ckpt" ]; then
    echo "ERROR: Checkpoints not found at $CKPT_DIR"
    echo "       Run download_autobrep_ckpt.slurm first."
    exit 1
fi

#------------------------------------------------------------------------------
# Run inference from core/ (required for autobrep package import)
#------------------------------------------------------------------------------
cd "$AUTOBREP_DIR/core" || { echo "ERROR: Could not cd into $AUTOBREP_DIR/core"; exit 1; }

"$PYTHON" -m autobrep.inference.sample_ar \
    --input_json "$SAMPLE_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --debug_dir  "$OUTPUT_DIR"

echo "================================================================"
echo " Inference complete!"
echo " Samples saved to: $OUTPUT_DIR"
echo "================================================================"
