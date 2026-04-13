#!/bin/bash
#SBATCH --job-name=autobrep_inference
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G                   # Total 32G (ar.ckpt is 12GB)
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/slochmann/logs/autobrep_inference_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/autobrep_inference_%j.err

#==============================================================================
# CONFIGURATION - Modify these parameters to change inference settings
#==============================================================================
# Suggested configurations to explore:
#   CONSERVATIVE:     T=0.5, p=0.7, c=easy   → deterministic, simple designs
#   BALANCED:         T=0.8, p=0.9, c=random → balanced diversity & quality
#   DIVERSE:          T=1.2, p=0.95, c=hard  → high diversity, complex designs
#   VERY_CREATIVE:    T=1.5, p=0.99, c=random→ maximum diversity, less predictable
#   CONTROLLED:       T=0.7, p=0.85, c=medium→ moderate diversity, med. complexity
#==============================================================================

OUTPUT_BASE="untuned"                     # Base name for output subdirectory
NUM_BATCHES=33                             # Number of batches to sample
BATCH_SIZE=4                              # Batch size per sample
COMPLEXITY="random"                         # Complexity level: easy, medium, hard, random
TEMPERATURE=0.6                          # Sampling temperature
TOP_P=0.9                                 # Top-p / nucleus sampling threshold
VERTEX_THRESHOLD=0.002                    # Vertex threshold
SEWING_TOLERANCE=0.002                    # Sewing tolerance
Z_THRESHOLD=0                              # Z threshold
USE_SEED=false                            # Whether to use seed
SEED_VALUE=689447                         # Seed value (if use_seed=true)

# Get timestamp for unique naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Construct output subdirectory with hyperparameters and timestamp
OUTPUT_SUBDIR="${OUTPUT_BASE}_b${NUM_BATCHES}_bs${BATCH_SIZE}_t${TEMPERATURE}_c${COMPLEXITY}_p${TOP_P}_${TIMESTAMP}"
#==============================================================================

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
# Paths and Generate Config JSON
#------------------------------------------------------------------------------
AUTOBREP_DIR="$SCRATCH/AutoBrep"
CKPT_DIR="$SCRATCH/AutoBrep/checkpoints"
OUTPUT_DIR="$SCRATCH/AutoBrep/samples/$OUTPUT_SUBDIR"

# Generate config JSON dynamically from parameters
SAMPLE_CONFIG="$OUTPUT_DIR/sample_config.json"
mkdir -p "$(dirname "$SAMPLE_CONFIG")"

export USE_SEED NUM_BATCHES BATCH_SIZE COMPLEXITY TEMPERATURE TOP_P VERTEX_THRESHOLD SEWING_TOLERANCE Z_THRESHOLD SEED_VALUE CKPT_DIR

envsubst < /dev/stdin > "$SAMPLE_CONFIG" << 'EOJSON'
{
    "api_version": {
        "major_version": 1,
        "minor_version": 0,
        "build_number": 0
    },
    "seed": {
        "use_seed": $USE_SEED,
        "seed_value": $SEED_VALUE
    },
    "debug_mode": true,
    "output_format": "step",
    "hyper_parameters": {
        "num_batches_to_sample": $NUM_BATCHES,
        "batch_size": $BATCH_SIZE,
        "complexity": "$COMPLEXITY",
        "temperature": $TEMPERATURE,
        "sample_method": {
            "name": "top_p",
            "top_p_threshold": $TOP_P
        },
        "vertex_threshold": $VERTEX_THRESHOLD,
        "sewing_tolerance": $SEWING_TOLERANCE,
        "z_threshold": $Z_THRESHOLD
    },
    "weight_folder": "$CKPT_DIR"
}
EOJSON

echo "✅ Config generated: $SAMPLE_CONFIG"

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
echo " To create collages, run: create_inference_collages.py <samples_dir>"
echo "================================================================"
