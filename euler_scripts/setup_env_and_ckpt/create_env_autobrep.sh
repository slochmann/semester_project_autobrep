#!/bin/bash

#==============================================================================
# HPC Module Loading on ETHZ Euler
#==============================================================================
MODULE_STACK="stack/2024-06"
MODULE_GCC="gcc/12.2.0"
MODULE_PYTHON="python_cuda/3.11.6"
MODULE_CUDA="cuda/12.8.0"
MODULE_GITLFS="git-lfs/3.3.0"

echo "Loading HPC modules..."
module load "$MODULE_STACK"
module load "$MODULE_GCC"
module load "$MODULE_PYTHON"
module load "$MODULE_CUDA"
module load "$MODULE_GITLFS"

#==============================================================================
# Install Miniconda if not already installed
#==============================================================================
CONDA_DIR="$HOME/miniconda3"
CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
CONDA_URL="https://repo.anaconda.com/miniconda/$CONDA_INSTALLER"

if [ ! -d "$CONDA_DIR" ]; then
    echo
    echo "==== Miniconda not found. Installing Miniconda... ===="
    wget "$CONDA_URL" -O "$CONDA_INSTALLER"
    bash "$CONDA_INSTALLER" -b -p "$CONDA_DIR"
    rm "$CONDA_INSTALLER"
    echo "Miniconda installed at $CONDA_DIR"
else
    echo "Miniconda already installed at $CONDA_DIR"
fi

# Add Conda to PATH
export PATH="$CONDA_DIR/bin:$PATH"
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Initialize Conda for future sessions
if ! grep -q "conda initialize" ~/.bashrc; then
    echo "Running conda init..."
    conda init bash
fi

#==============================================================================
# Clone AutoBrep Repository
#==============================================================================
REPOS=(
    "git@github.com:slochmann/AutoBrep.git"
)
PROJECTS_DIR="$SCRATCH"

echo
echo "==== Cloning repositories into $PROJECTS_DIR ===="
mkdir -p "$PROJECTS_DIR"

for REPO_URL in "${REPOS[@]}"; do
    REPO_NAME=$(basename "$REPO_URL" .git)
    CLONE_DIR="$PROJECTS_DIR/$REPO_NAME"

    if [ ! -d "$CLONE_DIR" ]; then
        echo "Cloning $REPO_NAME into $CLONE_DIR..."
        git clone "$REPO_URL" "$CLONE_DIR"
    else
        echo "Directory $CLONE_DIR already exists. Skipping clone."
    fi
done

#==============================================================================
# Create and Activate Conda Environment
#==============================================================================
CONDA_ENV_NAME="autobrep"
DEV_ENV_FILE="$PROJECTS_DIR/AutoBrep/core/dev-env.yaml"

echo
echo "==== Creating Conda environment: $CONDA_ENV_NAME ===="
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Conda environment $CONDA_ENV_NAME already exists. Skipping creation."
else
    if [ -f "$DEV_ENV_FILE" ]; then
        echo "Navigating to core directory..."
        cd "$PROJECTS_DIR/AutoBrep/core" || { echo "ERROR: Could not cd into core directory"; exit 1; }
        conda env create -f dev-env.yaml --yes
    else
        echo "ERROR: dev-env.yaml file not found at $DEV_ENV_FILE"
        exit 1
    fi
fi

echo
echo "==== Installing autobrep package into conda env ==="
CORE_DIR="$PROJECTS_DIR/AutoBrep/core"
PIP="$HOME/miniconda3/envs/$CONDA_ENV_NAME/bin/pip"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV_NAME/bin/python"

# Fix: dev-env.yaml installs an old pytorch_lightning/lightning_fabric that
# uses collections.MutableMapping (removed in Python 3.10) and has broken
# pkg_resources due to a vendored old pyparsing. First ensure setuptools is
# intact, then upgrade lightning packages to 2.x which has all these fixes.
echo "Ensuring setuptools/pkg_resources is intact..."
"$HOME/miniconda3/bin/conda" install -n "$CONDA_ENV_NAME" setuptools --force-reinstall -y
echo "Reinstalling setuptools via pip to ensure pkg_resources is available..."
"$PIP" install --force-reinstall --no-cache-dir setuptools

echo "Upgrading pytorch-lightning and lightning to Python 3.10-compatible versions..."
"$PIP" install --upgrade --force-reinstall --no-cache-dir "pytorch-lightning>=2.0" "lightning-fabric>=2.0" "lightning>=2.0"

echo "Installing autobrep package..."
"$PIP" install -e "$CORE_DIR"

# Verify the critical imports work before declaring success
echo "Verifying environment..."
"$PYTHON" -c "from pytorch_lightning import seed_everything; import pkg_resources; print('Environment OK')" || {
    echo "ERROR: Environment verification failed. Check the output above."
    exit 1
}

#==============================================================================
# Patch sample.json weight_folder to point to the checkpoint download directory
#==============================================================================
SAMPLE_JSON="$PROJECTS_DIR/AutoBrep/configs/sample.json"
CKPT_DIR="$SCRATCH/AutoBrep/checkpoints"

if [ -f "$SAMPLE_JSON" ]; then
    echo
    echo "==== Patching weight_folder in $SAMPLE_JSON ===="
    python3 -c "
import json
with open('$SAMPLE_JSON') as f:
    config = json.load(f)
config['weight_folder'] = '$CKPT_DIR'
with open('$SAMPLE_JSON', 'w') as f:
    json.dump(config, f, indent=4)
print('weight_folder set to: $CKPT_DIR')
"
else
    echo "WARNING: $SAMPLE_JSON not found. Skipping patch."
fi

#==============================================================================
# Final Instructions
#==============================================================================
echo
echo "================================================================"
echo " Environment setup complete!"
echo " Conda environment: $CONDA_ENV_NAME"
echo " Repositories have been cloned into: $PROJECTS_DIR"
echo ""
echo " To re-activate the environment in future sessions, run:"
echo "    conda activate $CONDA_ENV_NAME"
echo " Then, navigate to the core directory:"
echo "    cd \"$PROJECTS_DIR/AutoBrep/core\""
echo "================================================================"
