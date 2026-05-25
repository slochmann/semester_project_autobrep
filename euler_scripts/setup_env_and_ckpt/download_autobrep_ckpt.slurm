#!/bin/bash
#SBATCH --job-name=autobrep_download
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/slochmann/logs/autobrep_download_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/autobrep_download_%j.err

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load eth_proxy

CONDA_DIR="$HOME/miniconda3"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate autobrep

CKPT_DIR="$SCRATCH/AutoBrep/checkpoints"
mkdir -p "$CKPT_DIR"

pip install -q huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='SamGiantEagle/AutoBrep', local_dir='$CKPT_DIR',
                  ignore_patterns=['*.png', 'README.md', '.gitattributes'])
print('Done.')
"
