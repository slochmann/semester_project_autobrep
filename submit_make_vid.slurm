#!/bin/bash
#SBATCH --job-name=autobrep_download
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/slochmann/logs/make_vid_%j.out
#SBATCH --error=/cluster/scratch/slochmann/logs/make_vid_%j.err

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load eth_proxy

CONDA_DIR="$HOME/miniconda3"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate autobrep

IMG_DIR="$SCRATCH/AutoBrep/images"
VID_DIR="$SCRATCH/AutoBrep/videos"
mkdir -p "$VID_DIR"

python make_vid.py $IMG_DIR $VID_DIR
