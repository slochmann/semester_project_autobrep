# AutoBrep semester project

Scripts for setting up and running [AutoBrep](https://github.com/AutodeskAILab/AutoBrep) inference on the ETHZ Euler HPC cluster.

## Files

### Euler HPC Cluster Setup
| File | Purpose |
|---|---|
| `euler inference/create_env_autobrep.sh` | One-time setup: clone repo, create conda env, fix dependencies, patch config |
| `euler inference/download_autobrep_ckpt.slurm` | SLURM job to download pretrained checkpoints (~13 GB) from HuggingFace |
| `euler inference/run_inference_autobrep.slurm` | SLURM job to run B-Rep generation inference |

### Local Visualization & Fine-tuning
| File | Purpose |
|---|---|
| `local visualization/render_images2.py` | Render generated STEP files as PNG images from multiple viewpoints using OpenCASCADE |
| `local visualization/make_gifs3.py` | Create MP4 videos from sequences of rendered images |
| `local visualization/output_collage.mp4` | Example output video showing rendered geometry frames |
| `LoRA.ipynb` | Jupyter notebook for fine-tuning AutoBrep using LoRA (Low-Rank Adaptation) |

## Workflow

**1. Set up the environment** (run once on the login node)
```bash
bash euler\ inference/create_env_autobrep.sh
```
This clones AutoBrep into `$SCRATCH/AutoBrep`, creates the `autobrep` conda env, upgrades `pytorch-lightning` to a Python 3.10-compatible version, installs the `autobrep` package, and patches `configs/sample.json` with the correct checkpoint path.

**2. Download pretrained checkpoints** (~13 GB, submit as a job)
```bash
sbatch euler\ inference/download_autobrep_ckpt.slurm
```
Downloads `ar.ckpt` (12 GB), `surf-fsq.ckpt` (720 MB), `edge-fsq.ckpt` (213 MB) from `SamGiantEagle/AutoBrep` on HuggingFace into `$SCRATCH/AutoBrep/checkpoints/`. Skipped automatically if already present.

**3. Run inference**
```bash
sbatch euler\ inference/run_inference_autobrep.slurm
```
Generates B-Rep STEP files into `$SCRATCH/AutoBrep/samples/`. Sampling parameters (complexity, batch size, temperature) are controlled via `$SCRATCH/AutoBrep/configs/sample.json`.

**4. (Optional) Render and visualize locally**
```bash
python local\ visualization/render_images2.py /path/to/step/files
python local\ visualization/make_gifs3.py
```
These scripts render the generated STEP files as images from multiple viewpoints and create MP4 videos for easy visualization and analysis.

## Fine-tuning with LoRA

The `LoRA.ipynb` notebook enables efficient fine-tuning of AutoBrep models using Low-Rank Adaptation (LoRA). This approach significantly reduces computational and memory requirements compared to full model fine-tuning, making it practical for local systems. LoRA works by training small, low-rank adjustment matrices that modify the model's behavior for your specific use case, while keeping the base model weights frozen. This is particularly useful for adapting AutoBrep to generate B-Reps with specific geometric or topological characteristics relevant to your domain. The notebook provides step-by-step instructions for loading pre-trained checkpoints, preparing training data, configuring LoRA parameters, and evaluating the fine-tuned model.

## Notes

- **Complexity** can be `random`, `easy`, `medium`, or `hard`
- Logs are written to `/cluster/scratch/slochmann/logs/`
- The inference job requires step 2 to complete first; it will exit with an error if checkpoints are missing
- Do not use `conda activate autobrep` in SLURM scripts — the scripts use the explicit Python path `$HOME/miniconda3/envs/autobrep/bin/python` instead
