# AutoBrep on ETHZ Euler

Scripts for setting up and running [AutoBrep](https://github.com/AutodeskAILab/AutoBrep) inference on the ETHZ Euler HPC cluster.

## Files

| File | Purpose |
|---|---|
| `create_env_autobrep.sh` | One-time setup: clone repo, create conda env, fix dependencies, patch config |
| `download_autobrep_ckpt.slurm` | SLURM job to download pretrained checkpoints (~13 GB) from HuggingFace |
| `run_inference_autobrep.slurm` | SLURM job to run B-Rep generation inference |

## Workflow

**1. Set up the environment** (run once on the login node)
```bash
bash create_env_autobrep.sh
```
This clones AutoBrep into `$SCRATCH/AutoBrep`, creates the `autobrep` conda env, upgrades `pytorch-lightning` to a Python 3.10-compatible version, installs the `autobrep` package, and patches `configs/sample.json` with the correct checkpoint path.

**2. Download pretrained checkpoints** (~13 GB, submit as a job)
```bash
sbatch download_autobrep_ckpt.slurm
```
Downloads `ar.ckpt` (12 GB), `surf-fsq.ckpt` (720 MB), `edge-fsq.ckpt` (213 MB) from `SamGiantEagle/AutoBrep` on HuggingFace into `$SCRATCH/AutoBrep/checkpoints/`. Skipped automatically if already present.

**3. Run inference**
```bash
sbatch run_inference_autobrep.slurm
```
Generates B-Rep STEP files into `$SCRATCH/AutoBrep/samples/`. Sampling parameters (complexity, batch size, temperature) are controlled via `$SCRATCH/AutoBrep/configs/sample.json`.

## Notes

- **Complexity** can be `random`, `easy`, `medium`, or `hard`
- Logs are written to `/cluster/scratch/slochmann/logs/`
- The inference job requires step 2 to complete first; it will exit with an error if checkpoints are missing
- Do not use `conda activate autobrep` in SLURM scripts — the scripts use the explicit Python path `$HOME/miniconda3/envs/autobrep/bin/python` instead
