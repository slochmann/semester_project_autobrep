# AutoBrep Semester Project

Scripts for setting up, running, and fine-tuning [AutoBrep](https://github.com/AutodeskAILab/AutoBrep) on the ETHZ Euler HPC cluster.

## Files

### Initial Setup (One-time)
| File | Purpose |
|---|---|
| `euler_scripts/create_env_autobrep.sh` | Clone AutoBrep repo, create conda env, install dependencies, patch configs |
| `euler_scripts/download_autobrep_ckpt.slurm` | Download pretrained checkpoints (~13 GB) from HuggingFace to `$SCRATCH/AutoBrep/ckpt/` |

### Data Preprocessing
| File | Purpose |
|---|---|
| `euler_scripts/preprocess_step_to_parquet.py` | Convert STEP files → parquet dataset (UV grids + bboxes + topology) |

### Fine-tuning with LoRA
| File | Purpose |
|---|---|
| `euler_scripts/train_lora_ar.py` | Train LoRA adapters on custom STEP dataset (main script) |
| `euler_scripts/run_train_lora_autobrep.slurm` | SLURM job wrapper for training with hyperparameter control |
| `euler_scripts/sample_lora_ar.py` | Generate B-Reps from fine-tuned LoRA adapter (inference script) |
| `euler_scripts/run_inference_lora_autobrep.slurm` | SLURM job wrapper for inference and STEP export |

### Local Visualization
| File | Purpose |
|---|---|
| `local_visualization/render_images2.py` | Render STEP files as PNG images from multiple viewpoints (OpenCASCADE) |
| `local_visualization/make_gifs3.py` | Create MP4 videos from rendered image sequences |

## Workflow

### 1. Initial Environment Setup (Run Once)

```bash
ssh euler
bash $HOME/semester_project_euler/euler_scripts/create_env_autobrep.sh
sbatch euler_scripts/download_autobrep_ckpt.slurm
```

This clones AutoBrep to `$SCRATCH/AutoBrep`, creates the `autobrep` conda environment, and downloads checkpoints (~13 GB).

### 2. Prepare Training Data

Convert your STEP files to the parquet format expected by AutoBrep:

```bash
python euler_scripts/preprocess_step_to_parquet.py \
    --input_dir /path/to/step/files \
    --output_dir $SCRATCH/AutoBrep/data/parquet \
    --train_split 0.9
```

This generates:
- `train/` — parquet dataset with UV grids, bboxes, topology
- `val/` — validation split (~10%)

### 3. Fine-tune with LoRA

Submit a training job:

```bash
sbatch euler_scripts/run_train_lora_autobrep.slurm
```

**Configurable hyperparameters** (edit `run_train_lora_autobrep.slurm`):

| Parameter | Default | Notes |
|---|---|---|
| `--batch_size` | 4 | Batch size for training |
| `--num_epochs` | 50 | Number of training epochs (consider early stopping for small datasets) |
| `--learning_rate` | 1e-4 | Learning rate for AdamW (use 1e-4 to 5e-5 for small datasets to avoid overfitting) |
| `--lora_r` | 8 | LoRA rank (memory/capacity tradeoff; 8–16 typical) |
| `--lora_alpha` | 32 | LoRA scaling factor (effective scale = alpha/r) |

**Typical convergence:**
- 100 samples, 50 epochs, batch_size=4 → ~1,250 training steps
- Final loss should plateau around 0.8–1.0 (not go to near-zero, which indicates overfitting)
- W&B tracks loss, learning rate, and system metrics in real-time

The adapter is saved to `$SCRATCH/AutoBrep/checkpoints/<job_name>_<timestamp>/`.

### 4. Generate B-Reps (Inference)

```bash
sbatch euler_scripts/run_inference_lora_autobrep.slurm
```

**Configurable inference parameters**:

| Parameter | Default | Notes |
|---|---|---|
| `--num_samples` | 200 | Number of B-Reps to generate |
| `--complexity` | 14 | Conditioning token (14=easy, 15=medium, 16=hard, 17=random). Use 14 for simple shapes. |
| `--temperature` | 1.0 | Sampling temperature (lower = less diverse, more confident; 0.8–1.0 typical) |
| `--threshold` | 0.95 | Top-p sampling threshold (keep top p% of probability mass; 0.90–0.95 typical) |
| `--batch_size` | 10 | Concurrent samples (adjust based on available GPU memory) |
| `--seed` | 42 | Random seed for reproducibility |

Outputs:
- Generated STEP files: `$SCRATCH/AutoBrep/lora_samples/<job_name>/`
- Exit codes and success rate printed to stdout/stderr logs

**Expected success rates:**
- Well-trained adapter on in-distribution shapes: 70–90%
- Early-stage/overfit models: 5–50%
- Poor success rate suggests: overfitting (loss → 0), bad hyperparams (LR too high), or mismatched complexity conditioning

### 5. Pipeline Automation (Optional)

Chain training + inference in one job submission:

```bash
# Option A: SLURM job dependencies
TRAIN_JOB=$(sbatch euler_scripts/run_train_lora_autobrep.slurm | awk '{print $4}')
sbatch --dependency=afterok:$TRAIN_JOB euler_scripts/run_inference_lora_autobrep.slurm
```

Inference will wait in queue until training completes successfully.

### 6. Render and Visualize Results (Local)

Copy generated STEP files to local machine:

```bash
scp -r euler:/cluster/scratch/slochmann/AutoBrep/lora_samples /local/output/path
```

Render as images:

```bash
python local_visualization/render_images2.py /local/output/path/lora_samples
python local_visualization/make_gifs3.py
```

Creates PNG images and MP4 videos for visual inspection.

## Key Tuning Decisions

### Learning Rate & Epochs for Small Datasets

For 100–500 samples, use **conservative LR (1e-4)** and **moderate epochs (30–50)**:
- **Too high LR (5e-4+)**: Model memorizes training data → loss → 0, but inference crashes (bad success rate)
- **Too low LR (1e-5)**: Slow convergence, may not learn domain-specific patterns
- **Too many epochs**: Overfitting on small dataset
- **Too few epochs**: Under-trained model

Monitor W&B loss curve: should plateau around **0.8–1.0** (not collapse to near-zero).

### LoRA Hyperparameters

- **r=8**: Suitable for 100–500 sample fine-tunes. Increase to 16 if underfitting or if you have >1k samples.
- **alpha=32**: Default scaling factor; alpha/r = 4× effective learning rate for LoRA params.
- **target_modules=["to_q", "to_v"]**: Standard choice targeting attention query/value projections. Add "to_k" or "to_out" for more capacity if needed.

### Inference Temperature & Top-p

- **temperature=1.0**: Default, full entropy. Matches training distribution.
- **temperature=0.8**: Reduces randomness; use if model is underfitting or generating too much diversity.
- **threshold=0.95**: Keeps top 95% probability mass; reduces degenerate tokens.
- **Lower values (0.8, 0.90)**: More conservative, fewer degenerate outputs but potentially lower diversity.

If success rate is low (<50%), try:
1. Verify training loss plateaued at 0.8–1.0 (not near 0 = overfitting)
2. Lower temperature to 0.9 and threshold to 0.90
3. Increase `--num_samples` (generate more and accept rejections)

## Environment Variables

- `WANDB_PROJECT`: W&B project name (set in SLURM files)
- `WANDB_ENTITY`: W&B team/user name
- `WANDB_API_KEY`: Auto-logged by the scripts; set in SLURM files
- `HF_TOKEN`: HuggingFace API token for checkpoint downloads
- `PYTORCH_ALLOC_CONF=expandable_segments:True`: Reduces GPU fragmentation

**W&B Dashboard:** https://wandb.ai/slochmann-ethz/lora-autobrep

All training runs are logged here in real-time. Monitor loss curves, hyperparameters, and system metrics during training.

## File Locations

| Path | Purpose |
|---|---|
| `$SCRATCH/AutoBrep/core/` | AutoBrep source code (cloned by setup script) |
| `$SCRATCH/AutoBrep/ckpt/` | Pretrained base checkpoints (ar.ckpt, surf-fsq.ckpt, edge-fsq.ckpt) |
| `$SCRATCH/AutoBrep/data/parquet/` | Training dataset (train/ and val/ splits) |
| `$SCRATCH/AutoBrep/checkpoints/` | LoRA adapter checkpoints (organized by job_name_timestamp) |
| `$SCRATCH/AutoBrep/lora_samples/` | Generated STEP files and logs |
| `/cluster/scratch/slochmann/logs/` | SLURM job logs (*.out, *.err) |

## Notes

- **Do not use `conda activate autobrep`** in SLURM scripts — scripts use explicit Python path instead
- **Always run Python scripts from `$SCRATCH/AutoBrep/core/`** (required for `autobrep` package imports)
- **Timestamps are added to checkpoint directories** by the training script to avoid overwrites
- **W&B tracking** is enabled by default and logs to the configured project; disable with `--track False` if needed
- **Success rate metric**: Percentage of generated samples that successfully reconstruct as valid STEP files (sewing + geometry validity)

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| "LoRA adapter not found" | Inference uses wrong adapter path | Check checkpoint dir name matches training job output |
| Success rate ~5% | Model overfit (loss → 0) | Redo training with lower LR (1e-4) and more validation |
| "Could not cd into $AUTOBREP_DIR/core" | Checkpoints not downloaded | Run `download_autobrep_ckpt.slurm` first |
| OOM errors during training | Batch too large or GPU fragmentation | Reduce `--batch_size` or set `PYTORCH_ALLOC_CONF` |
| Loss not decreasing | LR too low, data issue, or model mismatch | Try LR=5e-4 briefly to check gradient flow, verify dataset format |
