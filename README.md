# AutoBrep Semester Project

Fine-tuning and inference scripts for [AutoBrep](https://github.com/AutodeskAILab/AutoBrep) on Euler HPC cluster, with local visualization utilities.

## Quick Start

**1. Setup (Euler, once only):**
```bash
bash euler_scripts/create_env_autobrep.sh
sbatch euler_scripts/download_autobrep_ckpt.slurm
```

**2. Prepare data:**
```bash
python euler_scripts/preprocess_step_to_parquet.py \
    --input_dir /path/to/steps --output_dir $SCRATCH/AutoBrep/data/parquet --train_split 0.9
```

**3. Train LoRA adapter:**
```bash
sbatch euler_scripts/run_train_lora_autobrep.slurm
```
Edit `run_train_lora_autobrep.slurm` to adjust: `--learning_rate` (1e-4 default), `--num_epochs` (50), `--batch_size` (4), `--lora_r` (8).

**4. Generate B-Reps:**
```bash
sbatch euler_scripts/run_inference_lora_autobrep.slurm
```
Edit for: `--num_samples` (200), `--complexity` (14=easy, 15=medium, 16=hard), `--temperature` (1.0), `--threshold` (0.95).

**5. Visualize locally:**
```bash
python local_visualization/render_images2.py /path/to/steps
python local_visualization/make_gifs3.py
```

## Key Scripts

| Script | Purpose |
|---|---|
| `euler_scripts/create_env_autobrep.sh` | Clone AutoBrep, setup conda env, download checkpoints |
| `euler_scripts/preprocess_step_to_parquet.py` | Convert STEP files → parquet (UV grids, bboxes, topology) |
| `euler_scripts/train_lora_ar.py` | LoRA training (called by SLURM wrapper) |
| `euler_scripts/sample_lora_ar.py` | Inference and STEP export (called by SLURM wrapper) |
| `local_visualization/render_images2.py` | Render STEP → PNG (OpenCASCADE) |
| `local_visualization/make_gifs3.py` | Create MP4 from PNG sequences |

## Hyperparameter Tuning

**For 100–500 samples:**
- **Learning rate:** 1e-4 (too high → overfitting, too low → slow learning)
- **Epochs:** 30–50 (monitor W&B loss, should plateau ~0.8–1.0, not collapse to 0)
- **LoRA r:** 8 for 100–500 samples; increase to 16 for >1k samples
- **Inference temp:** 1.0 default; lower to 0.8–0.9 if too diverse

**Low success rate (<50%)?** Check training loss (should not be near 0), try lower temperature (0.9) and threshold (0.90).

## File Locations (Euler)

- `$SCRATCH/AutoBrep/core/` — AutoBrep source
- `$SCRATCH/AutoBrep/ckpt/` — Pretrained checkpoints
- `$SCRATCH/AutoBrep/data/parquet/` — Training dataset
- `$SCRATCH/AutoBrep/checkpoints/` — Trained LoRA adapters
- `$SCRATCH/AutoBrep/lora_samples/` — Generated STEP files

## Important Notes

- Use explicit Python path in SLURM (no `conda activate`)
- W&B logging enabled by default; track at https://wandb.ai/slochmann-ethz/lora-autobrep
- Success rate = % of generated samples that form valid STEP files
- Chain jobs with SLURM dependencies: `sbatch --dependency=afterok:$JOB_ID euler_scripts/run_inference_lora_autobrep.slurm`

## Troubleshooting

| Issue | Fix |
|---|---|
| Success rate ~5% | Retrain with lower LR (1e-4), check loss doesn't collapse to 0 |
| OOM during training | Reduce `--batch_size` or set `PYTORCH_ALLOC_CONF=expandable_segments:True` |
| Checkpoints not found | Run `download_autobrep_ckpt.slurm` first |
| Loss not decreasing | Lower LR to 5e-4 briefly to verify gradients, check dataset format |
