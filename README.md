# AutoBrep Semester Project

Fine-tuning and inference scripts for [AutoBrep](https://github.com/AutodeskAILab/AutoBrep) on Euler HPC cluster, with local visualization utilities.

## Quick Start

**1. Setup (Euler, once):**
```bash
bash euler_scripts/create_env_autobrep.sh
sbatch euler_scripts/download_autobrep_ckpt.slurm
```

**2. Add training data to Euler:**
After setup completes, copy your STEP files to `$SCRATCH/AutoBrep/data/step/`:
```bash
scp -r /local/path/to/steps euler:$SCRATCH/AutoBrep/data/step/
```

**3. Preprocess data:**
```bash
sbatch euler_scripts/run_preprocess_autobrep.slurm
```
Converts STEP → parquet (UV grids, bboxes) into `$SCRATCH/AutoBrep/data/parquet/{train,val}/`

**4. Train LoRA adapter:**
```bash
sbatch euler_scripts/run_train_lora_autobrep.slurm
```
Edit `run_train_lora_autobrep.slurm` to adjust: `--learning_rate` (1e-4), `--num_epochs` (5), `--batch_size` (2), `--lora_r` (8).

**Samples generated during training** are logged to W&B in real-time. View them at https://wandb.ai/slochmann-ethz/lora-autobrep.

**5. (Optional) Generate more B-Reps:**
For additional inference runs beyond samples logged during training:
```bash
sbatch euler_scripts/run_inference_lora_autobrep.slurm
```
Edit for: `--num_samples` (10), `--complexity` (14=easy, 15=medium, 16=hard), `--temperature` (0.8).

**6. Visualize locally:**
```bash
scp -r euler:\$SCRATCH/AutoBrep/lora_samples /local/output/path
python local_visualization/render_images2.py /local/output/path/lora_samples
python local_visualization/make_gifs3.py
```

## Project Structure

```
euler_scripts/               # Euler HPC scripts & configs
├── create_env_autobrep.sh
├── *.slurm                 # Job wrappers (train, inference, preprocess)
├── train_lora_ar.py        # Training script
├── sample_lora_ar.py       # Inference script
└── preprocess_step_to_parquet.py

local_visualization/        # Local rendering tools
├── render_images2.py       # STEP → PNG (OpenCASCADE)
├── make_gifs3.py          # PNG sequence → MP4
└── verify_parquet.py       # Data validation

cloned_project/            # AutoBrep & BrepGen repos
├── AutoBrep/
└── BrepGen/

report/                     # Thesis/documentation
```

## Hyperparameter Tuning

**Training defaults (adjust in SLURM wrapper):**
- **Learning rate:** 1e-4 (too high → overfitting, too low → slow learning)
- **Epochs:** 5
- **Batch size:** 4
- **LoRA r:** 32

**Inference defaults (adjust in SLURM wrapper):**
- **Num samples:** 10 (increase for more diverse outputs)
- **Temperature:** 0.5 (lower = more deterministic, higher = more random)
- **Complexity:** 17 (14=easy, 15=medium, 16=hard, 17=random)
- **Sample batch size:** 10 (reduce if OOM)

**Monitoring:** Check W&B loss curve—should plateau around 0.8–1.0, not collapse to 0 (indicates overfitting).

**Low success rate (<50%)?** Check training loss plateaued above 0, increase epochs, lower learning rate.

## Euler File Locations

**Data pipeline on Euler:**
- `$SCRATCH/AutoBrep/data/step/` — ⬅️ **ADD YOUR STEP FILES HERE** (before preprocessing)
- `$SCRATCH/AutoBrep/data/parquet/` — Preprocessed dataset (train/ and val/)
- `$SCRATCH/AutoBrep/ckpt/` — Pretrained base checkpoints
- `$SCRATCH/AutoBrep/checkpoints/` — LoRA adapters (timestamped by training script)
- `$SCRATCH/AutoBrep/lora_samples/` — Generated STEP files & logs

## Important Notes

- **STEP file location:** Copy your training STEP files to `$SCRATCH/AutoBrep/data/step/` after environment setup but **before** running preprocessing
- Use explicit Python path in SLURM jobs (no `conda activate`)
- **W&B samples:** Samples are generated and logged to W&B during training; view at https://wandb.ai/slochmann-ethz/lora-autobrep (inference job is optional for additional samples)
- Success rate = % of generated samples that form valid STEP files
- Chain jobs with SLURM dependencies: `sbatch --dependency=afterok:$JOB_ID euler_scripts/run_inference_lora_autobrep.slurm`

## Troubleshooting

| Issue | Fix |
|---|---|
| Success rate <80% | Check training loss didn't collapse to 0 (overfitting); increase epochs or reduce batch size |
| OOM during training | Reduce `--batch_size` (default 2) or set `PYTORCH_ALLOC_CONF=expandable_segments:True` |
| Checkpoints not found | Run `download_autobrep_ckpt.slurm` first |
| Loss not decreasing | Verify learning rate is appropriate (default 1e-4), check dataset format |
