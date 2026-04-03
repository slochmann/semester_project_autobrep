# AutoBrep Semester Project

Fine-tuning and inference scripts for [AutoBrep](https://github.com/AutodeskAILab/AutoBrep) on Euler HPC cluster, with local visualization utilities.

## Quick Start

**1. Setup (Euler, once):**
```bash
bash euler_scripts/setup_env_and_ckpt/create_env_autobrep.sh
sbatch euler_scripts/setup_env_and_ckpt/download_autobrep_ckpt.slurm
```

**2. Add training data to Euler:**
After setup completes, copy your STEP files to `$SCRATCH/AutoBrep/data/step/`:
```bash
scp -r /local/path/to/steps euler:$SCRATCH/AutoBrep/data/step/
```

**3. Preprocess data:**
```bash
sbatch euler_scripts/job_submission/run_preprocess_autobrep.slurm
```
Converts STEP → parquet (UV grids, bboxes) into `$SCRATCH/AutoBrep/data/parquet/{train,val}/`

**4. Train LoRA adapter:**
```bash
sbatch euler_scripts/job_submission/run_train_lora_autobrep.slurm
```
Edit script to adjust hyperparameters (see Hyperparameter Tuning section below).

**Samples generated during training** are logged to W&B in real-time. View them at https://wandb.ai/slochmann-ethz/lora-autobrep.

**5. (Optional) Generate more B-Reps:**
For additional inference runs beyond samples logged during training:
```bash
sbatch euler_scripts/job_submission/run_inference_lora_autobrep.slurm
```
Edit script to adjust: `--num_samples`, `--complexity`, `--temperature`, `--threshold`.

**6. Visualize locally:**
```bash
scp -r euler:\$SCRATCH/AutoBrep/lora_samples /local/output/path
python local_visualization/render_images2.py /local/output/path/lora_samples
python local_visualization/make_gifs3.py
```

## Project Structure

```
.
├── README.md                          # This file
├── euler_scripts/                     # Euler HPC scripts & configs
│   ├── train_lora_ar.py              # Training script
│   ├── sample_lora_ar.py             # Inference script
│   ├── preprocess_step_to_parquet.py # Data preprocessing
│   ├── job_submission/               # SLURM job wrappers
│   └── setup_env_and_ckpt/          # Environment & checkpoint setup
├── local_visualization/              # Local rendering tools
│   ├── render_images2.py            # STEP → PNG (OpenCASCADE)
│   ├── make_gifs3.py                # PNG sequence → MP4
│   └── verify_parquet.py            # Data validation
├── euler_remote_mount/               # Euler remote mount utilities
│   ├── login.sh
│   └── ssh-keygen.sh
└── report/                           # Thesis & documentation
    ├── MAIN.tex
    ├── chapters/
    ├── figures/
    └── bib/
```

## Hyperparameter Tuning

**Training defaults (in `euler_scripts/job_submission/run_train_lora_autobrep.slurm`):**
- **Batch size:** 4
- **Epochs:** 4
- **Learning rate:** 5e-5 (too high → overfitting, too low → slow learning)
- **LoRA r:** 32
- **LoRA alpha:** 128
- **Sample temperature:** 0.4 (generation diversity during training)
- **Sample complexity:** 17 (random/unconditioned; 14=easy, 15=medium, 16=hard)
- **Sample interval:** 1 epoch or 65 batches

**Inference defaults (in `euler_scripts/job_submission/run_inference_lora_autobrep.slurm`):**
- **Num samples:** 200
- **Temperature:** 1.0 (lower = more deterministic, higher = more random)
- **Threshold:** 0.95 (top-p nucleus sampling; lower = stricter grammar compliance)
- **Complexity:** 14 (easy; values: 14=easy, 15=medium, 16=hard, 17=random)
- **Batch size:** 10 (reduce if OOM)
- **Seed:** 42 (set for reproducibility)

**Monitoring:** Check W&B loss curve—should plateau around 0.8–1.0, not collapse to 0 (indicates overfitting).

**Low success rate (<50%)?** Increase epochs, lower learning rate, or reduce temperature.

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
