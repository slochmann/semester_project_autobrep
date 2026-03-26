# Terminal command
TRAIN_JOB=$(sbatch run_train_lora_autobrep.slurm | awk '{print $4}')
sbatch --dependency=afterok:$TRAIN_JOB run_inference_lora_autobrep.slurm
