# Terminal command
TRAIN_JOB1=$(sbatch run_preprocess_autobrep.slurm | awk '{print $4}')
sbatch --dependency=afterok:$TRAIN_JOB1 run_train_lora_autobrep.slurm
