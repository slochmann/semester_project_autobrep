# Count files and submit all at once
NUM_FILES=$(ls $SCRATCH/AutoBrep/samples/*.step 2>/dev/null | wc -l)
echo "Submitting $NUM_FILES jobs..."
sbatch --array=1-3 render_single.slurm
