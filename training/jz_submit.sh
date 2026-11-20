#!/bin/bash
# Submits the 3 methods: 3 folds each in parallel, then the summary once its 3 folds succeed.
# Run from the repo root on Jean Zay:  bash training/jz_submit.sh
set -e
mkdir -p logs
for M in adversarial contrastive mmd; do
  TRAIN=$(sbatch --parsable --array=0-2 --job-name=da_$M training/jz_da.slurm $M train)
  POST=$(sbatch --parsable --dependency=afterok:$TRAIN --job-name=post_$M training/jz_da.slurm $M post)
  echo "$M: training job $TRAIN (3 folds), summary job $POST (starts when all 3 folds succeed)"
done
echo "follow with: squeue -u \$USER    logs in logs/"