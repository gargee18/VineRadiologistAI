#!/bin/bash
# Domain adaptation on Jean Zay. Submit with training/jz_submit.sh, not by hand.
#   sbatch --array=0-2 jz_da.slurm <method> train   -> one fold per array task, in parallel
#   sbatch jz_da.slurm <method> post                -> summary, replot, D1 agreement
# CHECK before the first run: the account (-A), the GPU type (-C) and the module name.
#SBATCH --job-name=vrai_da
#SBATCH -A hgo@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -eo pipefail
METHOD=$1
STEP=$2

module purge
module load arch/h100                     # needed for the H100 partition on Jean Zay
module load pytorch-gpu/py3/2.3.0         # CHECK: module avail pytorch-gpu
source $WORK/vrai_env/bin/activate        # venv made with --system-site-packages

export VRAI_REPO=$WORK/VineRadiologistAI
export VRAI_VIEWS=$SCRATCH/vrai/d2_views  # d1_views must sit next to it: $SCRATCH/vrai/d1_views
cd $VRAI_REPO

FOLDS=(
  "CEP011_AS1 CEP014_RES1 CEP017_S1 CEP020_APO1"
  "CEP012_AS2 CEP015_RES2 CEP018_S2 CEP021_APO2"
  "CEP013_AS3 CEP016_RES3 CEP019_S3 CEP022_APO3"
)

if [ "$STEP" = "train" ]; then
  python -u training/train_domain_adaptation.py --method "$METHOD" \
      --test ${FOLDS[$SLURM_ARRAY_TASK_ID]}
else
  for s in --summarize --replot --eval-d1; do
    python -u training/train_domain_adaptation.py $s --method "$METHOD"
  done
fi
