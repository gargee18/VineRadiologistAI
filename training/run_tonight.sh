#!/bin/bash
# Waits for the D1 views to finish, then: soil lines -> DA training (3 folds)
# -> summary -> agreement on the held-out D1 vines. Everything is logged.
# Run from the repo root, in its own tmux window:  bash training/run_tonight.sh
set -eo pipefail
cd /mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/gargee/code_python/VineRadiologistAI
source xrproj_env/bin/activate
V=/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/gargee/d1_views
LOG=results/domain_adaptation/tonight.log
mkdir -p results/domain_adaptation

echo "$(date)  waiting for 2000 D1 views and the end of make_views" | tee -a $LOG
while [ "$(ls $V/*/*_img.npy 2>/dev/null | wc -l)" -lt 2000 ] || pgrep -f "make_views.py --d1" > /dev/null; do
  sleep 60
done

echo "$(date)  soil lines" | tee -a $LOG
python -u training/make_views.py --d1 --soil 2>&1 | tee -a $LOG

echo "$(date)  DA training, 3 folds" | tee -a $LOG
for i in 1 2 3; do                       # restart after a crash; finished folds are skipped
  python -u training/train_domain_adaptation.py --folds 2>&1 | tee -a $LOG && break
  echo "$(date)  training crashed, restart $i" | tee -a $LOG
  [ "$i" -eq 3 ] && exit 1
  sleep 30
done
python -u training/train_domain_adaptation.py --summarize 2>&1 | tee -a $LOG
python -u training/train_domain_adaptation.py --eval-d1 2>&1 | tee -a $LOG
echo "$(date)  all done" | tee -a $LOG