#!/bin/bash
# run_weekend.sh  -  views -> 3 folds on stored views -> summary -> replots
#
# Start it in tmux, from anywhere:
#     tmux new -s weekend
#     bash training/run_weekend.sh
#     (Ctrl+B then D to leave it)
#
# Everything is logged in results/weekend_logs/. Check progress with
#     tail -3 results/weekend_logs/*.log

REPO=/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/gargee/code_python/VineRadiologistAI
VIEWS_DIR=/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/gargee/d2_views
N_VIEWS=1200          # 12 vines x 100
MAX_RETRIES=5         # restarts of make_views if it crashes

FOLD1="CEP011_AS1 CEP014_RES1 CEP017_S1 CEP020_APO1"
FOLD2="CEP012_AS2 CEP015_RES2 CEP018_S2 CEP021_APO2"
FOLD3="CEP013_AS3 CEP016_RES3 CEP019_S3 CEP022_APO3"

cd "$REPO" || exit 1
source xrproj_env/bin/activate
LOGS=results/weekend_logs
mkdir -p "$LOGS"
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOGS/weekend.log"; }

# ---- 1. views: wait for the running job, restart it if it crashed -------
count_views() { find "$VIEWS_DIR" -name '*_fg.npy' | wc -l; }   # _fg is written last

say "waiting for the running make_views to finish"
while pgrep -f make_views.py > /dev/null; do sleep 60; done

tries=0
while [ "$(count_views)" -lt "$N_VIEWS" ] && [ "$tries" -lt "$MAX_RETRIES" ]; do
    tries=$((tries + 1))
    say "views: $(count_views)/$N_VIEWS, (re)starting make_views, try $tries"
    python -u -X faulthandler training/make_views.py >> "$LOGS/make_views.log" 2>&1
done
if [ "$(count_views)" -lt "$N_VIEWS" ]; then
    say "STOP: only $(count_views)/$N_VIEWS views after $MAX_RETRIES tries, see $LOGS/make_views.log"
    exit 1
fi
say "views complete: $(count_views)"

# ---- 2. training, one fold after the other ------------------------------
i=0
for FOLD in "$FOLD1" "$FOLD2" "$FOLD3"; do
    i=$((i + 1))
    say "fold $i: test = $FOLD"
    if python -u training/train_baseline.py --stored --test $FOLD > "$LOGS/train_fold$i.log" 2>&1; then
        say "fold $i done"
    else
        say "fold $i FAILED (see $LOGS/train_fold$i.log), going on with the next fold"
    fi
done

# ---- 3. summary and plots ------------------------------------------------
say "summary and replots"
python -u training/train_baseline.py --stored --summarize > "$LOGS/summarize.log" 2>&1
python -u training/train_baseline.py --stored --replot    > "$LOGS/replot_stored.log" 2>&1
# old runs, redrawn with the new colours
python -u training/train_baseline.py --replot             > "$LOGS/replot_baseline.log" 2>&1
python -u training/train_baseline.py --augment --replot   > "$LOGS/replot_augmented.log" 2>&1
say "all done"