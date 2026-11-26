#!/bin/bash
# Launch a SuperMimic tracker run (robust task + uneven floors) and snapshot every new checkpoint.
# usage: train_v3.sh NAME GPU MAX_ITERS [extra finetune_tracker.py args...]
W=/path/to/workspace; cd $W/supermimic
NAME=$1; GPU=$2; IT=$3; shift 3
LOG=$W/logs/$NAME.log; SNAP=$W/snapshots/$NAME; mkdir -p $SNAP
before=$(ls -d logs/SuperMimic-GenCoord-G1-Robust/*/ 2>/dev/null | sort | tail -1)
G1R_TERRAIN=${G1R_TERRAIN:-1} CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline nohup .venv/bin/python -u ../tools/finetune_tracker.py \
  SuperMimic-GenCoord-G1-Robust --num-envs 4096 --motion-file ../data/pack_v3 \
  --runner.actor-obs proprio --runner.history-length 10 --runner.max-iterations $IT "$@" > $LOG 2>&1 &
PID=$!; echo $PID > $W/logs/$NAME.pid; echo "pid $PID"
# find the new run dir
for i in $(seq 120); do
  R=$(ls -d logs/SuperMimic-GenCoord-G1-Robust/*/ 2>/dev/null | sort | tail -1)
  [ "$R" != "$before" ] && [ -n "$R" ] && break; sleep 5; R=""
done
echo "run dir $R" | tee $SNAP/RUN_DIR
last=""
while kill -0 $PID 2>/dev/null; do
  m=$(stat -c %Y $R/actor.pt 2>/dev/null)
  if [ -n "$m" ] && [ "$m" != "$last" ]; then
    sleep 20
    it=$(grep -c "Learning iteration" $LOG)
    d=$SNAP/$(date +%m%d_%H%M)_log${it}; mkdir -p $d; cp $R/actor.pt $R/world_model.pt $d/; cp $R/critic.pt $d/ 2>/dev/null; last=$m
  fi
  sleep 30
done
echo "exited $(date)" >> $SNAP/RUN_DIR
