#!/bin/bash
# usage: snap_loop.sh NAME RUN_DIR  (snapshots each new checkpoint while logs/NAME.pid runs)
W=/path/to/workspace; cd $W/supermimic
NAME=$1; R=$2; LOG=$W/logs/$NAME.log; SNAP=$W/snapshots/$NAME; PID=$(cat $W/logs/$NAME.pid)
echo "run dir $R" > $SNAP/RUN_DIR; last=""
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
