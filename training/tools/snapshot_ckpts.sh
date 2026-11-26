#!/bin/bash
# Copy each new SuperMimic checkpoint (overwritten in place by the runner) to snapshots/<time>_step<N>/
R=$1; OUT=$2; LOG=$3; mkdir -p $OUT; last=""
while kill -0 $(cat logs/main.pid) 2>/dev/null; do
  m=$(stat -c %Y $R/actor.pt 2>/dev/null)
  if [ -n "$m" ] && [ "$m" != "$last" ]; then
    sleep 20
    it=$(grep -c "Learning iteration" $LOG)
    d=$OUT/$(date +%H%M)_log${it}; mkdir -p $d; cp $R/actor.pt $R/world_model.pt $d/; last=$m; echo "snap $d"
  fi
  sleep 30
done
