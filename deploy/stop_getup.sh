#!/bin/bash
# Stop: 0.5 s damping frames, hand back to the native 'ai' service. WARNING: a standing robot will go limp.
cd ~/supermimic_getup_v10_20261126
[ -f run.pid ] && kill -TERM $(cat run.pid) 2>/dev/null
for i in $(seq 20); do kill -0 $(cat run.pid) 2>/dev/null || break; sleep 0.5; done
pkill -P 1 -x tail 2>/dev/null; rm -f run.pid; echo stopped
