#!/bin/bash
# v10 get-up only (no dances). Starts in DAMPING; engage with 'g' (echo g >> cmds.txt) or remote START/A.
# Damping: echo q >> cmds.txt, or remote L2+B. Stop: bash stop_getup.sh
cd ~/supermimic_getup_v10_20261126
if [ -f run.pid ] && kill -0 $(cat run.pid) 2>/dev/null; then echo "already running (pid $(cat run.pid))"; exit 1; fi
TS=$(date +%Y%m%d_%H%M%S); echo $TS > current_run; : > cmds.txt
export PYTHONPATH=/home/unitree/supermimic_dance_20261103_f8f60d7/python_deps
setsid nohup bash -c "tail -f cmds.txt | exec python3 -u run_g1.py --backend real --iface eth0 --unified policies/tracker.onnx policies/settle_prone.onnx motions/lafan_prone.npz policies/settle_supine.onnx motions/lafan_supine.npz --getup-timeout 45 --diag --log live_$TS.npz" > live_$TS.log 2>&1 < /dev/null &
sleep 2
pgrep -n -f "^python3 -u run_g1.py" > run.pid
echo "started pid $(cat run.pid), log live_$TS.log"
