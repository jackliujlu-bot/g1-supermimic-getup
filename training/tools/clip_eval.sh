#!/bin/bash
# The 4 LAFAN get-up clips x 4 floors x 6 trials with an exported tracker (UP / CALM / jitter rms).
# usage: clip_eval.sh TRACKER.onnx OUT_DIR
W=/path/to/workspace; T=$(readlink -f $1); mkdir -p $2/clip3; O=$(readlink -f $2)
PY=$W/supermimic/.venv/bin/python; L=$W/data/lafan_getups; cd $W/deploy; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
for c in lafan_prone lafan_prone_x1.25 lafan_supine lafan_supine_x1.25; do
  for f in hard:"" soft:"--soft 1.0" rough:"--terrain 0.015" roughsoft:"--terrain 0.015 --soft 0.6"; do
    n=${f%%:*}; a=${f#*:}; $PY clip_track_test.py $T $L/$c.npz --trials 6 --hold 3 $a > $O/clip3/${c}_$n.txt 2>&1 &
  done
done
wait
$PY $W/tools/clip_summary.py $O clip3
