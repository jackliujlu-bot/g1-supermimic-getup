#!/bin/bash
# Tracker-only checks: export, 94 dances x 3 (mjlab), the 4 LAFAN get-up clips x 4 floors x 6 (sim2sim, UP / CALM).
# usage: quick_eval.sh CKPT_DIR OUT_DIR GPU
W=/path/to/workspace; C=$(readlink -f $1); mkdir -p $2; O=$(readlink -f $2); G=$3; cd $W
PY=$W/supermimic/.venv/bin/python; L=$W/data/lafan_getups
CUDA_VISIBLE_DEVICES=$G $PY tools/export_tracker.py $C $L/lafan_prone.npz $O/tracker.onnx > $O/export.log 2>&1
(cd supermimic && CUDA_VISIBLE_DEVICES=$G $PY ../tools/eval_tracker.py $C ../data/eval_dances.txt --envs 1024 --rounds 3 --out $O/dances.json > $O/dances.log 2>&1) &
cd deploy; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1; mkdir -p $O/clip2
for c in lafan_prone lafan_prone_x1.25 lafan_supine lafan_supine_x1.25; do
  for f in hard:"" soft:"--soft 1.0" rough:"--terrain 0.015" roughsoft:"--terrain 0.015 --soft 0.6"; do
    n=${f%%:*}; a=${f#*:}; $PY clip_track_test.py $O/tracker.onnx $L/$c.npz --trials 6 --hold 3 $a > $O/clip2/${c}_$n.txt 2>&1 &
  done
done
wait
$PY - $O <<"PY"
import json, glob, re, sys
O = sys.argv[1]
d = json.load(open(f"{O}/dances.json"))
print("DANCES", round(d["success_rate"], 4), "joint_err", round(d["joint_pos_err_rad"], 3))
up = calm = n = 0
for p in sorted(glob.glob(f"{O}/clip2/*.txt")):
  m = re.search(r"up (\d+)/(\d+) calm (\d+)/", open(p).read())
  up += int(m[1]); calm += int(m[3]); n += int(m[2])
  print(" ", p.split("/")[-1][:-4], m[0])
print(f"CLIPS up {up}/{n} calm {calm}/{n}")
PY
