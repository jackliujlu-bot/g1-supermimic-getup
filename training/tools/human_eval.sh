#!/bin/bash
# Human-style get-up evaluation (settle policy -> tracker on the human clip -> v3 stand hold).
# usage: human_eval.sh SETTLE.onnx OUTDIR [TRIALS] [TRACK.onnx] [CLIP.npz]
set -e
W=/path/to/workspace
S=$(readlink -f $1); O=$2; N=${3:-5}
T=$(readlink -f ${4:-$W/exports/refslow/getup_supine_human.onnx}); C=$(readlink -f ${5:-$W/exports/refslow/getup_supine_human.npz})
G=$W/exports/getup_v3_10000/getup.onnx
mkdir -p $O; O=$(readlink -f $O); cd $W/deploy
run() { CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ../supermimic/.venv/bin/python sim2sim.py --getup $G --human $S $T $C --limit 30 --trials $N "$@"; }
for s in supine prone side sitting random standing; do run --scenarios $s --out $O/open_$s.json > $O/open_$s.log 2>&1 & done
for s in supine prone side sitting random; do
  run --scenarios $s --walls 0.15 --out $O/box15_$s.json > $O/box15_$s.log 2>&1 &
  run --scenarios $s --walls 0.30 --out $O/box30_$s.json > $O/box30_$s.log 2>&1 &
done
wait
../supermimic/.venv/bin/python - $O <<'PY'
import json, glob, sys, collections
import numpy as np
O = sys.argv[1]; out = {}
for tag in ("open", "box15", "box30"):
  r = [x for f in sorted(glob.glob(f"{O}/{tag}_*.json")) for x in json.load(open(f))["results"]]
  fl = [x for x in r if x["scenario"] != "boot_standing"]
  def m(k, rr=fl):
    v = [x[k] for x in rr if x.get(k) is not None]
    return [round(float(np.mean(v)), 2), round(float(np.max(v)), 2)] if v else None
  paths = collections.Counter(x.get("path", "") for x in r)
  s = {"ok": f"{sum(x['ok'] for x in r)}/{len(r)}", "t_stand": m("t_stand"), "sweep_m2": m("sweep_area_m2"),
       "arm_span": m("arm_span"), "splay_span": m("splay_span"), "qd_ratio": m("max_qd_ratio"), "sat": m("sat_frac"),
       "paths": dict(paths)}
  if tag != "open":
    s["wall_hit"] = f"{sum(x['wall_hit_ticks'] > 0 for x in r)}/{len(r)}"
  fails = [(x["scenario"], x["seed"], x.get("path")) for x in r if not x["ok"]]
  s["fails"] = fails
  out[tag] = s
  print(tag, json.dumps(s))
json.dump(out, open(f"{O}/summary.json", "w"), indent=1)
PY
