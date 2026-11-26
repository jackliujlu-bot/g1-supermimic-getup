#!/bin/bash
# Full get-up evaluation: open floor (6 scenarios) + aisle (walls 0.15 m beyond the lying body on both sides)
# + box (walls on all four sides, 0.15 m / 0.30 m beyond the lying body).
# usage: getup_eval.sh <getup.onnx | checkpoint.pt> OUTDIR [TRIALS]
set -e
W=/path/to/workspace
cd $W
IN=$1; O=$2; N=${3:-5}; mkdir -p $O
if [[ $IN == *.pt ]]; then
  CUDA_VISIBLE_DEVICES=${GPU:-2} supermimic/.venv/bin/python tools/export_getup.py $IN $O 2>&1 | grep parity
  IN=$O/getup.onnx
fi
IN=$(readlink -f $IN); O=$(readlink -f $O)
cd deploy
run() { CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ../supermimic/.venv/bin/python sim2sim.py --getup $IN --trials $N "$@"; }
for s in supine prone side sitting random standing; do run --scenarios $s --out $O/open_$s.json > $O/open_$s.log 2>&1 & done
for s in supine prone side sitting random; do
  run --scenarios $s --corridor 0.15 --out $O/aisle15_$s.json > $O/aisle15_$s.log 2>&1 &
  run --scenarios $s --walls 0.15 --out $O/box15_$s.json > $O/box15_$s.log 2>&1 &
  run --scenarios $s --walls 0.30 --out $O/box30_$s.json > $O/box30_$s.log 2>&1 &
done
wait
../supermimic/.venv/bin/python - $O <<'PY'
import json, glob, sys
import numpy as np
O = sys.argv[1]
out = {}
for tag in ("open", "aisle15", "box15", "box30"):
  r = [x for f in sorted(glob.glob(f"{O}/{tag}_*.json")) for x in json.load(open(f))["results"]]
  fl = [x for x in r if x["scenario"] != "boot_standing"]
  def m(k, rr=fl):
    v = [x[k] for x in rr if x.get(k) is not None]
    return [round(float(np.mean(v)), 2), round(float(np.max(v)), 2)] if v else None
  s = {"ok": f"{sum(x['ok'] for x in r)}/{len(r)}", "t_stand": m("t_stand"), "fp_excess_m": m("fp_excess"),
       "sweep_m2": m("sweep_area_m2"), "limb_reach_m": m("limb_reach_m"),
       "arm_span": m("arm_span"), "splay_span": m("splay_span"), "qd_ratio": m("max_qd_ratio"), "sat": m("sat_frac")}
  if tag != "open":
    s["wall_hit"] = f"{sum(x['wall_hit_ticks'] > 0 for x in r)}/{len(r)}"
    s["aisle_w"] = m("aisle_width")
  out[tag] = s
  print(tag, json.dumps(s))
json.dump(out, open(f"{O}/summary.json", "w"), indent=1)
PY
