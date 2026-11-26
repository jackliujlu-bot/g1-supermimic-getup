#!/bin/bash
# usage: fp_eval.sh ONNX OUTDIR TRIALS   -> footprint (no walls) + narrow box (walls +0.15 m)
cd /path/to/workspace/deploy
O=$2; N=${3:-5}; mkdir -p $O
for s in supine prone side sitting random; do
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ../supermimic/.venv/bin/python sim2sim.py --getup $1 --trials $N --scenarios $s --out $O/open_$s.json > $O/open_$s.log 2>&1 &
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ../supermimic/.venv/bin/python sim2sim.py --getup $1 --trials $N --scenarios $s --walls 0.15 --out $O/wall_$s.json > $O/wall_$s.log 2>&1 &
done
wait
../supermimic/.venv/bin/python - $O <<'PY'
import json,glob,sys,numpy as np
O=sys.argv[1]
for tag in ("open","wall"):
  r=[x for f in sorted(glob.glob(f"{O}/{tag}_*.json")) for x in json.load(open(f))["results"]]
  m=lambda k: f"{np.mean([x[k] for x in r if x[k] is not None]):.2f}/{np.max([x[k] for x in r if x[k] is not None]):.2f}"
  hits=sum(x["wall_hit_ticks"]>0 for x in r)
  print(f"[{tag}] ok {sum(x['ok'] for x in r)}/{len(r)}  t_stand {m('t_stand')}  fp_r0 {m('fp_r0')}  fp_excess(m) {m('fp_excess')}  "
        f"arm_span {m('arm_span')}  splay {m('splay_span')}  qd {m('max_qd_ratio')}  sat {m('sat_frac')}"
        + (f"  wall_hit_trials {hits}/{len(r)} wall_force_bw {m('wall_peak_force_bw')}" if tag=="wall" else ""))
PY
