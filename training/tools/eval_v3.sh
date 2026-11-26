#!/bin/bash
# Evaluate one v3 tracker checkpoint (actor.pt + world_model.pt) with the two LAFAN hand-supported get-ups:
#   dances (94 eval dances x 3 rounds, mjlab, flat hard floor);
#   deploy stack in sim2sim (CPU MuJoCo): get-up from 5 fallen poses x 6 trials on 4 floors
#   (hard, soft mat u=1.0, uneven paving 1.5 cm, uneven + soft 0.6); dance + push x 3 dances on 3 floors.
# usage: SETTLE_PRONE=.. SETTLE_SUPINE=.. eval_v3.sh CKPT_DIR OUT_DIR [GPU]
W=/path/to/workspace; mkdir -p $2; C=$(readlink -f $1); O=$(readlink -f $2); G=${3:-2}; cd $W
PY=$W/supermimic/.venv/bin/python; L=$W/data/lafan_getups
[ -f $O/tracker.onnx ] || CUDA_VISIBLE_DEVICES=$G $PY tools/export_tracker.py $C $L/lafan_prone.npz $O/tracker.onnx > $O/export.log 2>&1
[ -n "$SKIP_DANCES" ] || (cd supermimic && CUDA_VISIBLE_DEVICES=$G $PY ../tools/eval_tracker.py $C ../data/eval_dances.txt --envs 1024 --rounds 3 --out $O/dances.json > $O/dances.log 2>&1) &
cd deploy; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 G1_STAND_CLIP=${G1_STAND_CLIP:-$L/lafan_prone.npz}
U="$O/tracker.onnx ${SETTLE_PRONE:?} $L/lafan_prone.npz ${SETTLE_SUPINE:?} $L/lafan_supine.npz"
declare -A FLOORS=([hard]="" [soft]="--soft 1.0" [rough]="--terrain 0.015" [roughsoft]="--terrain 0.015 --soft 0.6")
for f in "${!FLOORS[@]}"; do
  for s in supine prone side sitting random; do
    $PY sim2sim.py --getup x --unified $U --human-tilt 180 --scenarios $s --trials 6 --limit ${LIMIT:-40} ${FLOORS[$f]} --out $O/getup_${f}_$s.json > /dev/null 2>&1 &
  done
done
for f in hard soft rough; do
  for d in victory_dance_macarena_change_direction_R_001__A321 dance_hiphop_hip_hop_iv_R_001__A320 dance_latino_jive_sequence_R_002__A312; do
    $PY sim2sim.py --getup x --unified $U --human-tilt 180 --scenarios standing --trials 3 ${FLOORS[$f]} --dance motions/$d.npz --out $O/dance_${f}_$d.json > /dev/null 2>&1 &
  done
done
wait
$PY - $O <<'PY'
import json, glob, sys
O = sys.argv[1]
d = json.load(open(f"{O}/dances.json")) if glob.glob(f"{O}/dances.json") else {}
print("DANCES(mjlab, flat)", d.get("success_rate"), "episodes", d.get("episodes"), "joint_err", d.get("joint_pos_err_rad"))
for f in ("hard", "soft", "rough", "roughsoft"):
  per = {}
  for p in sorted(glob.glob(f"{O}/getup_{f}_*.json")):
    r = json.load(open(p))["results"]
    per[p.split("_")[-1][:-5]] = f"{sum(x['ok'] for x in r)}/{len(r)}"
  tot = [x for p in glob.glob(f"{O}/getup_{f}_*.json") for x in json.load(open(p))["results"]]
  print(f"GETUP {f:9s} {sum(x['ok'] for x in tot)}/{len(tot)}", per)
for f in ("hard", "soft", "rough"):
  r = [x for p in glob.glob(f"{O}/dance_{f}_*.json") for x in json.load(open(p))["results"] if x["scenario"].startswith("dance")]
  print(f"DANCE+PUSH {f:5s} {sum(x['ok'] for x in r)}/{len(r)}")
PY
