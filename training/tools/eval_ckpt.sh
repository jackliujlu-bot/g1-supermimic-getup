#!/bin/bash
# Evaluate one tracker checkpoint dir (actor.pt + world_model.pt):
#   dances (94 eval dances x 3 rounds, mjlab, hard floor, same as the 99.84% baseline),
#   get-up from 5 fallen poses on hard floor and on a soft mat (deploy stack, sim2sim),
#   get-up from the real robot's logged poses (hard / soft), dance + push.
# usage: eval_ckpt.sh CKPT_DIR OUT_DIR [GPU]
W=/path/to/workspace; mkdir -p $2; C=$(readlink -f $1); O=$(readlink -f $2); G=${3:-2}; cd $W
PY=$W/supermimic/.venv/bin/python
CUDA_VISIBLE_DEVICES=$G $PY tools/export_tracker.py $C exports/refslow/getup_s88.npz $O/tracker.onnx > $O/export.log 2>&1
(cd supermimic && CUDA_VISIBLE_DEVICES=$G $PY ../tools/eval_tracker.py $C ../data/eval_dances.txt --envs 1024 --rounds 3 --out $O/dances.json > $O/dances.log 2>&1) &
cd deploy; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
SETTLE=${SETTLE:-$W/exports/settle_final/settle.onnx}; U="$O/tracker.onnx $SETTLE motions/getup_supine.npz"
for s in supine prone side sitting random; do
  $PY sim2sim.py --getup x --unified $U --human-tilt 180 --scenarios $s --trials 8 --limit 25 --out $O/hard_$s.json > /dev/null 2>&1 &
  $PY sim2sim.py --getup x --unified $U --human-tilt 180 --scenarios $s --trials 8 --limit 25 --soft 1.0 --out $O/soft_$s.json > /dev/null 2>&1 &
done
for f in "" "--soft 0.6" "--soft 1.0"; do
  $PY real_replay_test.py /tmp/live1_init_q.npy /tmp/live1_stuck.npy /tmp/live2_track_start.npy --trials 6 --tracker $O/tracker.onnx --settle $SETTLE $f > "$O/replay${f// /_}.txt" 2>/dev/null &
done
for d in victory_dance_macarena_change_direction_R_001__A321 dance_hiphop_hip_hop_iv_R_001__A320 dance_latino_jive_sequence_R_002__A312; do
  $PY sim2sim.py --getup x --unified $U --human-tilt 180 --scenarios standing --trials 4 --dance motions/$d.npz --out $O/dance_$d.json > /dev/null 2>&1 &
done
wait
$PY - $O <<'PY'
import json, glob, sys
O = sys.argv[1]
d = json.load(open(f"{O}/dances.json")) if glob.glob(f"{O}/dances.json") else {}
print("DANCES success", d.get("success_rate"), "episodes", d.get("episodes"), "joint_err", d.get("joint_pos_err_rad"))
for tag in ("hard", "soft"):
  r = [x for f in sorted(glob.glob(f"{O}/{tag}_*.json")) for x in json.load(open(f))["results"]]
  print(f"GETUP {tag}: {sum(x['ok'] for x in r)}/{len(r)}",
        {f.split('_')[-1][:-5]: sum(x['ok'] for x in json.load(open(f))['results']) for f in sorted(glob.glob(f'{O}/{tag}_*.json'))})
r = [x for f in glob.glob(f"{O}/dance_*.json") for x in json.load(open(f))["results"] if x["scenario"].startswith("dance")]
print("DANCE+PUSH", sum(x["ok"] for x in r), "/", len(r))
for f in sorted(glob.glob(f"{O}/replay*.txt")):
  print(f.split("/")[-1], [l.split(" t ")[0] for l in open(f) if "stood" in l])
PY
