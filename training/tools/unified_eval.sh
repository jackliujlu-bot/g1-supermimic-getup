#!/bin/bash
# Full evaluation of the unified (one tracker) stack vs the old stack. usage: unified_eval.sh OUTDIR SETTLE.onnx
W=/path/to/workspace
O=$(readlink -f $1); S=$(readlink -f $2); mkdir -p $O; cd $W/deploy
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
P=../supermimic/.venv/bin/python; G=../exports/getup_v3_10000/getup.onnx
U="../final/policies/tracker.onnx $S motions/getup_supine.npz"
D=(macarena:victory_dance_macarena_change_direction_R_001__A321 hiphop:dance_hiphop_hip_hop_iv_R_001__A320 jive:dance_latino_jive_sequence_R_002__A312)
for s in supine prone side sitting random standing; do
  $P sim2sim.py --getup $G --unified $U --human-tilt 180 --scenarios $s --trials 10 --limit 25 --out $O/open_$s.json > /dev/null 2>&1 &
done
for s in supine prone side sitting random; do
  $P sim2sim.py --getup $G --unified $U --human-tilt 180 --scenarios $s --trials 5 --limit 25 --walls 0.15 --out $O/box15_$s.json > /dev/null 2>&1 &
done
for d in "${D[@]}"; do n=${d%%:*}; c=${d#*:}
  $P sim2sim.py --getup $G --unified $U --human-tilt 180 --scenarios standing --trials 5 --dance motions/$c.npz --out $O/dance_$n.json > /dev/null 2>&1 &
done
$P hold_test.py --getup $G --unified $U --seconds 60 --trials 5 > $O/hold_unified.jsonl 2>/dev/null &
$P hold_test.py --getup $G --unified $U --seconds 60 --push 300 --trials 5 > $O/hold_unified_push300.jsonl 2>/dev/null &
$P hold_test.py --getup $G --seconds 60 --trials 5 > $O/hold_v3.jsonl 2>/dev/null &
$P hold_test.py --getup $G --seconds 60 --push 300 --trials 5 > $O/hold_v3_push300.jsonl 2>/dev/null &
for d in "${D[@]}"; do n=${d%%:*}; c=${d#*:}
  $P hold_test.py --getup $G --unified $U --dance motions/$c.npz --seconds 5 --trials 5 > $O/switch_unified_$n.jsonl 2>/dev/null &
done
wait
echo done
