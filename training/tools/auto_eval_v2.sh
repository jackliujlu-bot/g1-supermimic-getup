#!/bin/bash
# After settle_soft finishes: export it; then evaluate every ft_v2 snapshot (new settle) as it appears.
cd /path/to/workspace
while kill -0 $(cat logs/settle_soft.pid) 2>/dev/null; do sleep 30; done
S=$(ls -d logs/rsl_rl/g1_getup/*settle_soft); L=$(ls $S | grep model_ | sort -V | tail -1)
CUDA_VISIBLE_DEVICES=2 supermimic/.venv/bin/python tools/export_getup.py $S/$L exports/settle_soft > logs/export_settle_soft.log 2>&1
mv exports/settle_soft/getup.onnx exports/settle_soft/settle.onnx; echo $L > exports/settle_soft/CHECKPOINT
export SETTLE=$PWD/exports/settle_soft/settle.onnx
bash tools/eval_ckpt.sh snapshots/ft_robust/1236_log317 exports/ckeval/ft3000_ssoft 2 > logs/ckeval_ft3000_ssoft.log 2>&1
done_list=""
while true; do
  for d in snapshots/ft_v2/*; do
    n=$(basename $d); case " $done_list " in *" $n "*) continue;; esac
    [ "$(ls $d | wc -l)" -lt 2 ] && continue
    sleep 30; bash tools/eval_ckpt.sh $d exports/ckeval/v2_$n 2 > logs/ckeval_v2_$n.log 2>&1; done_list="$done_list $n"
  done
  if ! kill -0 $(cat logs/ft_v2.pid) 2>/dev/null; then
    R=supermimic/logs/SuperMimic-GenCoord-G1-Robust/2026-11-24_14-59-48
    mkdir -p snapshots/ft_v2/final; cp $R/actor.pt $R/world_model.pt snapshots/ft_v2/final/
    bash tools/eval_ckpt.sh snapshots/ft_v2/final exports/ckeval/v2_final 2 > logs/ckeval_v2_final.log 2>&1
    break
  fi
  sleep 60
done
echo ALL_DONE
