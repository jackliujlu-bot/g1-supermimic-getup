#!/bin/bash
# Evaluate every new snapshot of the running fine-tune (and the final one), one at a time.
cd /path/to/workspace
done_list=""
while true; do
  for d in snapshots/ft_robust2/*; do
    n=$(basename $d); [ "$n" = "1355_log5" ] && continue
    case " $done_list " in *" $n "*) continue;; esac
    sleep 30; bash tools/eval_ckpt.sh $d exports/ckeval/ft2_$n 3 > logs/ckeval_ft2_$n.log 2>&1; done_list="$done_list $n"
  done
  kill -0 $(cat logs/ft_robust2.pid) 2>/dev/null || { [ -z "$(ls snapshots/ft_robust2 | grep -v -x -e 1355_log5 $(for x in $done_list; do echo -e "-e $x"; done))" ] && break; }
  sleep 60
done
R=supermimic/logs/SuperMimic-GenCoord-G1-Robust/2026-11-24_13-54-28
mkdir -p snapshots/ft_robust2/final; cp $R/actor.pt $R/world_model.pt snapshots/ft_robust2/final/
bash tools/eval_ckpt.sh snapshots/ft_robust2/final exports/ckeval/ft2_final 3 > logs/ckeval_ft2_final.log 2>&1
echo ALL_DONE
