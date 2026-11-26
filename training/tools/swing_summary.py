"""Per floor: get-up ok, ended upright, and standing joint swing (deg, last 1 s) from an eval_v3 output dir."""
import glob, json, sys
import numpy as np
O = sys.argv[1]
for f in ("hard", "soft", "rough", "roughsoft"):
  r = [x for p in glob.glob(f"{O}/getup_{f}_*.json") for x in json.load(open(p))["results"]]
  if not r:
    continue
  sw = [x["end_swing_deg"] for x in r if x.get("upright_end") and x.get("end_swing_deg") is not None]
  calm = sum(1 for x in r if x.get("upright_end") and (x.get("end_swing_deg") or 99) <= 5.0)
  print(f"{f:9s} ok {sum(x['ok'] for x in r)}/{len(r)}  upright {sum(x['upright_end'] for x in r)}/{len(r)}  "
        f"upright&swing<=5deg {calm}/{len(r)}  swing median {np.median(sw) if sw else float('nan'):.1f} max {max(sw) if sw else float('nan'):.1f}")
