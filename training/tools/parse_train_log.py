"""Training text log (SuperMimic runner or RSL-RL PPO) -> CSV, one row per logged iteration.

Usage: python parse_train_log.py LOG OUT.csv
Columns: iter, every 'Key: number' printed in that iteration's block (losses, rewards, tracking errors,
'Time elapsed' converted to seconds as elapsed_s).
"""
import csv
import re
import sys

ansi = re.compile(r"\x1b\[[0-9;]*m")
it_re = re.compile(r"Learning iteration (\d+)/(\d+)")
kv_re = re.compile(r"^\s*([A-Za-z_][\w/ .-]*?):\s+(-?[\d.]+(?:e-?\d+)?)\s*$")
rows, cur = [], None
for line in open(sys.argv[1], errors="ignore"):
  line = ansi.sub("", line)
  m = it_re.search(line)
  if m:
    if cur:
      rows.append(cur)
    cur = {"iter": int(m.group(1))}
    continue
  if cur is None:
    continue
  if "Time elapsed:" in line:
    h, mi, s = line.split("Time elapsed:")[1].strip().split(":")
    cur["elapsed_s"] = int(h) * 3600 + int(mi) * 60 + float(s)
    continue
  m = kv_re.match(line)
  if m:
    cur[m.group(1).strip()] = float(m.group(2))
if cur:
  rows.append(cur)
keys = ["iter"] + sorted({k for r in rows for k in r} - {"iter"})
with open(sys.argv[2], "w", newline="") as f:
  w = csv.DictWriter(f, fieldnames=keys)
  w.writeheader()
  w.writerows(rows)
print(f"{sys.argv[1]}: {len(rows)} iterations, {len(keys)} columns -> {sys.argv[2]}")
