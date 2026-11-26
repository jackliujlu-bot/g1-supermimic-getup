"""Summarize clip_eval.sh output: python clip_summary.py OUT_DIR [subdir=clip3]"""
import collections
import glob
import re
import sys

O = sys.argv[1]
sub = sys.argv[2] if len(sys.argv) > 2 else "clip3"
tot = collections.Counter()
jit = collections.defaultdict(list)
for p in sorted(glob.glob(f"{O}/{sub}/*.txt")):
  m = re.search(r"up (\d+)/(\d+) calm (\d+)/\d+ jitter_rms ([\d.na]+)", open(p).read())
  if not m:
    print("unparsed", p)
    continue
  fl = p.split("_")[-1][:-4]
  tot["up"] += int(m[1]); tot["calm"] += int(m[3]); tot["n"] += int(m[2])
  tot["up_" + fl] += int(m[1]); tot["n_" + fl] += int(m[2])
  if m[4] != "nan":
    jit[fl].append(float(m[4]))
fl = ["hard", "soft", "rough", "roughsoft"]
parts = [f"{f} {tot['up_' + f]}/{tot['n_' + f]} jit {sum(jit[f]) / max(1, len(jit[f])):.2f}" for f in fl]
print(f"CLIPS up {tot['up']}/{tot['n']} calm {tot['calm']}/{tot['n']} | " + " ".join(parts))
