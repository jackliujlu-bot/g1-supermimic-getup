"""Pick the BONES-SEED clips used for training (data/bones50/keep.csv), from convert_bones.py's stats.csv
and the BONES-SEED metadata csv. Drops clips where the body is held up by a prop (chairs, boxes, ...),
very fast joints (> 35 rad/s) and clips shorter than 1 s.

Usage: python filter_bones.py STATS_CSV SEED_METADATA_CSV OUT_KEEP_CSV
(run 2026-11-23: keep 134,603 clips, 274.5 h)
"""
import sys

import pandas as pd

s = pd.read_csv(sys.argv[1])
m = pd.read_csv(sys.argv[2])
m["rel"] = m["move_g1_path"].str.replace("g1/csv/", "", regex=False)
s = s.merge(m[["rel", "package", "category", "content_props", "content_body_position", "move_name", "is_mirror"]], on="rel")
props = s.content_props.astype(str).str.lower()
seat = props.str.contains(r"chair|bench|stool|bed|sofa|couch|table|ladder|stairs|scaffold|box|crate|wall|car|bike|horse|treadmill") & (s.support_s > 0.3)
seat_pos = s.content_body_position.astype(str).str.contains("chair|on box|on table|bench|stool|sitting on (a )?(chair|box)", regex=True)
sit_chair = props.str.contains("chair|bench|stool|sofa|couch") & s.content_body_position.astype(str).str.contains("sitting")
drop = (s.support_s > 0.5) | seat | seat_pos | sit_chair | (s.max_jvel > 35) | (s.dur_s < 1.0)
k = s[~drop]
print("keep", len(k), "hours", round(k.dur_s.sum() / 3600, 2), "drop", int(drop.sum()))
k[["rel", "move_name", "package", "category", "dur_s", "zmin_min", "is_mirror"]].to_csv(sys.argv[3], index=False)
