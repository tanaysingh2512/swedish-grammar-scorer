"""Combine the toy train.jsonl with data/synthetic.jsonl into one training set,
split train/val by `group` so a good/poor pair never straddles the split.

Usage: python data/merge_data.py
Writes data/train_full.jsonl and data/val.jsonl.
"""
import json
import random

SEED = 0

rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
rows += [json.loads(l) for l in open("data/synthetic.jsonl", encoding="utf-8")]

groups = sorted({r["group"] for r in rows})
rng = random.Random(SEED)
rng.shuffle(groups)
n_val = max(1, int(0.1 * len(groups)))
val_groups = set(groups[:n_val])

with open("data/train_full.jsonl", "w", encoding="utf-8") as ftr, \
     open("data/val.jsonl", "w", encoding="utf-8") as fva:
    n_tr = n_va = 0
    for r in rows:
        line = json.dumps(r, ensure_ascii=False) + "\n"
        if r["group"] in val_groups:
            fva.write(line)
            n_va += 1
        else:
            ftr.write(line)
            n_tr += 1

print(f"{len(rows)} total rows, {len(groups)} groups")
print(f"train: {n_tr} rows, val: {n_va} rows ({len(val_groups)} groups held out)")
