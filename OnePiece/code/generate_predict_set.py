#!/usr/bin/env python3
"""
Generate predict_set.jsonl from the candidate parquet file.

Usage:
    python3 generate_predict_set.py
"""
import json
import os

import pyarrow.parquet as pq

RAW_DATA = os.environ.get("RAW_DATA_PATH", "/scratch/cy2668/TAAC2025/data/TencentGR-1M")
OUTPUT = os.environ.get("USER_CACHE_PATH", "/scratch/cy2668/TAAC2025/OnePiece/user_cache")

candidate_dir = os.path.join(RAW_DATA, "candidate")
output_path = os.path.join(OUTPUT, "predict_set.jsonl")

print(f"Reading candidate parquet from {candidate_dir}...")
table = pq.read_table(candidate_dir)
print(f"  Rows: {table.num_rows}")

# Feature columns (same as base_item_sparse in main_dist.py)
FEAT_COLS = ['100', '101', '102', '112', '114', '115', '116', '117', '118', '119', '120', '121', '122']

print(f"Writing to {output_path}...")
count = 0
with open(output_path, 'w') as f:
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pydict()
        creative_id = int(row['item_id'][0])
        retrieval_id = int(row['retrieval_id'][0])

        features = {}
        for col in FEAT_COLS:
            if col in row:
                val = row[col][0]
                if val is not None:
                    val = val.as_py() if hasattr(val, 'as_py') else val
                    if isinstance(val, dict):
                        features[col] = val.get('feature_value', '0')
                    else:
                        features[col] = str(val)
                else:
                    features[col] = '0'

        line = json.dumps({
            "creative_id": creative_id,
            "retrieval_id": retrieval_id,
            "features": features
        })
        f.write(line + '\n')
        count += 1
        if count % 100000 == 0:
            print(f"  Written {count} rows...")

print(f"Done. Total: {count} rows written to {output_path}")
