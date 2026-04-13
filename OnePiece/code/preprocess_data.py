#!/usr/bin/env python3
"""
Data Preprocessing Script for OnePiece Model Training
=====================================================
Converts raw parquet data from TencentGR-1M into the exact formats
required by OnePiece's dataset.py, item_exposure_data.py, and timestamp_buckets.py.

Input:  /scratch/cy2668/TAAC2025/data/TencentGR-1M/  (parquet files)
Output:TRAIN_DATA_PATH directory with:
  - seq.jsonl           : user sequence data (one line per user)
  - seq_offsets.pkl     : user_reid -> byte offset in seq.jsonl
  - item_feat_dict.json : str(item_reid) -> {feat_id: feat_value, ...}
  - user_action_type.json : user_reid -> last_action_type
  - indexer.pkl         : copied from raw data
  - creative_emb/       : mm_emb data in OnePiece format
  - predict_seq.jsonl   : empty placeholder (for test dataset)
  - predict_seq_offsets.pkl : empty placeholder
"""

import os
import sys
import json
import pickle
import time
import argparse
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Ensure pyarrow is importable (may be installed in a custom location)
# ---------------------------------------------------------------------------
try:
    import pyarrow.parquet as pq
except ImportError:
    _custom = "/scratch/cy2668/TAAC2025/pylibs"
    sys.path.insert(0, _custom)
    import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="OnePiece data preprocessing")
    parser.add_argument("--raw_data_path", type=str,
                        default="/scratch/cy2668/TAAC2025/data/TencentGR-1M",
                        help="Path to raw TencentGR-1M parquet data")
    parser.add_argument("--output_path", type=str,
                        default="/scratch/cy2668/TAAC2025/OnePiece/data",
                        help="Output TRAIN_DATA_PATH directory")
    parser.add_argument("--user_cache_path", type=str,
                        default="/scratch/cy2668/TAAC2025/OnePiece/user_cache",
                        help="Output USER_CACHE_PATH directory")
    parser.add_argument("--skip_item_feat", action="store_true",
                        help="Skip item_feat_dict.json generation")
    parser.add_argument("--skip_seq", action="store_true",
                        help="Skip seq.jsonl generation")
    parser.add_argument("--skip_mm_emb", action="store_true",
                        help="Skip mm_emb conversion")
    parser.add_argument("--skip_exposure", action="store_true",
                        help="Skip item_exposure_data.py execution")
    parser.add_argument("--skip_timestamp", action="store_true",
                        help="Skip timestamp_buckets.py execution")
    return parser.parse_args()


# ===========================================================================
# 1. Build item_feat_dict.json
# ===========================================================================
def build_item_feat_dict(raw_data_path: Path, user_cache_path: Path):
    """
    Read all item_feat parquet files and produce item_feat_dict.json.

    Output format:
        {str(item_reid): {feat_id: feat_value, ...}, ...}

    Feature IDs in the parquet are: '100','101','102','112','114','115',
    '116','117','118','119','120','121','122'.
    None values are skipped.
    """
    print("=" * 60)
    print("[1/5] Building item_feat_dict.json ...")
    print("=" * 60)
    start = time.time()

    item_feat_dir = raw_data_path / "item_feat"
    parquet_files = sorted(item_feat_dir.glob("*.parquet"))
    print(f"  Found {len(parquet_files)} item_feat parquet files")

    item_feat_dict = {}
    feat_cols = ['100', '101', '102', '112', '114', '115', '116', '117',
                 '118', '119', '120', '121', '122']

    for pf in parquet_files:
        table = pq.read_table(pf)
        # Read columns in batch for efficiency
        item_ids = table.column("item_id").to_pylist()
        col_data = {}
        for col in feat_cols:
            col_data[col] = table.column(col).to_pylist()

        for row_idx in range(len(item_ids)):
            iid = item_ids[row_idx]
            feat = {}
            for col in feat_cols:
                val = col_data[col][row_idx]
                if val is not None:
                    feat[col] = val
            if feat:  # only store if there is at least one feature
                item_feat_dict[str(iid)] = feat

        print(f"  Processed {pf.name}, total items so far: {len(item_feat_dict)}")

    # Save
    user_cache_path.mkdir(parents=True, exist_ok=True)
    out_file = user_cache_path / "item_feat_dict.json"
    print(f"  Saving item_feat_dict.json ({len(item_feat_dict)} items) ...")
    with open(out_file, 'w') as f:
        json.dump(item_feat_dict, f)

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s, size: {out_file.stat().st_size / 1024 / 1024:.1f} MB")
    return item_feat_dict


# ===========================================================================
# 2. Build user_feat_dict (in-memory) for use during seq.jsonl generation
# ===========================================================================
def load_user_feats(raw_data_path: Path):
    """
    Load all user features from parquet into a dict: user_reid -> {feat_id: value}.

    User feature columns: '103','104','105','106','107','108','109','110'
    Some are list type (106,107,108,110), others are scalar.
    """
    print("  Loading user features from parquet ...")
    start = time.time()

    user_feat_dir = raw_data_path / "user_feat"
    parquet_files = sorted(user_feat_dir.glob("*.parquet"))

    user_feat_dict = {}
    feat_cols = ['103', '104', '105', '106', '107', '108', '109', '110']

    for pf in parquet_files:
        table = pq.read_table(pf)
        user_ids = table.column("user_id").to_pylist()
        col_data = {}
        for col in feat_cols:
            col_data[col] = table.column(col).to_pylist()

        for row_idx in range(len(user_ids)):
            uid = user_ids[row_idx]
            feat = {}
            for col in feat_cols:
                val = col_data[col][row_idx]
                if val is not None:
                    feat[col] = val
            user_feat_dict[uid] = feat

        print(f"    Processed {pf.name}, total users: {len(user_feat_dict)}")

    elapsed = time.time() - start
    print(f"  User features loaded: {len(user_feat_dict)} users in {elapsed:.1f}s")
    return user_feat_dict


# ===========================================================================
# 3. Build item_feat_dict (in-memory) for use during seq.jsonl generation
# ===========================================================================
def load_item_feats(raw_data_path: Path):
    """
    Load item features into a dict: item_reid -> {feat_id: value}.
    Same as item_feat_dict.json but as Python dict for quick lookup.
    """
    print("  Loading item features from parquet ...")
    start = time.time()

    item_feat_dir = raw_data_path / "item_feat"
    parquet_files = sorted(item_feat_dir.glob("*.parquet"))

    item_feat_dict = {}
    feat_cols = ['100', '101', '102', '112', '114', '115', '116', '117',
                 '118', '119', '120', '121', '122']

    for pf in parquet_files:
        table = pq.read_table(pf)
        item_ids = table.column("item_id").to_pylist()
        col_data = {}
        for col in feat_cols:
            col_data[col] = table.column(col).to_pylist()

        for row_idx in range(len(item_ids)):
            iid = item_ids[row_idx]
            feat = {}
            for col in feat_cols:
                val = col_data[col][row_idx]
                if val is not None:
                    feat[col] = val
            item_feat_dict[iid] = feat

    elapsed = time.time() - start
    print(f"  Item features loaded: {len(item_feat_dict)} items in {elapsed:.1f}s")
    return item_feat_dict


# ===========================================================================
# 4. Build seq.jsonl, seq_offsets.pkl, user_action_type.json
# ===========================================================================
def build_seq_files(raw_data_path: Path, output_path: Path,
                    user_feat_dict: dict, item_feat_dict: dict,
                    indexer_u_rev: dict):
    """
    Read seq parquet files and produce:
      - seq.jsonl: each line is a JSON list of records
        [ [user_reid, item_reid, user_feat_dict, item_feat_dict, action_type, timestamp], ... ]
        Format per record:
          - User placeholder: [user_reid, None, {user_feats}, None, action_type, timestamp]
          - Item record:      [None, item_reid, None, {item_feats}, action_type, timestamp]
        (Following dataset.py's _load_user_data format)

      - seq_offsets.pkl: {sequential_index: byte_offset_in_seq_jsonl}
        Keys are 0, 1, 2, ..., N-1 because PyTorch DataLoader passes
        sequential indices (0..len(dataset)-1) to __getitem__(uid),
        which then calls self.seq_offsets[uid].

      - user_action_type.json: {original_user_id_str: last_action_type}
        Keys are original user IDs (strings like "user_00953188") because
        dataset.py accesses this via indexer_u_rev[reid] which returns
        the original string ID.

    The seq data from parquet has:
      - user_id: int (already a reid, NOT original string id)
      - seq: list of {item_id: int(reid), action_type: int, timestamp: int}
    """
    print("=" * 60)
    print("[2/5] Building seq.jsonl, seq_offsets.pkl, user_action_type.json ...")
    print("=" * 60)
    start = time.time()

    seq_dir = raw_data_path / "seq"
    parquet_files = sorted(seq_dir.glob("*.parquet"))
    print(f"  Found {len(parquet_files)} seq parquet files")

    output_path.mkdir(parents=True, exist_ok=True)
    seq_jsonl_path = output_path / "seq.jsonl"
    offsets_path = output_path / "seq_offsets.pkl"
    user_action_path = output_path / "user_action_type.json"

    seq_offsets = {}
    user_action_type = {}

    total_users = 0
    total_records = 0

    with open(seq_jsonl_path, 'wb') as fout:
        for pf in parquet_files:
            table = pq.read_table(pf)
            user_ids = table.column("user_id").to_pylist()
            seqs = table.column("seq").to_pylist()

            for row_idx in range(len(user_ids)):
                uid = user_ids[row_idx]
                seq = seqs[row_idx]  # list of {item_id, action_type, timestamp}

                # Build the record list for this user
                records = []

                # First record: user placeholder with user features
                u_feat = user_feat_dict.get(uid)
                if seq:
                    first_action = seq[0].get('action_type', 0)
                    first_ts = seq[0].get('timestamp', 0)
                else:
                    first_action = 0
                    first_ts = 0

                # User placeholder row: (user_reid, None, user_feat, None, action_type, timestamp)
                # dataset.py checks: if u: (truthy), and then if user_feat or not item_feat
                # It inserts user row at the beginning
                user_record = [uid, None, u_feat, None, first_action, first_ts]
                records.append(user_record)

                # Item records
                last_action = 0
                for event in seq:
                    item_id = event['item_id']      # already reid
                    action_type_val = event['action_type']
                    timestamp = event['timestamp']

                    i_feat = item_feat_dict.get(item_id)
                    # Item record: (None, item_reid, None, item_feat, action_type, timestamp)
                    item_record = [None, item_id, None, i_feat, action_type_val, timestamp]
                    records.append(item_record)
                    last_action = action_type_val
                    total_records += 1

                # Record the byte offset before writing.
                # Use sequential index (0, 1, 2, ...) as key because PyTorch
                # DataLoader passes sequential indices to __getitem__(uid).
                offset = fout.tell()
                seq_offsets[total_users] = offset

                # Write as JSON line
                line = json.dumps(records, ensure_ascii=False) + '\n'
                fout.write(line.encode('utf-8'))

                # Track last action type for this user.
                # Key must be original user_id string (e.g., "user_00953188")
                # because dataset.py accesses via: self.user_action_type[user_id]
                # where user_id = self.indexer_u_rev[u] (reid -> original str)
                original_uid = indexer_u_rev.get(uid)
                if original_uid is not None:
                    user_action_type[original_uid] = last_action

                total_users += 1

            print(f"  Processed {pf.name}, cumulative users: {total_users}")

    # Save seq_offsets.pkl
    print(f"  Saving seq_offsets.pkl ({len(seq_offsets)} users) ...")
    with open(offsets_path, 'wb') as f:
        pickle.dump(seq_offsets, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save user_action_type.json
    print(f"  Saving user_action_type.json ({len(user_action_type)} users) ...")
    with open(user_action_path, 'wb') as f:
        f.write(json.dumps(user_action_type).encode('utf-8'))

    elapsed = time.time() - start
    seq_size = seq_jsonl_path.stat().st_size / 1024 / 1024 / 1024
    print(f"  Done in {elapsed:.1f}s")
    print(f"  seq.jsonl size: {seq_size:.2f} GB")
    print(f"  Total users: {total_users}, total item records: {total_records}")


# ===========================================================================
# 5. Convert mm_emb parquet to OnePiece's creative_emb format
# ===========================================================================
def convert_mm_emb(raw_data_path: Path, output_path: Path):
    """
    Convert mm_emb parquet files into the format expected by load_mm_emb()
    in dataset.py.

    load_mm_emb() expects:
      For feat_id != '81' (or debug=True for '81'):
        Path: creative_emb/emb_{feat_id}_{shape}/*.json
        Each json file: one JSON object per line
          {"anonymous_cid": str(original_item_id), "emb": [float, ...]}
      For feat_id == '81' and not debug:
        Path: creative_emb/emb_81_{shape}.pkl
        A pickle of {original_item_id: np.array(float16)}

    We convert all to the JSON line format since the debug flag is False by default
    for the actual training. However, the code in run.sh copies the data as-is,
    and dataset.py handles the loading.

    Since the raw data has OID (original item IDs as strings) and the code looks
    up by OID, we keep the original IDs. The indexer_i_rev maps reid -> OID,
    so the lookup works.

    For feat_id '81', the code tries to load a pkl file. We will convert parquet
    to pkl for '81' and to JSON lines for others.
    """
    print("=" * 60)
    print("[3/5] Converting mm_emb parquet to OnePiece format ...")
    print("=" * 60)
    start = time.time()

    import numpy as np

    mm_emb_dir = raw_data_path / "mm_emb"
    creative_emb_dir = output_path / "creative_emb"
    creative_emb_dir.mkdir(parents=True, exist_ok=True)

    SHAPE_DICT = {"81": 32, "82": 1024}

    for feat_id, shape in SHAPE_DICT.items():
        parquet_subdir = mm_emb_dir / f"emb_{feat_id}_{shape}_parquet"
        if not parquet_subdir.exists():
            print(f"  Skipping feat_id {feat_id}: directory not found")
            continue

        parquet_files = sorted(parquet_subdir.glob("*.parquet"))
        if not parquet_files:
            print(f"  Skipping feat_id {feat_id}: no parquet files")
            continue

        if feat_id == '81':
            # For feat_id 81, create a pickle file as expected by non-debug mode
            # creative_emb/emb_81_32.pkl -> dict[original_item_id_str -> np.array(float16)]
            pkl_path = creative_emb_dir / f"emb_{feat_id}_{shape}.pkl"
            if pkl_path.exists():
                print(f"  {pkl_path.name} already exists, skipping")
                continue

            print(f"  Converting feat_id {feat_id} (dim={shape}) to pkl ...")
            emb_dict = {}
            for pf in parquet_files:
                table = pq.read_table(pf)
                cids = table.column("anonymous_cid").to_pylist()
                embs = table.column("emb").to_pylist()
                for cid, emb in zip(cids, embs):
                    emb_dict[cid] = np.array(emb, dtype=np.float16)
                print(f"    Processed {pf.name}, total embeddings: {len(emb_dict)}")

            with open(pkl_path, 'wb') as f:
                pickle.dump(emb_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"    Saved {pkl_path.name} ({len(emb_dict)} embeddings)")
        else:
            # For other feat_ids, create JSON line files as expected by load_mm_emb
            # creative_emb/emb_{feat_id}_{shape}/part_NNNNN.json
            json_dir = creative_emb_dir / f"emb_{feat_id}_{shape}"
            json_dir.mkdir(parents=True, exist_ok=True)

            # Check if already done
            existing_jsons = list(json_dir.glob("*.json"))
            if existing_jsons:
                print(f"  {json_dir.name} already has {len(existing_jsons)} json files, skipping")
                continue

            print(f"  Converting feat_id {feat_id} (dim={shape}) to JSON lines ...")
            total_embs = 0
            for pf_idx, pf in enumerate(parquet_files):
                table = pq.read_table(pf)
                cids = table.column("anonymous_cid").to_pylist()
                embs = table.column("emb").to_pylist()

                json_file = json_dir / f"part-{pf_idx:05d}.json"
                with open(json_file, 'w') as jf:
                    for cid, emb in zip(cids, embs):
                        record = json.dumps({"anonymous_cid": cid, "emb": emb})
                        jf.write(record + '\n')
                total_embs += len(cids)
                print(f"    Processed {pf.name} -> {json_file.name} ({len(cids)} embs)")

            print(f"    Total embeddings for feat_id {feat_id}: {total_embs}")

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")


# ===========================================================================
# 6. Copy indexer.pkl and create placeholder files
# ===========================================================================
def copy_indexer_and_placeholders(raw_data_path: Path, output_path: Path):
    """
    Copy indexer.pkl from raw data to output.
    Create empty placeholder files for test/inference if they don't exist.
    """
    print("=" * 60)
    print("[4/5] Copying indexer.pkl and creating placeholders ...")
    print("=" * 60)

    import shutil

    # Copy indexer.pkl
    src = raw_data_path / "indexer.pkl"
    dst = output_path / "indexer.pkl"
    if not dst.exists():
        print(f"  Copying indexer.pkl ...")
        shutil.copy2(str(src), str(dst))
    else:
        print(f"  indexer.pkl already exists, skipping")

    # Create empty predict_seq.jsonl and predict_seq_offsets.pkl
    predict_seq_path = output_path / "predict_seq.jsonl"
    predict_offsets_path = output_path / "predict_seq_offsets.pkl"

    if not predict_seq_path.exists():
        print("  Creating empty predict_seq.jsonl placeholder")
        with open(predict_seq_path, 'w') as f:
            pass

    if not predict_offsets_path.exists():
        print("  Creating empty predict_seq_offsets.pkl placeholder")
        with open(predict_offsets_path, 'wb') as f:
            pickle.dump({}, f)

    print("  Done")


# ===========================================================================
# 7. Run item_exposure_data.py and timestamp_buckets.py
# ===========================================================================
def run_post_processing(output_path: Path, user_cache_path: Path,
                        skip_exposure: bool, skip_timestamp: bool):
    """
    Run item_exposure_data.py and timestamp_buckets.py which read seq.jsonl
    and produce additional pkl files in user_cache_path/item_exposure/.
    """
    print("=" * 60)
    print("[5/5] Running post-processing scripts ...")
    print("=" * 60)

    code_dir = Path(__file__).parent

    if not skip_exposure:
        print("  Running item_exposure_data.py ...")
        import subprocess
        env = os.environ.copy()
        env['TRAIN_DATA_PATH'] = str(output_path)
        env['USER_CACHE_PATH'] = str(user_cache_path)
        result = subprocess.run(
            [sys.executable, str(code_dir / "item_exposure_data.py")],
            env=env, cwd=str(code_dir),
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  ERROR in item_exposure_data.py:\n{result.stderr}")
        else:
            print("  item_exposure_data.py completed successfully")
    else:
        print("  Skipping item_exposure_data.py ( --skip_exposure )")

    if not skip_timestamp:
        print("  Running timestamp_buckets.py ...")
        import subprocess
        env = os.environ.copy()
        env['TRAIN_DATA_PATH'] = str(output_path)
        env['USER_CACHE_PATH'] = str(user_cache_path)
        result = subprocess.run(
            [sys.executable, str(code_dir / "timestamp_buckets.py"),
             "--method", "frequency", "--buckets", "8192"],
            env=env, cwd=str(code_dir),
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  ERROR in timestamp_buckets.py:\n{result.stderr}")
        else:
            print("  timestamp_buckets.py completed successfully")
    else:
        print("  Skipping timestamp_buckets.py ( --skip_timestamp )")


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()

    raw_data_path = Path(args.raw_data_path)
    output_path = Path(args.output_path)
    user_cache_path = Path(args.user_cache_path)

    print("=" * 60)
    print("OnePiece Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Raw data:      {raw_data_path}")
    print(f"  Output (TRAIN_DATA_PATH): {output_path}")
    print(f"  User cache:    {user_cache_path}")
    print("=" * 60)

    total_start = time.time()

    # --- Step 0: Load indexer for reverse mappings ---
    print("Loading indexer.pkl ...")
    with open(raw_data_path / "indexer.pkl", "rb") as f:
        indexer = pickle.load(f)
    indexer_u_rev = {v: k for k, v in indexer["u"].items()}  # reid -> original str
    indexer_i_rev = {v: k for k, v in indexer["i"].items()}  # reid -> original int
    print(f"  Indexer loaded: {len(indexer['u'])} users, {len(indexer['i'])} items")

    # --- Step 1: Load feature dicts into memory ---
    user_feat_dict = load_user_feats(raw_data_path)
    item_feat_dict_inmem = load_item_feats(raw_data_path)

    # --- Step 2: Build item_feat_dict.json ---
    if not args.skip_item_feat:
        build_item_feat_dict(raw_data_path, user_cache_path)
    else:
        print("Skipping item_feat_dict.json ( --skip_item_feat )")

    # --- Step 3: Build seq.jsonl, seq_offsets.pkl, user_action_type.json ---
    if not args.skip_seq:
        build_seq_files(raw_data_path, output_path, user_feat_dict, item_feat_dict_inmem,
                        indexer_u_rev)
    else:
        print("Skipping seq.jsonl ( --skip_seq )")

    # --- Step 4: Copy indexer.pkl and create placeholders ---
    copy_indexer_and_placeholders(raw_data_path, output_path)

    # --- Step 5: Convert mm_emb ---
    if not args.skip_mm_emb:
        convert_mm_emb(raw_data_path, output_path)
    else:
        print("Skipping mm_emb conversion ( --skip_mm_emb )")

    # --- Step 6: Run post-processing ---
    run_post_processing(output_path, user_cache_path,
                        args.skip_exposure, args.skip_timestamp)

    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"ALL DONE. Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
