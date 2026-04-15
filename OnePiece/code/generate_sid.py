#!/usr/bin/env python3
"""
Generate Semantic ID (SID) via hierarchical KMeans on item embeddings.

Uses the trained model's feat2emb + itemdnn to extract item embeddings,
then runs 2-level KMeans to produce (sid1, sid2) mapping.

Usage:
    python3 generate_sid.py --checkpoint <path> --output_dir <dir>
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--n_clusters_l1", type=int, default=512)
    parser.add_argument("--n_clusters_l2", type=int, default=32)
    # Reuse main_dist.py's args so Dataset/Model init correctly
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_device_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--maxlen", type=int, default=101)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--similarity_function", type=str, default="cosine")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_annealing", action="store_true", default=True)
    parser.add_argument("--lr_eta_min", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--sparse_embedding", action="store_true", default=False)
    parser.add_argument("--embedding_zero_init", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--pure_bf16", action="store_true", default=False)
    parser.add_argument("--infonce", action="store_true", default=True)
    parser.add_argument("--infonce_temp", type=float, default=0.02)
    parser.add_argument("--learnable_temp", action="store_true", default=False)
    parser.add_argument("--muon", action="store_true", default=True)
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--interest_k", type=int, default=1)
    parser.add_argument("--use_multi_interest", action="store_true", default=False)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=24)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--l2_emb", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--inference_only", action="store_true", default=True)
    parser.add_argument("--state_dict_path", type=str, default=None)
    parser.add_argument("--use_my_dataparallel", action="store_true", default=True)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--norm_first", action="store_true", default=True)
    parser.add_argument("--rope", action="store_true", default=False)
    parser.add_argument("--mm_emb_gate", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--random_perturbation", action="store_true", default=False)
    parser.add_argument("--random_perturbation_value", type=float, default=5e-3)
    parser.add_argument("--hash_emb_size", type=int, default=256)
    parser.add_argument("--timestamp_bucket_emb_size", type=int, default=128)
    parser.add_argument("--infer_logq", action="store_true", default=False)
    parser.add_argument("--use_hstu", action="store_true", default=True)
    parser.add_argument("--hstu_rope", action="store_true", default=False)
    parser.add_argument("--rms_norm", action="store_true", default=False)
    parser.add_argument("--dnn_hidden_units", type=int, default=4)
    parser.add_argument("--feed_forward_hidden_units", type=int, default=2)
    parser.add_argument("--base_user_sparse", type=str, nargs="+",
                        default=["103", "104", "105", "109"])
    parser.add_argument("--base_item_sparse", type=str, nargs="+",
                        default=["100", "117", "118", "101", "102", "119", "120", "114", "112", "121", "115",
                                 "122", "116"])
    parser.add_argument("--base_user_array", type=str, nargs="+", default=["106", "107", "108", "110"])
    parser.add_argument("--user_sparse", type=str, nargs="+", default=None)
    parser.add_argument("--item_sparse", type=str, nargs="+",
                        default=["exposure_start_year", "exposure_start_month", "exposure_start_day",
                                 "exposure_end_year", "exposure_end_month", "exposure_end_day"])
    parser.add_argument("--user_array", type=str, nargs="+", default=None)
    parser.add_argument("--item_array", type=str, nargs="+", default=None)
    parser.add_argument("--user_continual", type=str, nargs="+", default=None)
    parser.add_argument("--item_continual", type=str, nargs="+", default=None)
    parser.add_argument("--context_item_sparse", type=str, nargs="+",
                        default=["time_diff_day", "time_diff_hour", "time_diff_minute", "action_type",
                                 "next_action_type", "timestamp_bucket_id", "hot_bucket_1000"])
    parser.add_argument("--feature_dropout_list", type=str, nargs="+",
                        default=["timestamp_bucket_id", "hot_bucket_1000"])
    parser.add_argument("--feature_dropout_rate", type=float, default=0.5)
    parser.add_argument("--bucket_sizes", type=int, nargs="+", default=[])
    parser.add_argument("--user_id_exclude", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mm_emb_id", type=str, nargs="+", default=["81", "82"])
    parser.add_argument("--mm_sid", type=str, nargs="+", default=["81"])
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--test_train_valid_process", action="store_true", default=False)
    parser.add_argument("--generate_sid", action="store_true", default=True)
    parser.add_argument("--sid", action="store_true", default=False)
    parser.add_argument("--sid_codebook_layer", type=int, default=2)
    parser.add_argument("--sid_codebook_size", type=int, default=16384)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--reward", action="store_true", default=False)
    parser.add_argument("--reward_only", action="store_true", default=False)
    parser.add_argument("--use_preprocessing", action="store_true", default=False)
    parser.add_argument("--force_reprocess", action="store_true", default=False)
    parser.add_argument("--train_infer_result_path", type=str, default="")
    parser.add_argument("--save_infer_model", action="store_true", default=True)
    parser.add_argument("--use_moe", action="store_true", default=False)
    parser.add_argument("--moe_num_experts", type=int, default=64)
    parser.add_argument("--moe_top_k", type=int, default=3)
    parser.add_argument("--moe_intermediate_size", type=int, default=512)
    parser.add_argument("--moe_load_balancing_alpha", type=float, default=0)
    parser.add_argument("--moe_load_balancing_update_freq", type=int, default=1)
    parser.add_argument("--moe_shared_expert_num", type=int, default=1)
    parser.add_argument("--moe_use_sequence_aux_loss", action="store_true", default=True)
    parser.add_argument("--moe_sequence_aux_loss_coeff", type=float, default=0.02)
    parser.add_argument("--moe_dynamic_aux_loss", action="store_true", default=False)
    parser.add_argument("--moe_aux_loss_adjust_rate", type=float, default=0.0001)
    parser.add_argument("--moe_gini_target_min", type=float, default=0.09)
    parser.add_argument("--moe_gini_target_max", type=float, default=0.31)
    parser.add_argument("--moe_dynamic_aux_loss_start_step", type=int, default=700)
    parser.add_argument("--beam_search_generate", action="store_true", default=False)
    parser.add_argument("--beam_search_top_k", type=int, default=256)
    parser.add_argument("--beam_search_beam_size", type=int, default=20)
    parser.add_argument("--sid_resort", action="store_true", default=False)
    parser.add_argument("--sid_path", type=str, default=None)
    parser.add_argument("--sid_resort_topk", type=int, nargs="+", default=[32, 128, 256])
    args = parser.parse_args()

    # Set paths from env
    args.batch_size = args.single_device_batch_size
    args.log_path = os.environ.get("TRAIN_LOG_PATH", "/tmp")
    args.tb_path = os.environ.get("TRAIN_TF_EVENTS_PATH", "/tmp")
    args.data_path = os.environ.get("TRAIN_DATA_PATH", "/scratch/cy2668/TAAC2025/OnePiece/data")
    args.ckpt_path = os.environ.get("TRAIN_CKPT_PATH", "/scratch/cy2668/TAAC2025/OnePiece/checkpoints")
    args.user_cache_path = os.environ.get("USER_CACHE_PATH",
                                          "/scratch/cy2668/TAAC2025/OnePiece/user_cache")
    return args


def extract_item_embeddings(args, checkpoint_path):
    """Load model + dataset, extract all item embeddings via model.feat2emb."""
    from dataset import MyDataset
    from model import BaselineModel

    print("Loading dataset...")
    dataset = MyDataset(args.data_path, args)
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    print(f"  Items: {itemnum}, Users: {usernum}")

    print(f"Loading model from {checkpoint_path}...")
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    model = model.to(args.device)
    model.eval()
    print("  Model loaded.")

    hidden_dim = args.hidden_units * args.dnn_hidden_units
    print(f"  Hidden dim: {hidden_dim}")

    # Determine all feature keys the model expects
    all_sparse_feats = list(feat_types.get('item_sparse', []))
    all_array_feats = list(feat_types.get('item_array', []))
    all_emb_feats = list(feat_types.get('item_emb', []))

    # Iterate over all item IDs
    batch_size = 4096
    all_embeddings = []
    all_item_ids = list(range(1, itemnum + 1))

    print(f"\nExtracting embeddings for {len(all_item_ids)} items...")

    with torch.no_grad():
        for start in range(0, len(all_item_ids), batch_size):
            batch_ids = all_item_ids[start:start + batch_size]
            B = len(batch_ids)

            # Build seq tensor: [B, 1] — treat each item as a sequence of length 1
            seq = torch.tensor(batch_ids, dtype=torch.long, device=args.device).unsqueeze(1)

            # Build mask
            mask = torch.ones(B, 1, dtype=torch.long, device=args.device)

            # Step 1: Build seq_feat as list-of-list-of-dicts (same format as dataset.__getitem__)
            seq_feat_list = []
            for item_id in batch_ids:
                item_key = dataset.indexer_i_rev.get(item_id)
                feat = {}
                if item_key:
                    raw_feat = dataset.item_feat_dict.get(str(item_key), {})
                    feat.update(raw_feat)
                    # Fill mm_emb
                    for fid in args.mm_emb_id:
                        mm_dict = dataset.mm_emb_dict.get(fid, {})
                        item_str = str(item_key)
                        if item_str in mm_dict:
                            feat[fid] = mm_dict[item_str]
                        elif item_key in mm_dict:
                            feat[fid] = mm_dict[item_key]
                # Wrap in list for seq_len=1
                seq_feat_list.append([feat])

            # Step 2: Convert to dict-of-tensors (same as dataset.collate_fn)
            seq_feat_dict = {}

            # Sparse features
            for k in all_sparse_feats:
                batch_data = np.zeros((B, 1), dtype=np.int64)
                default_val = dataset.feature_default_value.get(k, 0)
                for i in range(B):
                    batch_data[i, 0] = seq_feat_list[i][0].get(k, default_val)
                seq_feat_dict[k] = torch.from_numpy(batch_data)

            # Array features
            for k in all_array_feats:
                max_arr_len = 0
                for i in range(B):
                    val = seq_feat_list[i][0].get(k, [])
                    if val:
                        max_arr_len = max(max_arr_len, len(val))
                max_arr_len = max(max_arr_len, 1)
                batch_data = np.zeros((B, 1, max_arr_len), dtype=np.int64)
                for i in range(B):
                    val = seq_feat_list[i][0].get(k, [])
                    if val:
                        actual_len = min(len(val), max_arr_len)
                        batch_data[i, 0, :actual_len] = val[:actual_len]
                seq_feat_dict[k] = torch.from_numpy(batch_data)

            # Embedding features (mm_emb)
            for k in all_emb_feats:
                emb_dim = dataset.feature_default_value[k].shape[0]
                batch_data = np.zeros((B, 1, emb_dim), dtype=np.float32)
                default_val = np.zeros(emb_dim, dtype=np.float32)
                for i in range(B):
                    val = seq_feat_list[i][0].get(k, default_val)
                    if isinstance(val, torch.Tensor):
                        val = val.numpy()
                    if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] >= emb_dim:
                        batch_data[i, 0] = val[:emb_dim]
                    elif isinstance(val, np.ndarray) and val.ndim >= 1:
                        batch_data[i, 0, :val.shape[0]] = val
                    else:
                        batch_data[i, 0] = default_val
                seq_feat_dict[k] = torch.from_numpy(batch_data)

            # Use model's feat2emb (include_user=False for pure item embedding)
            seq_embs = model.feat2emb(seq, seq_feat_dict, mask=mask, include_user=False)
            # seq_embs: [B, 1, hidden_dim] — already through itemdnn + relu
            item_embs = seq_embs.squeeze(1).cpu().numpy()  # [B, hidden_dim]
            all_embeddings.append(item_embs)

            if (start // batch_size) % 100 == 0:
                print(f"  Processed {start + B} / {len(all_item_ids)}")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Done. Shape: {all_embeddings.shape}")
    return all_item_ids, all_embeddings


def hierarchical_kmeans(item_ids, embeddings, n_clusters_l1, n_clusters_l2):
    """Run hierarchical KMeans: L1 then L2 within each L1 cluster."""
    print(f"\nRunning hierarchical KMeans...")
    print(f"  L1 clusters: {n_clusters_l1}, L2 clusters per L1: {n_clusters_l2}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings_norm = embeddings / norms

    print("  Running L1 KMeans...")
    kmeans_l1 = KMeans(n_clusters=n_clusters_l1, random_state=42, n_init=10, max_iter=300)
    sid1_labels = kmeans_l1.fit_predict(embeddings_norm)
    print(f"  L1 done. Unique clusters: {len(np.unique(sid1_labels))}")

    print("  Running L2 KMeans within each L1 cluster...")
    sid2_labels = np.zeros(len(item_ids), dtype=np.int32)

    for l1_id in range(n_clusters_l1):
        mask = sid1_labels == l1_id
        cluster_embs = embeddings_norm[mask]
        n_in_cluster = cluster_embs.shape[0]

        if n_in_cluster <= n_clusters_l2:
            sid2_labels[mask] = np.arange(n_in_cluster)
        else:
            kmeans_l2 = KMeans(n_clusters=n_clusters_l2, random_state=42, n_init=5, max_iter=200)
            sid2_labels[mask] = kmeans_l2.fit_predict(cluster_embs)

        if (l1_id + 1) % 100 == 0:
            print(f"    L1 cluster {l1_id + 1}/{n_clusters_l1} done")

    print("  L2 done.")

    sid_map = {}
    for idx, item_id in enumerate(item_ids):
        sid_map[item_id] = [int(sid1_labels[idx]), int(sid2_labels[idx])]

    unique_sids = set(tuple(v) for v in sid_map.values())
    print(f"\n  Total items: {len(sid_map)}")
    print(f"  Unique (sid1, sid2) pairs: {len(unique_sids)}")
    print(f"  Collision rate: {1 - len(unique_sids) / len(sid_map):.2%}")

    return sid_map


def main():
    # Phase 1: Extract our custom args (--checkpoint, --output_dir, --n_clusters_l1/l2)
    # before calling get_args(), which would fail because those args aren't in its parser.
    ckpt_path = None
    output_dir = None
    n_l1 = 512
    n_l2 = 32
    remaining = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--checkpoint' and i + 1 < len(sys.argv):
            ckpt_path = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == '--output_dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == '--n_clusters_l1' and i + 1 < len(sys.argv):
            n_l1 = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == '--n_clusters_l2' and i + 1 < len(sys.argv):
            n_l2 = int(sys.argv[i + 1]); i += 2
        else:
            remaining.append(sys.argv[i]); i += 1

    if ckpt_path is None or output_dir is None:
        print("Usage: generate_sid.py --checkpoint <path> --output_dir <dir>")
        sys.exit(1)

    # Phase 2: Parse remaining args through the standard argparse (mirrors main_dist.py args)
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()

    # Step 1: Extract item embeddings
    item_ids, embeddings = extract_item_embeddings(args, ckpt_path)

    # Step 2: Hierarchical KMeans
    sid_map = hierarchical_kmeans(item_ids, embeddings, n_l1, n_l2)

    # Step 3: Save
    os.makedirs(output_dir, exist_ok=True)
    mm_suffix = "_".join(args.mm_emb_id)
    output_path = Path(output_dir) / f"sid_{mm_suffix}.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(sid_map, f)

    print(f"\nSID mapping saved to: {output_path}")
    print(f"  Total items: {len(sid_map)}")
    sample_keys = list(sid_map.keys())[:5]
    for k in sample_keys:
        print(f"  Item {k}: sid={sid_map[k]}")


if __name__ == "__main__":
    main()
