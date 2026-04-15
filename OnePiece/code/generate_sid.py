#!/usr/bin/env python3
"""
Generate Semantic ID (SID) via hierarchical KMeans on item embeddings.

Usage:
    python3 generate_sid.py --checkpoint <path_to_model.pt> --output_dir <dir>

Produces:
    <output_dir>/sid_81_82.pkl  — dict mapping item_id -> [sid1, sid2]
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save sid pkl")
    parser.add_argument("--user_cache_path", type=str,
                        default=os.environ.get("USER_CACHE_PATH", "/scratch/cy2668/TAAC2025/OnePiece/user_cache"))
    parser.add_argument("--n_clusters_l1", type=int, default=512, help="Number of L1 clusters (sid1)")
    parser.add_argument("--n_clusters_l2", type=int, default=32, help="Number of L2 clusters per L1 (sid2)")
    parser.add_argument("--mm_emb_ids", type=str, nargs="+", default=["81", "82"])
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--dnn_hidden_units", type=int, default=4)
    parser.add_argument("--hash_emb_size", type=int, default=256)
    parser.add_argument("--item_sparse_feats", type=str, nargs="+",
                        default=["100", "117", "118", "101", "102", "119", "120", "114", "112", "121", "115",
                                 "122", "116"])
    parser.add_argument("--item_array_feats", type=str, nargs="+", default=["106", "107", "108", "110"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_mm_emb(emb_dir, mm_emb_ids):
    """Load multimodal embeddings from pkl files."""
    mm_embs = {}
    for fid in mm_emb_ids:
        # Try different naming conventions
        candidates = [
            emb_dir / f"emb_{fid}.pkl",
            emb_dir / fid,
        ]
        # Also try globbing
        for p in emb_dir.glob(f"*{fid}*"):
            candidates.append(p)

        loaded = False
        for p in candidates:
            if p.exists():
                with open(p, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    mm_embs[fid] = data
                    loaded = True
                    print(f"  Loaded mm_emb {fid}: {len(data)} items from {p.name}")
                    break
        if not loaded:
            print(f"  WARNING: mm_emb {fid} not found, using zeros")
            mm_embs[fid] = {}
    return mm_embs


def build_item_features(args):
    """Build feature matrix for all items."""
    print("Loading item feature dict...")
    with open(Path(args.user_cache_path, "item_feat_dict.json"), "r") as f:
        item_feat_dict = json.load(f)
    print(f"  Total items in feat dict: {len(item_feat_dict)}")

    # Load mm embeddings
    emb_dir = Path(args.user_cache_path) / "creative_emb"
    print("Loading mm embeddings...")
    mm_embs = load_mm_emb(emb_dir, args.mm_emb_ids)

    # Feature dimensions:
    # item_id_emb: 32, hash_a: 256, hash_b: 256
    # sparse feats: each hidden_units, array feats: each hidden_units
    # mm_emb: each mapped through linear to hidden_units
    hidden_dim = args.hidden_units
    id_dim = 32
    hash_dim = args.hash_emb_size * 2
    sparse_dim = hidden_dim * len(args.item_sparse_feats)
    array_dim = hidden_dim * len(args.item_array_feats)
    # mm_emb dims will be handled by the model's emb_transform layers

    total_sparse_dim = id_dim + hash_dim + sparse_dim + array_dim
    # We'll compute mm_emb contribution separately via model

    # Collect all item IDs (as ints), excluding 0
    all_item_ids = sorted([int(k) for k in item_feat_dict.keys() if int(k) > 0])
    max_item_id = max(all_item_ids)
    print(f"  Item ID range: 1 .. {max_item_id}, total: {len(all_item_ids)}")

    return all_item_ids, item_feat_dict, mm_embs


def extract_item_embeddings_from_model(args):
    """Load checkpoint and extract all item embeddings through itemdnn."""
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Inspect checkpoint to find relevant keys
    item_emb_weight = checkpoint.get("item_emb.weight", None)
    if item_emb_weight is None:
        print("ERROR: 'item_emb.weight' not found in checkpoint")
        print("Available keys (first 20):", list(checkpoint.keys())[:20])
        sys.exit(1)

    num_items = item_emb_weight.shape[0] - 1  # minus padding
    emb_dim = item_emb_weight.shape[1]
    print(f"  item_emb: {num_items} items, dim={emb_dim}")
    print(f"  itemdnn weight shape: {checkpoint['itemdnn.weight'].shape}")

    # Build item feature tensor for all items
    # Need to replicate feat2emb logic
    with open(Path(args.user_cache_path, "item_feat_dict.json"), "r") as f:
        item_feat_dict = json.load(f)

    # Load mm_embs
    emb_dir = Path(args.user_cache_path) / "creative_emb"
    mm_embs = load_mm_emb(emb_dir, args.mm_emb_ids)

    # Get sparse feat statistics from dataset to know vocab sizes
    # We'll construct features manually
    hidden_units = args.hidden_units

    # Item id embedding
    item_emb_layer = torch.nn.Embedding(num_items + 1, 32, padding_idx=0)
    item_emb_layer.weight.data = checkpoint["item_emb.weight"]

    # Hash embeddings
    hash_prime_a = 2000003
    hash_prime_b = 3000017
    hash_emb_a = torch.nn.Embedding(hash_prime_a + 1, args.hash_emb_size, padding_idx=0)
    hash_emb_b = torch.nn.Embedding(hash_prime_b + 1, args.hash_emb_size, padding_idx=0)
    hash_emb_a.weight.data = checkpoint["item_hash_emb_a.weight"]
    hash_emb_b.weight.data = checkpoint["item_hash_emb_b.weight"]

    # Sparse embeddings for item features
    sparse_emb_layers = {}
    for k in args.item_sparse_feats + args.item_array_feats:
        key = f"sparse_emb.{k}.weight"
        if key in checkpoint:
            vocab_size = checkpoint[key].shape[0]
            sparse_emb_layers[k] = torch.nn.Embedding(vocab_size, hidden_units, padding_idx=0)
            sparse_emb_layers[k].weight.data = checkpoint[key]
        else:
            print(f"  WARNING: sparse_emb key '{k}' not found in checkpoint")

    # mm_emb transform layers
    emb_transform_layers = {}
    for k in args.mm_emb_ids:
        key = f"emb_transform.{k}.weight"
        if key in checkpoint:
            in_dim = checkpoint[key].shape[1]
            out_dim = checkpoint[key].shape[0]
            emb_transform_layers[k] = torch.nn.Linear(in_dim, out_dim)
            emb_transform_layers[k].weight.data = checkpoint[key]
            if f"emb_transform.{k}.bias" in checkpoint:
                emb_transform_layers[k].bias.data = checkpoint[f"emb_transform.{k}.bias"]
        else:
            print(f"  WARNING: emb_transform key '{k}' not found in checkpoint")

    # Item DNN
    itemdnn_weight = checkpoint["itemdnn.weight"]
    itemdnn_bias = checkpoint["itemdnn.bias"]
    hidden_dim = itemdnn_bias.shape[0]

    # Process items in batches to avoid OOM
    batch_size = 4096
    all_item_ids = sorted([int(k) for k in item_feat_dict.keys() if int(k) > 0])
    all_embeddings = []

    print(f"\nExtracting embeddings for {len(all_item_ids)} items in batches of {batch_size}...")

    for start in range(0, len(all_item_ids), batch_size):
        batch_ids = all_item_ids[start:start + batch_size]
        feat_list = []

        # Item id embeddings
        id_tensor = torch.tensor(batch_ids, dtype=torch.long)
        feat_list.append(item_emb_layer(id_tensor))  # [B, 32]

        # Hash embeddings
        feat_list.append(hash_emb_a(id_tensor % hash_prime_a))  # [B, 256]
        feat_list.append(hash_emb_b(id_tensor % hash_prime_b))  # [B, 256]

        # Sparse features
        for k in args.item_sparse_feats:
            if k in sparse_emb_layers:
                vals = torch.tensor(
                    [item_feat_dict.get(str(i), {}).get(k, 0) for i in batch_ids],
                    dtype=torch.long
                )
                feat_list.append(sparse_emb_layers[k](vals))  # [B, hidden_units]

        # Array features (sum over last dim)
        for k in args.item_array_feats:
            if k in sparse_emb_layers:
                vals = torch.tensor(
                    [item_feat_dict.get(str(i), {}).get(k, [0]) for i in batch_ids],
                    dtype=torch.long
                )
                if vals.dim() == 1:
                    vals = vals.unsqueeze(1)
                feat_list.append(sparse_emb_layers[k](vals).sum(dim=1))  # [B, hidden_units]

        # mm_emb features (through transform)
        for k in args.mm_emb_ids:
            if k in emb_transform_layers:
                emb_data = mm_embs.get(k, {})
                mm_vals = []
                for i in batch_ids:
                    if str(i) in emb_data:
                        v = emb_data[str(i)]
                    elif i in emb_data:
                        v = emb_data[i]
                    else:
                        v = np.zeros(emb_transform_layers[k].weight.shape[1], dtype=np.float32)
                    mm_vals.append(v)
                mm_tensor = torch.tensor(np.array(mm_vals), dtype=torch.float32)
                feat_list.append(emb_transform_layers[k](mm_tensor))  # [B, hidden_units]

        # Concatenate and pass through itemdnn
        all_feats = torch.cat(feat_list, dim=-1)  # [B, total_dim]
        item_embs = torch.relu(all_feats @ itemdnn_weight.t() + itemdnn_bias)  # [B, hidden_dim]

        all_embeddings.append(item_embs.detach().cpu().numpy())

        if (start // batch_size) % 50 == 0:
            print(f"  Processed {start + len(batch_ids)} / {len(all_item_ids)} items")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Done. Embedding shape: {all_embeddings.shape}")

    return all_item_ids, all_embeddings


def hierarchical_kmeans(item_ids, embeddings, n_clusters_l1, n_clusters_l2):
    """Run hierarchical KMeans: L1 then L2 within each L1 cluster."""
    print(f"\nRunning hierarchical KMeans...")
    print(f"  L1 clusters: {n_clusters_l1}, L2 clusters per L1: {n_clusters_l2}")

    # Normalize embeddings for better clustering
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings_norm = embeddings / norms

    # L1 KMeans
    print("  Running L1 KMeans...")
    kmeans_l1 = KMeans(n_clusters=n_clusters_l1, random_state=42, n_init=10, max_iter=300)
    sid1_labels = kmeans_l1.fit_predict(embeddings_norm)
    print(f"  L1 done. Unique clusters: {len(np.unique(sid1_labels))}")

    # L2 KMeans within each L1 cluster
    print("  Running L2 KMeans within each L1 cluster...")
    sid2_labels = np.zeros(len(item_ids), dtype=np.int32)

    for l1_id in range(n_clusters_l1):
        mask = sid1_labels == l1_id
        cluster_embs = embeddings_norm[mask]
        n_in_cluster = cluster_embs.shape[0]

        if n_in_cluster <= n_clusters_l2:
            # Fewer items than L2 clusters — assign each item a unique L2
            sid2_labels[mask] = np.arange(n_in_cluster)
        else:
            kmeans_l2 = KMeans(n_clusters=n_clusters_l2, random_state=42, n_init=5, max_iter=200)
            sid2_labels[mask] = kmeans_l2.fit_predict(cluster_embs)

        if (l1_id + 1) % 100 == 0:
            print(f"    L1 cluster {l1_id + 1}/{n_clusters_l1} done")

    print("  L2 done.")

    # Build SID mapping: item_id -> [sid1, sid2]
    sid_map = {}
    for idx, item_id in enumerate(item_ids):
        sid_map[item_id] = [int(sid1_labels[idx]), int(sid2_labels[idx])]

    # Stats
    unique_sids = set(tuple(v) for v in sid_map.values())
    print(f"\n  Total items: {len(sid_map)}")
    print(f"  Unique (sid1, sid2) pairs: {len(unique_sids)}")
    print(f"  Collision rate: {1 - len(unique_sids) / len(sid_map):.2%}")

    return sid_map


def main():
    args = get_args()

    # Step 1: Extract item embeddings from model
    item_ids, embeddings = extract_item_embeddings_from_model(args)

    # Step 2: Hierarchical KMeans
    sid_map = hierarchical_kmeans(
        item_ids, embeddings,
        n_clusters_l1=args.n_clusters_l1,
        n_clusters_l2=args.n_clusters_l2,
    )

    # Step 3: Save
    os.makedirs(args.output_dir, exist_ok=True)
    mm_suffix = "_".join(args.mm_emb_ids)
    output_path = Path(args.output_dir) / f"sid_{mm_suffix}.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(sid_map, f)

    print(f"\nSID mapping saved to: {output_path}")
    print(f"  Total items: {len(sid_map)}")

    # Preview
    sample_keys = list(sid_map.keys())[:5]
    for k in sample_keys:
        print(f"  Item {k}: sid={sid_map[k]}")


if __name__ == "__main__":
    main()
