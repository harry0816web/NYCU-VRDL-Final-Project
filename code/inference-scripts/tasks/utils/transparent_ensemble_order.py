"""
Transparent ensemble ordering: LoMa/ALIKED rank-fusion for ring ordering.

Reimplements best.ipynb's blend_transparent_order_counts() logic as a
pipeline task.  Takes two CSVs (one with LoMa match counts, one with ALIKED
match counts), computes rank-normalized scores, blends with configurable
weights, and writes a merged image_pair.csv with the blended scores as
match_num — ready for transparent_ring_poses in reconstruction.
"""

import os
import pandas as pd
import numpy as np


def _rank_scores(pairs, counts):
    """
    Rank-normalize match counts for a set of pairs.
    Returns dict: (key1, key2) -> float score in [0, 1].
    """
    sorted_pairs = sorted(
        pairs,
        key=lambda p: (counts.get(p, 0), p),
    )
    denom = max(1, len(sorted_pairs) - 1)
    return {pair: float(rank) / float(denom) for rank, pair in enumerate(sorted_pairs)}


def task_transparent_ensemble_order(params):
    if params.get("pdb", False):
        import pdb; pdb.set_trace()

    work_dir = params["work_dir"]

    # Load LoMa and ALIKED match counts
    loma_df = pd.read_csv(os.path.join(work_dir, params["input"]["loma_counts_csv"]))
    aliked_df = pd.read_csv(os.path.join(work_dir, params["input"]["aliked_counts_csv"]))

    loma_weight = params.get("loma_weight", 0.35)
    aliked_weight = params.get("aliked_weight", 0.65)

    # Build count dicts: (key1, key2) sorted -> match_num
    loma_counts = {}
    for _, row in loma_df.iterrows():
        key = tuple(sorted((row["key1"], row["key2"])))
        loma_counts[key] = int(row["match_num"])

    aliked_counts = {}
    for _, row in aliked_df.iterrows():
        key = tuple(sorted((row["key1"], row["key2"])))
        aliked_counts[key] = int(row["match_num"])

    # Union of all pairs
    all_pairs = list(set(loma_counts.keys()) | set(aliked_counts.keys()))

    # Rank-normalize each
    loma_rank = _rank_scores(all_pairs, loma_counts)
    aliked_rank = _rank_scores(all_pairs, aliked_counts)

    # Blend
    blended = {}
    for pair in all_pairs:
        score = (
            loma_weight * loma_rank.get(pair, 0.0)
            + aliked_weight * aliked_rank.get(pair, 0.0)
        )
        blended[pair] = int(round(score * 1_000_000))

    # Use loma_df as base template for columns (sim, dir1, dir2, etc.)
    # Merge with aliked pairs that might not be in loma
    all_rows = {}
    for _, row in loma_df.iterrows():
        key = tuple(sorted((row["key1"], row["key2"])))
        all_rows[key] = dict(row)
    for _, row in aliked_df.iterrows():
        key = tuple(sorted((row["key1"], row["key2"])))
        if key not in all_rows:
            all_rows[key] = dict(row)

    # Build output
    dst_data = {
        "key1": [],
        "key2": [],
        "sim": [],
        "dir1": [],
        "dir2": [],
        "match_num": [],
    }
    for pair in all_pairs:
        row = all_rows[pair]
        dst_data["key1"].append(row["key1"])
        dst_data["key2"].append(row["key2"])
        dst_data["sim"].append(row.get("sim", 0))
        dst_data["dir1"].append(row.get("dir1", 0))
        dst_data["dir2"].append(row.get("dir2", 0))
        dst_data["match_num"].append(blended[pair])

    dst_df = pd.DataFrame.from_dict(dst_data)
    # Sort by blended match_num descending for easier inspection
    dst_df = dst_df.sort_values("match_num", ascending=False).reset_index(drop=True)

    output_path = os.path.join(work_dir, params["output"])
    print(f"[transparent_ensemble_order] loma_weight={loma_weight}, aliked_weight={aliked_weight}")
    print(f"[transparent_ensemble_order] {len(all_pairs)} pairs blended -> {output_path}")
    dst_df.to_csv(output_path, index=False)
