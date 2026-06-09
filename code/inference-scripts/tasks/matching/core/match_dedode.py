"""DeDoDe matching utilities — MNN and DualSoftMax wrappers.

Output format: index-pair array ``[M, 2]`` identical to LightGlue matcher,
so downstream concat / ransac / reconstruction code works unchanged.
"""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import h5py


# ---------------------------------------------------------------------------
# Mutual Nearest Neighbor matcher (preferred — fast, returns index pairs)
# ---------------------------------------------------------------------------

def match_mnn(desc1, desc2, threshold=0.85):
    """Mutual nearest neighbor matching on L2-normalised descriptors.

    Parameters
    ----------
    desc1 : Tensor [N, D]
    desc2 : Tensor [M, D]
    threshold : float  — minimum cosine-similarity to keep

    Returns
    -------
    indices : ndarray [K, 2]  — each row is (idx_in_desc1, idx_in_desc2)
    """
    desc1 = F.normalize(desc1, dim=-1)
    desc2 = F.normalize(desc2, dim=-1)
    sim = desc1 @ desc2.T  # [N, M]

    nn12 = sim.argmax(dim=-1)   # [N]
    nn21 = sim.argmax(dim=-2)   # [M]

    ids1 = torch.arange(sim.shape[0], device=sim.device)
    mutual = (nn21[nn12] == ids1)

    # Filter by similarity threshold
    scores = sim[ids1, nn12]
    valid = mutual & (scores > threshold)

    indices = torch.stack([ids1[valid], nn12[valid]], dim=-1)
    return indices.cpu().numpy().astype(np.int64)  # [K, 2]


# ---------------------------------------------------------------------------
# DualSoftMax matcher (higher quality, more memory)
# ---------------------------------------------------------------------------

def match_dual_softmax(desc1, desc2, inv_temp=20, threshold=0.01):
    """DualSoftMax matching (DeDoDe official matcher logic).

    Parameters
    ----------
    desc1 : Tensor [N, D]
    desc2 : Tensor [M, D]
    inv_temp : float  — inverse temperature for softmax sharpness
    threshold : float — minimum dual-softmax probability to accept

    Returns
    -------
    indices : ndarray [K, 2]
    """
    desc1 = F.normalize(desc1, dim=-1)
    desc2 = F.normalize(desc2, dim=-1)
    sim = inv_temp * (desc1 @ desc2.T)  # [N, M]

    p_row = F.softmax(sim, dim=-1)   # [N, M]
    p_col = F.softmax(sim, dim=-2)   # [N, M]
    p_mutual = p_row * p_col         # [N, M]

    # Best match per row
    max_vals, nn12 = p_mutual.max(dim=-1)  # [N]
    valid = max_vals > threshold

    ids1 = torch.arange(sim.shape[0], device=sim.device)
    indices = torch.stack([ids1[valid], nn12[valid]], dim=-1)
    return indices.cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# H5-based matching loop (analogous to keypoint_matcing_LG)
# ---------------------------------------------------------------------------

def keypoint_matching_dedode(
    image_pairs,
    keypoints_h5_path,
    descriptions_h5_path,
    match_method="mnn",
    mnn_threshold=0.85,
    dual_softmax_inv_temp=20,
    dual_softmax_threshold=0.01,
    min_matches=15,
    verbose=False,
    device=torch.device("cpu"),
):
    """Match keypoints stored in h5 files using MNN or DualSoftMax.

    Returns
    -------
    matches : dict[str, dict[str, ndarray]]
        matches[key1][key2] = ndarray [K, 2] of index pairs
    """
    matches = {}

    with h5py.File(keypoints_h5_path, mode="r") as f_kp, \
         h5py.File(descriptions_h5_path, mode="r") as f_desc:

        for key1, key2 in tqdm(image_pairs, desc="Matching (DeDoDe)"):
            d1 = torch.from_numpy(f_desc[key1][...]).to(device)  # [N, 256]
            d2 = torch.from_numpy(f_desc[key2][...]).to(device)  # [M, 256]

            with torch.inference_mode():
                if match_method == "mnn":
                    idx_pairs = match_mnn(d1, d2, threshold=mnn_threshold)
                elif match_method == "dual_softmax":
                    idx_pairs = match_dual_softmax(
                        d1, d2,
                        inv_temp=dual_softmax_inv_temp,
                        threshold=dual_softmax_threshold,
                    )
                else:
                    raise ValueError(f"Unknown match_method: {match_method}")

            n_matches = len(idx_pairs)
            if verbose:
                print(f"{key1}-{key2}: {n_matches} matches")

            if n_matches >= min_matches:
                matches.setdefault(key1, {})
                matches[key1][key2] = idx_pairs.reshape(-1, 2)

    return matches
