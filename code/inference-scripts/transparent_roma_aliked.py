"""
Transparent scene matching: RoMa + ALIKED ensemble for TSP distance computation.

This module provides drop-in replacements for the notebook's inline transparent
scene functions. Instead of using ALIKED-only for computing matching_flows,
it uses RoMa dense matching + ALIKED sparse matching ensemble.

Usage in notebook (replace the existing get_matching_flows call):
--------------------------------------------------------------

    # === Setup (add after existing ALIKED/LightGlue init) ===
    from transparent_roma_aliked import (
        init_roma_model,
        get_matching_flows_ensemble,
    )
    roma_model = init_roma_model(
        roma_code_path="/kaggle/input/roma-weights/RoMa",
        roma_weights_path="/kaggle/input/roma-weights/roma_outdoor.pth",
        dinov2_weights_path="/kaggle/input/imc24lightglue/weights/dinov2_vitl14_pretrain.pth",
        device="cuda",
    )

    # === In the transparent scene loop (replace get_matching_flows) ===
    matching_flows = get_matching_flows_ensemble(
        fnames,
        aliked_extractor=aliked_extractor,
        lightglue_matcher=lightglue_matcher,
        roma_model=roma_model,
        image_sizes=[1024, 1280, 1600],  # ALIKED multi-resolution
    )
    # Everything else (SSIM, TSP, uniform rotation) stays the same.
"""

import sys
import gc
import numpy as np
import cv2
import torch
import kornia.feature as KF
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Roma model init (call once, reuse across scenes)
# ---------------------------------------------------------------------------

def init_roma_model(roma_code_path, roma_weights_path, dinov2_weights_path,
                    coarse_res=560, upsample_res=864, device="cuda"):
    """Load RoMa model from local weights."""
    sys.path.insert(0, roma_code_path)
    from romatch import roma_outdoor

    torch.set_float32_matmul_precision("highest")
    roma_weights = torch.load(roma_weights_path, map_location=device)
    dinov2_weights = torch.load(dinov2_weights_path, map_location=device)

    model = roma_outdoor(
        device=device,
        weights=roma_weights,
        dinov2_weights=dinov2_weights,
        coarse_res=coarse_res,
        upsample_res=upsample_res,
        amp_dtype=torch.float16,
        use_custom_corr=False,  # local_corr CUDA ext not available on Kaggle
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helper: read + resize (same as notebook)
# ---------------------------------------------------------------------------

def _resize(image, image_size):
    h, w = image.shape[:2]
    aspect_ratio = h / w
    smaller_side_size = int(image_size / max(aspect_ratio, 1 / aspect_ratio))
    if aspect_ratio > 1:
        new_size = (image_size, smaller_side_size)
    else:
        new_size = (smaller_side_size, image_size)
    image = cv2.resize(image, new_size[::-1])
    return image, new_size


def _read_image_kornia(fname):
    """Read image as float32 HWC numpy (same as kornia read_image but without torch)."""
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _numpy_image_to_torch(img):
    """HWC uint8 numpy → CHW float32 torch tensor."""
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# RoMa matching for one pair
# ---------------------------------------------------------------------------

def roma_pair_match(roma_model, fname1, fname2, max_matches=10000):
    """Run RoMa on a pair, return number of matches after RANSAC filtering."""
    img1 = Image.open(fname1).convert("RGB")
    img2 = Image.open(fname2).convert("RGB")

    W1, H1 = img1.size
    W2, H2 = img2.size

    try:
        warp, certainty = roma_model.match(img1, img2)
        dense_matches, _ = roma_model.sample(warp, certainty, num=max_matches)
    except Exception as e:
        print(f"[RoMa] Failed on {fname1}-{fname2}: {e}")
        return 0

    kpts = dense_matches.detach().cpu().numpy()  # [K, 4]
    mkpts1 = np.zeros((kpts.shape[0], 2), dtype=np.float32)
    mkpts2 = np.zeros((kpts.shape[0], 2), dtype=np.float32)
    mkpts1[:, 0] = (kpts[:, 0] + 1) / 2 * (W1 - 1)
    mkpts1[:, 1] = (kpts[:, 1] + 1) / 2 * (H1 - 1)
    mkpts2[:, 0] = (kpts[:, 2] + 1) / 2 * (W2 - 1)
    mkpts2[:, 1] = (kpts[:, 3] + 1) / 2 * (H2 - 1)

    if len(mkpts1) < 8:
        return len(mkpts1)

    try:
        _, inliers = cv2.findFundamentalMat(
            mkpts1, mkpts2, cv2.USAC_MAGSAC,
            ransacReprojThreshold=5, confidence=0.9999, maxIters=50000,
        )
        return int(inliers.ravel().sum()) if inliers is not None else 0
    except:
        return len(mkpts1)


# ---------------------------------------------------------------------------
# ALIKED matching for one pair (same logic as notebook's matching_inference)
# ---------------------------------------------------------------------------

def aliked_pair_match(aliked_extractor, lightglue_matcher, cache,
                      fname1, fname2, image_sizes):
    """Run ALIKED+LightGlue on a pair with co-orientation, return match count."""

    for fname in [fname1, fname2]:
        if fname not in cache:
            img = _read_image_kornia(fname)
            h, w = img.shape[:2]
            cache[fname] = {"image": img, "h": h, "w": w}
            for sz in image_sizes:
                if max(h, w) != sz:
                    img_r, (h_r, w_r) = _resize(img, sz)
                else:
                    img_r = img.copy()
                    h_r, w_r = img_r.shape[:2]
                tensor = _numpy_image_to_torch(img_r)
                cache[fname][sz] = {
                    "tensor": tensor,
                    "h_r": h_r, "w_r": w_r,
                    0: {}, 1: {}, 2: {}, 3: {},
                }

    def _extract_and_match(f1, f2, sz, rot_code_2):
        """Extract + match at given size/rotation, return num inlier matches."""
        c1 = cache[f1][sz]
        c2 = cache[f2][sz]

        if "keypoints" not in c1[0]:
            with torch.inference_mode():
                t = c1["tensor"].cuda()
                pred = aliked_extractor.extract(t, resize=None)
                c1[0]["keypoints"] = pred["keypoints"]
                c1[0]["descriptors"] = pred["descriptors"]

        if "keypoints" not in c2[rot_code_2]:
            with torch.inference_mode():
                t = torch.rot90(c2["tensor"], rot_code_2, [1, 2]).cuda()
                pred = aliked_extractor.extract(t, resize=None)
                c2[rot_code_2]["keypoints"] = pred["keypoints"]
                c2[rot_code_2]["descriptors"] = pred["descriptors"]

        with torch.inference_mode():
            _, indices = lightglue_matcher(
                c1[0]["descriptors"][0],
                c2[rot_code_2]["descriptors"][0],
                KF.laf_from_center_scale_ori(c1[0]["keypoints"]),
                KF.laf_from_center_scale_ori(c2[rot_code_2]["keypoints"]),
            )
            kpts1 = c1[0]["keypoints"][0].cpu().numpy()
            kpts2 = c2[rot_code_2]["keypoints"][0].cpu().numpy()
            indices = indices.cpu().numpy()

        mkpts1 = kpts1[indices[..., 0]].astype(np.float32)
        mkpts2 = kpts2[indices[..., 1]].astype(np.float32)

        try:
            _, inliers = cv2.findFundamentalMat(
                mkpts1, mkpts2, cv2.USAC_MAGSAC,
                ransacReprojThreshold=5, confidence=0.9999, maxIters=50000,
            )
            inliers = inliers.ravel() > 0
            return int(inliers.sum())
        except:
            return len(mkpts1)

    # Co-orientation at smallest resolution
    best_rot, best_n = 0, 0
    for rc2 in range(4):
        n = _extract_and_match(fname1, fname2, image_sizes[0], rc2)
        if n > best_n:
            best_rot = rc2
            best_n = n

    # Multi-resolution matching with best rotation
    total = best_n
    for sz in image_sizes[1:]:
        total += _extract_and_match(fname1, fname2, sz, best_rot)

    return total


# ---------------------------------------------------------------------------
# Ensemble: RoMa + ALIKED combined match count
# ---------------------------------------------------------------------------

def get_matching_flows_ensemble(fnames, aliked_extractor, lightglue_matcher,
                                roma_model, image_sizes=(1024, 1280, 1600),
                                roma_max_matches=10000):
    """
    Compute matching distance for all pairs using RoMa + ALIKED ensemble.

    Returns dict: flows[(fname_i, fname_j)] = int distance (lower = more similar).
    Drop-in replacement for the notebook's get_matching_flows().
    """
    index_pairs = []
    for i in range(len(fnames)):
        for j in range(i + 1, len(fnames)):
            index_pairs.append((i, j))

    aliked_cache = {}
    flows = {}

    for idx1, idx2 in tqdm(index_pairs, desc="Matching (RoMa+ALIKED ensemble)"):
        fname1, fname2 = fnames[idx1], fnames[idx2]

        # ALIKED matching
        n_aliked = aliked_pair_match(
            aliked_extractor, lightglue_matcher, aliked_cache,
            fname1, fname2, image_sizes,
        )

        # RoMa matching
        n_roma = roma_pair_match(roma_model, fname1, fname2, max_matches=roma_max_matches)

        # Ensemble: sum match counts → combined distance
        total_matches = n_aliked + n_roma
        total_matches = max(total_matches, 1)  # avoid div by zero
        dist = int((1 / total_matches) * 1e8)

        flows[(fname1, fname2)] = dist
        flows[(fname2, fname1)] = dist

    return flows
