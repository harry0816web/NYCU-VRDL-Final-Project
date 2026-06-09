"""RoMa dense matching wrapper.

Follows the same pair-wise pattern as keypoint_matcing_DKM:
  - Load model once
  - For each pair: match → sample sparse → revert rotate/crop → accumulate into keypoints/matches dicts
  - Output format: keypoints[key] = ndarray [N, 2] pixel coords, matches[key1][key2] = ndarray [K, 2] index pairs

Weight files required (offline, no network on Kaggle):
  - roma_outdoor.pth  (~500 MB)
  - dinov2_vitl14_pretrain.pth  (~1.1 GB, shared with DINOv2 pairing)
"""

import os
import sys
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image

from tasks.matching.core.match import revert_rotate


def _load_roma_model(roma_args, device):
    """Load RoMa model from local weights (no internet)."""
    roma_code_path = roma_args.get("roma_code_path", None)
    if roma_code_path:
        sys.path.insert(0, roma_code_path)

    from romatch import roma_outdoor

    roma_weights_path = roma_args["roma_weights"]
    dinov2_weights_path = roma_args["dinov2_weights"]

    roma_weights = torch.load(roma_weights_path, map_location=device)
    dinov2_weights = torch.load(dinov2_weights_path, map_location=device)

    # RoMa requires highest precision for matmul
    torch.set_float32_matmul_precision("highest")

    coarse_res = roma_args.get("coarse_res", 560)
    upsample_res = roma_args.get("upsample_res", 864)

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


def keypoint_matcing_RoMa(
    image_pairs: list[tuple[str, str]],
    dir_pairs: list[tuple[int, int]],
    images_dir: str,
    roma_args: dict,
    max_matching_num: int = 10000,
    sample_thresh: float = 0.05,
    min_matches: int = 15,
    rects: dict = None,
    device: torch.device = torch.device("cpu"),
):
    """RoMa pair-wise dense matching → sparse keypoints + matches.

    Parameters
    ----------
    image_pairs : list of (key1, key2) filename pairs
    dir_pairs : list of (dir1, dir2) rotation directions (0/90/180/270)
    images_dir : path to image directory
    roma_args : dict with keys:
        roma_code_path, roma_weights, dinov2_weights, coarse_res, upsample_res
    max_matching_num : number of sparse matches to sample per pair
    sample_thresh : certainty threshold for RoMa sampling
    min_matches : minimum matches to keep a pair
    rects : optional crop rectangles dict
    device : torch device

    Returns
    -------
    keypoints : dict[str, ndarray]  — keypoints[image_name] = [N, 2] pixel coords
    matches : dict[str, dict[str, ndarray]]  — matches[key1][key2] = [K, 2] index pairs
    """
    roma_model = _load_roma_model(roma_args, device)
    roma_model.sample_thresh = sample_thresh

    keypoints = {}
    matches = {}

    for pair, dir_pair in tqdm(zip(image_pairs, dir_pairs), total=len(image_pairs), desc="RoMa matching"):
        key1, key2 = pair
        dir1, dir2 = dir_pair

        img1 = cv2.imread(str(images_dir / key1), cv2.IMREAD_COLOR)
        img2 = cv2.imread(str(images_dir / key2), cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            continue

        # --- Crop ---
        if rects is not None and key1 in rects:
            if isinstance(rects[key1], list):
                rect1 = rects[key1]
            elif isinstance(rects[key1], dict):
                rect1 = rects[key1][key2]
            else:
                rect1 = [0, 0, 0, 0]
            img1 = img1[rect1[1]:rect1[3], rect1[0]:rect1[2], :]
        else:
            rect1 = [0, 0, 0, 0]

        if rects is not None and key2 in rects:
            if isinstance(rects[key2], list):
                rect2 = rects[key2]
            elif isinstance(rects[key2], dict):
                rect2 = rects[key2][key1]
            else:
                rect2 = [0, 0, 0, 0]
            img2 = img2[rect2[1]:rect2[3], rect2[0]:rect2[2], :]
        else:
            rect2 = [0, 0, 0, 0]

        # --- Rotate ---
        if dir1 == 90:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        elif dir1 == 180:
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
        elif dir1 == 270:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if dir2 == 90:
            img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
        elif dir2 == 180:
            img2 = cv2.rotate(img2, cv2.ROTATE_180)
        elif dir2 == 270:
            img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # --- RoMa expects PIL RGB ---
        img1PIL = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2PIL = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        W_A, H_A = img1PIL.size
        W_B, H_B = img2PIL.size

        try:
            warp, certainty = roma_model.match(img1PIL, img2PIL)
            dense_matches, dense_certainty = roma_model.sample(warp, certainty, num=max_matching_num)
        except Exception as e:
            print(f"[RoMa] Failed on {key1}-{key2}: {e}")
            continue

        # Convert normalised [-1, 1] coords to pixel coords
        # dense_matches shape: [K, 4] → (x_A_norm, y_A_norm, x_B_norm, y_B_norm)
        kpts1 = dense_matches[:, :2].detach().cpu().numpy()
        kpts2 = dense_matches[:, 2:].detach().cpu().numpy()

        # [-1, 1] → pixel
        kpts1[:, 0] = (kpts1[:, 0] + 1) / 2 * (W_A - 1)
        kpts1[:, 1] = (kpts1[:, 1] + 1) / 2 * (H_A - 1)
        kpts2[:, 0] = (kpts2[:, 0] + 1) / 2 * (W_B - 1)
        kpts2[:, 1] = (kpts2[:, 1] + 1) / 2 * (H_B - 1)

        if kpts1.shape[0] < min_matches:
            continue

        # --- Revert Rotate ---
        kpts1 = revert_rotate(kpts1, dir1, W_A, H_A)
        kpts2 = revert_rotate(kpts2, dir2, W_B, H_B)

        # --- Revert Crop ---
        kpts1[:, 0] += rect1[0]
        kpts1[:, 1] += rect1[1]
        kpts2[:, 0] += rect2[0]
        kpts2[:, 1] += rect2[1]

        # --- Accumulate (same logic as DKM) ---
        if key1 not in keypoints:
            keypoints[key1] = kpts1
            matches1 = np.arange(kpts1.shape[0])
        else:
            n = keypoints[key1].shape[0]
            keypoints[key1] = np.concatenate([keypoints[key1], kpts1])
            matches1 = np.arange(kpts1.shape[0]) + n

        if key2 not in keypoints:
            keypoints[key2] = kpts2
            matches2 = np.arange(kpts2.shape[0])
        else:
            n = keypoints[key2].shape[0]
            keypoints[key2] = np.concatenate([keypoints[key2], kpts2])
            matches2 = np.arange(kpts2.shape[0]) + n

        _matches = np.stack([matches1, matches2], axis=1).astype(np.int64)
        matches.setdefault(key1, {})
        matches[key1][key2] = _matches

    # Free GPU
    roma_model.cpu()
    del roma_model
    torch.cuda.empty_cache()

    return keypoints, matches
