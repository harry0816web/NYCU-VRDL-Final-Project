"""LoMa pair-wise matching wrapper.

Two-phase approach (same as best.ipynb for performance):
  Phase 1: detect_and_describe ONCE per unique image, cache features
  Phase 2: for each pair, run LoMa matcher on cached features (very fast)

This avoids redundant feature extraction — if image A appears in 20 pairs,
detect_and_describe runs only once instead of 20 times.
"""

import sys
import os
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image

from tasks.matching.core.match import revert_rotate


def _ensure_loma_path(loma_args):
    """Add LoMa source to sys.path if not already present."""
    loma_code_path = loma_args.get("loma_code_path", None)
    if not loma_code_path:
        return
    # loma_code_path may point to LoMa repo root; the Python package is under src/
    src_path = os.path.join(loma_code_path, "src")
    if os.path.isdir(os.path.join(src_path, "loma")):
        path_to_add = src_path
    else:
        path_to_add = loma_code_path
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)


def _load_loma_model(loma_args, device_str):
    """Load LoMa model from config."""
    _ensure_loma_path(loma_args)

    from loma import LoMa
    from loma.loma import LoMaB, LoMaL, LoMaG, LoMaR, LoMaB128

    preset_map = {
        "loma_B": LoMaB,
        "loma_L": LoMaL,
        "loma_G": LoMaG,
        "loma_R": LoMaR,
        "loma_B128": LoMaB128,
    }

    preset_name = loma_args.get("preset", "loma_B")
    num_keypoints = loma_args.get("num_keypoints", 2048)
    filter_threshold = loma_args.get("filter_threshold", 0.1)

    preset_cls = preset_map.get(preset_name, LoMaB)

    weights_url = loma_args.get("weights_path", None)
    if weights_url and not weights_url.startswith(("http://", "https://", "file://")):
        weights_url = f"file://{weights_url}"

    cfg_kwargs = {}
    if weights_url:
        cfg_kwargs["weights_url"] = weights_url
    if num_keypoints != 2048:
        cfg_kwargs["num_keypoints"] = num_keypoints
    if filter_threshold != 0.1:
        cfg_kwargs["filter_threshold"] = filter_threshold

    if cfg_kwargs:
        import dataclasses
        preset_instance = preset_cls()
        field_dict = {f.name: getattr(preset_instance, f.name) for f in dataclasses.fields(preset_instance)
                      if f.name not in ("name",)}
        field_dict.update(cfg_kwargs)
        field_dict.pop("name", None)
        cfg = LoMa.Cfg(**field_dict)
    else:
        cfg = preset_cls()

    model = LoMa(cfg)
    model.eval()
    return model


def keypoint_matching_LoMa(
    image_pairs: list[tuple[str, str]],
    dir_pairs: list[tuple[int, int]],
    images_dir: str,
    loma_args: dict,
    num_keypoints: int = 2048,
    filter_threshold: float = 0.1,
    min_matches: int = 15,
    rects: dict = None,
    device: torch.device = torch.device("cpu"),
):
    """LoMa pair-wise matching with per-image feature caching.

    Phase 1: Extract features once per unique (image, direction, crop) combo.
    Phase 2: Match each pair using cached features.

    This mirrors best.ipynb's extract_loma_scene_features + loma_match_pair
    two-phase approach for optimal performance.
    """
    loma_args = dict(loma_args)
    loma_args.setdefault("num_keypoints", num_keypoints)
    loma_args.setdefault("filter_threshold", filter_threshold)

    _ensure_loma_path(loma_args)
    from loma.loma import filter_matches, to_pixel_coords

    loma_model = _load_loma_model(loma_args, str(device))

    # --- Phase 1: Extract features once per unique image ---
    # Collect all unique (image_key, direction) combos
    unique_images = {}  # key -> (dir, rect)
    for pair, dir_pair in zip(image_pairs, dir_pairs):
        key1, key2 = pair
        dir1, dir2 = dir_pair

        rect1 = _get_rect(rects, key1, key2)
        rect2 = _get_rect(rects, key2, key1)

        # Cache key includes direction and crop rect to handle different transforms
        cache_key1 = (key1, dir1, tuple(rect1))
        cache_key2 = (key2, dir2, tuple(rect2))
        unique_images[cache_key1] = (key1, dir1, rect1)
        unique_images[cache_key2] = (key2, dir2, rect2)

    # Extract features for each unique image
    feature_cache = {}  # cache_key -> {keypoints, descriptors, pixel_kpts, h, w, H_orig, W_orig}
    for cache_key, (img_key, direction, rect) in tqdm(
        unique_images.items(),
        desc="LoMa feature extraction",
    ):
        img = cv2.imread(str(images_dir / img_key), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Crop
        if rect != [0, 0, 0, 0]:
            img = img[rect[1]:rect[3], rect[0]:rect[2], :]

        # Rotate
        if direction == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif direction == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif direction == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        H_orig, W_orig = img.shape[:2]

        # Convert to tensor for LoMa (pass file path or tensor)
        with torch.inference_mode():
            kpts, desc, h, w = loma_model.detect_and_describe(
                str(images_dir / img_key),
                num_keypoints=loma_args.get("num_keypoints", num_keypoints),
            ) if rect == [0, 0, 0, 0] and direction == 0 else \
            loma_model.detect_and_describe(
                _img_to_tensor(img),
                num_keypoints=loma_args.get("num_keypoints", num_keypoints),
            )

        pixel_kpts = to_pixel_coords(kpts, h, w).detach().cpu().numpy()[0].astype(np.float32)

        feature_cache[cache_key] = {
            "keypoints": kpts.detach(),
            "descriptors": desc.detach(),
            "pixel_kpts": pixel_kpts,
            "h": h, "w": w,
            "H_orig": H_orig, "W_orig": W_orig,
        }

    # --- Phase 2: Match pairs using cached features (fast) ---
    keypoints = {}
    matches = {}

    for pair, dir_pair in tqdm(
        zip(image_pairs, dir_pairs),
        total=len(image_pairs),
        desc="LoMa matching",
    ):
        key1, key2 = pair
        dir1, dir2 = dir_pair

        rect1 = _get_rect(rects, key1, key2)
        rect2 = _get_rect(rects, key2, key1)

        cache_key1 = (key1, dir1, tuple(rect1))
        cache_key2 = (key2, dir2, tuple(rect2))

        if cache_key1 not in feature_cache or cache_key2 not in feature_cache:
            continue

        feat1 = feature_cache[cache_key1]
        feat2 = feature_cache[cache_key2]

        try:
            with torch.inference_mode():
                scores = loma_model(
                    feat1["keypoints"],
                    feat2["keypoints"],
                    feat1["descriptors"],
                    feat2["descriptors"],
                )["scores"]

            m0, _, _, _ = filter_matches(scores, loma_model.cfg.filter_threshold)
            valid = m0[0] > -1
            idx0 = torch.where(valid)[0]
            idx1 = m0[0][valid]

            kpts1 = feat1["pixel_kpts"][idx0.cpu().numpy()]
            kpts2 = feat2["pixel_kpts"][idx1.cpu().numpy()]

        except Exception as e:
            print(f"[LoMa] Failed on {key1}-{key2}: {e}")
            continue

        if kpts1.shape[0] < min_matches:
            continue

        # Revert Rotate
        kpts1 = revert_rotate(kpts1, dir1, feat1["W_orig"], feat1["H_orig"])
        kpts2 = revert_rotate(kpts2, dir2, feat2["W_orig"], feat2["H_orig"])

        # Revert Crop
        kpts1[:, 0] += rect1[0]
        kpts1[:, 1] += rect1[1]
        kpts2[:, 0] += rect2[0]
        kpts2[:, 1] += rect2[1]

        # Accumulate
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

    # Cleanup
    del feature_cache
    loma_model.cpu()
    del loma_model
    torch.cuda.empty_cache()

    return keypoints, matches


def _get_rect(rects, key, partner_key):
    """Extract crop rectangle for an image."""
    if rects is None or key not in rects:
        return [0, 0, 0, 0]
    r = rects[key]
    if isinstance(r, list):
        return r
    elif isinstance(r, dict):
        return r.get(partner_key, [0, 0, 0, 0])
    return [0, 0, 0, 0]


def _img_to_tensor(img_bgr, patch_size=14):
    """Convert BGR numpy image to float32 [1, 3, H, W] tensor in [0, 1].

    Resizes so H and W are multiples of patch_size (required by DINOv2 ViT).
    """
    h, w = img_bgr.shape[:2]
    new_h = max(patch_size, (h // patch_size) * patch_size)
    new_w = max(patch_size, (w // patch_size) * patch_size)
    if new_h != h or new_w != w:
        img_bgr = cv2.resize(img_bgr, (new_w, new_h))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb).permute(2, 0, 1).float()[None] / 255.0
