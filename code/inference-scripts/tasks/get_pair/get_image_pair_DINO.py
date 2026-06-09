"""
Image pair retrieval using DINOv3 (global CLS descriptors + same filtering logic
as the original DINOv2 / HF script).

Pipeline contract (unchanged):
  - ``params["pair_matching_args"]``: passed to ``get_image_pairs`` (see below).
  - ``params["output"]``: CSV filename under ``work_dir``.
  - CSV columns: ``key1``, ``key2``, ``dir1``, ``dir2``, ``sim``, ``match_num``
    (keys are image *basenames*, same as ``get_image_pair_exhaustive``).

Weights:
  After the usual ``os.path.join(input_dir_root, pair_matching_args["model_name"])``,
  ``model_name`` must be a path to a ``.pth`` checkpoint (offline DINOv3).

Optional ``pair_matching_args`` keys:
  - ``dinov3_arch`` (str): ``vits16`` | ``vits16plus`` | ``vitb16`` | ``vitl16`` (default ``vitb16``).
  - ``batch_size`` (int): default ``32``.
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def _ensure_dinov3_on_path() -> None:
    extra = os.environ.get("DINOV3_REPO_ROOT", "").strip()
    candidates: List[str] = []
    if extra:
        candidates.append(extra)
    candidates.extend(
        [
            "/kaggle/input/dinov3",
            "/kaggle/input/dinov3-code",
            "/kaggle/working/dinov3",
        ]
    )
    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            if os.path.isdir(os.path.join(c, "dinov3")):
                sys.path.insert(0, c)
                return


def _load_backbone(arch: str, weights_path: str, device: torch.device) -> torch.nn.Module:
    _ensure_dinov3_on_path()
    from dinov3.hub import backbones as dinov3_backbones

    name = (arch or "vitb16").lower().strip()
    builders = {
        "vits16": dinov3_backbones.dinov3_vits16,
        "vits16plus": dinov3_backbones.dinov3_vits16plus,
        "vitb16": dinov3_backbones.dinov3_vitb16,
        "vitl16": dinov3_backbones.dinov3_vitl16,
    }
    if name not in builders:
        raise ValueError(f"Unknown dinov3_arch '{arch}'. Choose from {sorted(builders)}.")
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"DINOv3 weights not found at {weights_path!r}. "
            "Set pair_matching_args['model_name'] (relative to input_dir_root) to a .pth file."
        )
    net = builders[name](pretrained=True, weights=weights_path)
    return net.to(device).eval()


def _build_eval_transform():
    _ensure_dinov3_on_path()
    from dinov3.data.transforms import make_classification_eval_transform

    return make_classification_eval_transform()


@torch.inference_mode()
def _embed_images(
    paths: Sequence[Path],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """L2-normalized descriptors (N, D) on CPU float32."""
    feats: List[torch.Tensor] = []
    use_amp = device.type == "cuda"
    for start in tqdm(range(0, len(paths), batch_size), desc="DINOv3 global descriptors"):
        batch_paths = paths[start : start + batch_size]
        tensors: List[torch.Tensor] = []
        for p in batch_paths:
            with Image.open(p) as im:
                im = im.convert("RGB")
            tensors.append(transform(im))
        x = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model.forward_features(x)
            cls = out["x_norm_clstoken"].float()
        cls = F.normalize(cls, dim=-1, eps=1e-12)
        feats.append(cls.cpu())
    return torch.cat(feats, dim=0)


def embed_images_dinov3(
    paths: list[Path],
    weights_path: str,
    arch: str = "vitb16",
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Image embeddings of shape ``[len(paths), D]`` (L2-normalized)."""
    model = _load_backbone(arch, weights_path, device)
    transform = _build_eval_transform()
    return _embed_images(paths, model, transform, device, batch_size=batch_size)


def get_pairs_exhaustive(lst: list[Any]) -> list[tuple[int, int]]:
    """All index pairs."""
    return list(itertools.combinations(range(len(lst)), 2))


def get_image_pairs(
    paths: list[Path],
    model_name: str,
    similarity_threshold: float = 0.6,
    tolerance: int = 1000,
    min_matches: int = 20,
    exhaustive_if_less: int = 20,
    p: float = 2.0,
    device: torch.device = torch.device("cpu"),
    dinov3_arch: str = "vitb16",
    batch_size: int = 32,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    ``model_name``: absolute path to DINOv3 ``.pth`` weights (offline).

    Same distance / threshold behaviour as the original DINO task: ``torch.cdist``
    on L2-normalized CLS features, then threshold + min_matches + tolerance.
    """
    if len(paths) <= exhaustive_if_less:
        pairs = get_pairs_exhaustive(paths)
        distances = np.zeros((len(paths), len(paths)))
        return pairs, distances

    embeddings = embed_images_dinov3(
        paths,
        weights_path=model_name,
        arch=dinov3_arch,
        batch_size=batch_size,
        device=device,
    )
    distances = torch.cdist(embeddings, embeddings, p=p)

    mask = distances <= similarity_threshold
    image_indices = np.arange(len(paths))
    matches: list[tuple[int, int]] = []

    for current_image_index in range(len(paths)):
        mask_row = mask[current_image_index]
        indices_to_match = image_indices[mask_row.cpu().numpy()]

        if len(indices_to_match) < min_matches:
            indices_to_match = np.argsort(distances[current_image_index].cpu().numpy())[
                :min_matches
            ]

        for other_image_index in indices_to_match:
            if other_image_index == current_image_index:
                continue
            if distances[current_image_index, other_image_index] < tolerance:
                matches.append(
                    tuple(sorted((current_image_index, int(other_image_index))))
                )

    pairs = sorted(list(set(matches)))
    distances_np = distances.cpu().numpy()
    return pairs, distances_np


def task_get_image_pair_DINO(params):
    if params["pdb"]:
        import pdb

        pdb.set_trace()

    image_paths = params["data_dict"]
    print(f"image_num = {len(image_paths)}")
    input_dir_root = params["input_dir_root"]
    pair_args = dict(params["pair_matching_args"])
    pair_args["model_name"] = os.path.join(input_dir_root, pair_args["model_name"])

    dinov3_arch = str(pair_args.pop("dinov3_arch", "vitb16"))
    batch_size = int(pair_args.pop("batch_size", 32))

    pairs, distances = get_image_pairs(
        image_paths,
        dinov3_arch=dinov3_arch,
        batch_size=batch_size,
        **pair_args,
        device=params["device"],
    )
    print(f"pair_num = {len(pairs)}")

    res = {
        "key1": [],
        "key2": [],
        "dir1": [],
        "dir2": [],
        "sim": [],
        "match_num": [],
    }
    for pair in pairs:
        p1, p2 = pair
        res["key1"].append(image_paths[p1].name)
        res["key2"].append(image_paths[p2].name)
        res["dir1"].append(0)
        res["dir2"].append(0)
        res["sim"].append(float(distances[p1][p2]))
        res["match_num"].append(0)
    res_df = pd.DataFrame.from_dict(res)

    work_dir = Path(params["work_dir"])
    output_path = work_dir / params["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_path, index=False)
    print(f"save -> {output_path}")


if __name__ == "__main__":
    # Example (requires valid ``.pth`` and ``dinov3`` on PYTHONPATH):
    # images_list = list(Path(".../images").glob("*.jpg"))[:10]
    # pairs, dist = get_image_pairs(
    #     images_list,
    #     model_name="/path/to/dinov3_vitb16_pretrain.pth",
    #     exhaustive_if_less=0,
    #     device=torch.device("cuda"),
    # )
    pass
