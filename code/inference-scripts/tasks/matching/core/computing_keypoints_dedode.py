"""DeDoDe keypoint detection — drop-in replacement interface for detect_keypoints()."""

from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2


# ImageNet normalisation (required by DeDoDe)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def _to_pixel_coords(norm_coords, H, W):
    """Convert DeDoDe normalised coords [-1, 1] → pixel coords.

    DeDoDe convention: x maps to width, y maps to height.
    pixel_x = W * (x + 1) / 2
    pixel_y = H * (y + 1) / 2
    """
    pixel_coords = norm_coords.clone()
    pixel_coords[..., 0] = W * (norm_coords[..., 0] + 1) / 2
    pixel_coords[..., 1] = H * (norm_coords[..., 1] + 1) / 2
    return pixel_coords


def _load_and_preprocess(path, H, W, device, dtype=torch.float32):
    """Load image via PIL, resize to (W, H), return tensor + original size."""
    pil_im = Image.open(path).convert("RGB")
    orig_W, orig_H = pil_im.size

    pil_resized = pil_im.resize((W, H), Image.BILINEAR)
    tensor = transforms.ToTensor()(pil_resized).to(device, dtype)   # [3, H, W]
    tensor = _normalize(tensor)
    return tensor, orig_W, orig_H


def detect_keypoints_dedode(
    paths,
    detector,
    descriptor,
    num_keypoints=10000,
    H=784,
    W=784,
    rects=None,
    dynamic_resize=None,
    dtype="float32",
    device=torch.device("cpu"),
):
    """Detect keypoints with DeDoDe detector + descriptor.

    Returns
    -------
    keypoints : dict[str, ndarray]   — {key: [N, 2] pixel coords in original image}
    descriptors : dict[str, ndarray] — {key: [N, 256]}

    The output format is identical to ``detect_keypoints()`` so downstream
    h5 saving / postprocessing works unchanged.
    """
    if dtype == "float16":
        _dtype = torch.float16
    else:
        _dtype = torch.float32

    keypoints_dict = {}
    descriptors_dict = {}

    for data in tqdm(paths, desc="Computing keypoints (DeDoDe)"):
        if type(data) == tuple:
            path = data[0]
            direction = data[1]
        else:
            path = data
            direction = -1

        key = path.name
        if direction != -1:
            key = key + f" {direction}"

        with torch.inference_mode():
            # --- Load image via OpenCV (same as ALIKED path for consistency) ---
            img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Crop
            if rects is not None and path.name in rects:
                rect = rects[path.name]
                img_rgb = img_rgb[rect[1]:rect[3], rect[0]:rect[2]]

            # Rotate
            if direction == 90:
                img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
            elif direction == 180:
                img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_180)
            elif direction == 270:
                img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

            orig_H_img, orig_W_img = img_rgb.shape[:2]

            # Determine resize target
            resize_H, resize_W = H, W
            if dynamic_resize is not None:
                max_edge = max(orig_H_img, orig_W_img)
                best_size = min(dynamic_resize, key=lambda x: abs(x - max_edge))
                resize_H = resize_W = best_size

            # Resize to fixed size + ImageNet normalize
            pil_resized = Image.fromarray(img_rgb).resize((resize_W, resize_H), Image.BILINEAR)
            tensor = transforms.ToTensor()(pil_resized).to(device, _dtype)  # [3, H, W]
            tensor = _normalize(tensor)

            # Detect keypoints (normalised coords)
            batch = {"image": tensor[None]}  # [1, 3, H, W]
            detections = detector.detect(batch, num_keypoints=num_keypoints)
            kpts_norm = detections["keypoints"]  # [1, N, 2] in [-1, 1]

            # Describe keypoints
            descs = descriptor.describe_keypoints(batch, kpts_norm)["descriptions"]  # [1, N, 256]

            # Normalised coords → pixel coords (relative to resized image)
            kpts_pixel = _to_pixel_coords(kpts_norm, resize_H, resize_W)  # [1, N, 2]

            # Scale back to original image coordinates
            scale_x = orig_W_img / resize_W
            scale_y = orig_H_img / resize_H
            kpts_pixel[..., 0] *= scale_x
            kpts_pixel[..., 1] *= scale_y

            kp_np = kpts_pixel.squeeze(0).cpu().numpy().astype(np.float32)   # [N, 2]
            desc_np = descs.squeeze(0).cpu().numpy().astype(np.float32)       # [N, 256]

            # Shape guard
            kp_np = kp_np.reshape((-1, 2))
            desc_np = desc_np.reshape((-1, 256))

            # Range check (clip to image bounds)
            mask = (
                (kp_np[:, 0] >= 0)
                & (kp_np[:, 0] < orig_W_img)
                & (kp_np[:, 1] >= 0)
                & (kp_np[:, 1] < orig_H_img)
            )
            kp_np = kp_np[mask]
            desc_np = desc_np[mask]

            keypoints_dict[key] = kp_np
            descriptors_dict[key] = desc_np

    return keypoints_dict, descriptors_dict
