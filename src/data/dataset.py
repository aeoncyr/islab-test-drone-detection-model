"""
src/data/dataset.py

DroneYOLODataset — Dataset loader for UAV drone detection.

Responsibilities (Clean OOP separation):
    - Load images and raw bounding box annotations from disk.
    - Apply spatial + color augmentations (via transforms.py).
    - Return (image_tensor, raw_boxes, labels) tuples.

Target assignment (FCOS-style) is intentionally NOT here.
It is the responsibility of the TargetAssigner in src/engine/target_assigner.py,
which is called in the DataLoader collate function.
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import Compose


class DroneYOLODataset(Dataset):
    """
    Dataset for loading drone images with YOLO-format annotations.

    Images and labels are discovered via a text manifest file, where each
    line is the absolute or relative path to one image. The corresponding
    label file is expected in the same directory with a `.txt` extension.

    YOLO label format (per line):
        <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    All values are normalized to [0, 1] relative to image dimensions.

    Args:
        manifest_path: Path to the .txt file listing image paths (one per line).
        input_size   : Target square resolution. Images are resized to this. Default: 416.
        transforms   : Optional Compose of augmentation transforms. Default: None.
        cache_images : Whether to cache images in memory. Default: True.
    """

    def __init__(
        self,
        manifest_path: str,
        input_size: int = 416,
        transforms: Optional[Compose] = None,
        cache_images: bool = True,
    ) -> None:
        self.input_size = input_size
        self.transforms = transforms
        self.cache_images = cache_images

        manifest = Path(manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_path}\n"
                f"Run: python scripts/prepare_splits.py first."
            )

        with open(manifest, "r") as f:
            self.image_paths: List[Path] = [
                Path(line.strip()) for line in f if line.strip()
            ]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in manifest: {manifest_path}")

        # In-memory RAM cache for Kaggle I/O bottleneck
        self.cache = {}
        if cache_images:
            print(f"[*] Caching {len(self.image_paths)} images (resized to {input_size}x{input_size}) in RAM to bypass disk I/O...")
            from concurrent.futures import ThreadPoolExecutor
            failed = []
            def _cache_img(p):
                try:
                    img = Image.open(p).convert("RGB")
                    img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
                    return p, img, None
                except Exception as e:
                    return p, None, str(e)
            with ThreadPoolExecutor(max_workers=8) as exc:
                for p, img, err in exc.map(_cache_img, self.image_paths):
                    if img is not None:
                        self.cache[p] = img
                    else:
                        failed.append((p, err))
            if failed:
                warnings.warn(
                    f"[Dataset] Failed to cache {len(failed)} images "
                    f"(first: {failed[0][0]} — {failed[0][1]}). "
                    f"They will be loaded on-the-fly.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> Image.Image:
        """Load image and ensure RGB format. Falls back gracefully on corrupt files."""
        if path in self.cache:
            # Return a copy to avoid unintended mutation by transforms
            return self.cache[path].copy()
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            warnings.warn(
                f"[Dataset] Failed to load image {path}: {e}. Returning blank placeholder.",
                RuntimeWarning,
                stacklevel=2,
            )
            return Image.new("RGB", (self.input_size, self.input_size), (0, 0, 0))

    def _load_labels(self, img_path: Path) -> Tuple[List[List[float]], List[int]]:
        """
        Load YOLO-format labels for a given image path.

        Returns:
            boxes : List of [cx_norm, cy_norm, w_norm, h_norm].
            labels: List of integer class IDs.
        """
        label_path = Path(img_path).with_suffix(".txt")
        boxes, labels = [], []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue
                    if len(parts) != 5:
                        warnings.warn(
                            f"[Dataset] Malformed label line {line_num} in {label_path}: "
                            f"expected 5 fields, got {len(parts)}. Skipping.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    try:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                    except ValueError as e:
                        warnings.warn(
                            f"[Dataset] Unparseable label line {line_num} in {label_path}: {e}. Skipping.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    # Clamp to [0, 1] range to guard against bad annotations
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w  = max(0.0, min(1.0, w))
                    h  = max(0.0, min(1.0, h))
                    if w <= 0.0 or h <= 0.0:
                        continue  # Skip degenerate boxes
                    boxes.append([cx, cy, w, h])
                    labels.append(class_id)

        return boxes, labels

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load, augment, and return one sample.

        Returns:
            image  : Float tensor (3, input_size, input_size), normalized [0, 1].
            boxes  : Float tensor (N, 4) in NORMALIZED (cx, cy, w, h) format.
                     Returns shape (0, 4) if no objects.
            labels : Long tensor (N,) with class IDs.
        """
        img_path = self.image_paths[idx]
        image = self._load_image(img_path)
        boxes, labels = self._load_labels(img_path)

        # Apply augmentations (on PIL image, before tensor conversion).
        # Always augment — including background-only images — to avoid
        # the distribution shift where no-box images are never flipped/jittered.
        if self.transforms is not None:
            packed = [[labels[i]] + boxes[i] for i in range(len(boxes))]
            image, packed = self.transforms(image, packed)
            if packed:
                labels = [int(b[0]) for b in packed]
                boxes = [b[1:] for b in packed]
            else:
                labels, boxes = [], []

        # Resize to target square resolution only if not already the correct size.
        # MultiScaleResize (in transforms) may have resized to a different scale;
        # we honour that and only enforce input_size if no transform ran.
        if image.size != (self.input_size, self.input_size):
            image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        # Convert to tensor without float conversion (fix non-writable warning)
        image_tensor = torch.from_numpy(
            np.array(image, dtype=np.uint8, copy=True)
        ).permute(2, 0, 1)

        # Convert annotations to tensors
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        return image_tensor, boxes_tensor, labels_tensor
