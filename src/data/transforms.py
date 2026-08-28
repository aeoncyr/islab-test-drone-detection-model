"""
src/data/transforms.py

Image augmentation transforms for drone detection training.

All transforms are implemented from scratch using PyTorch/PIL.
They support both image and bounding box (in YOLO normalized format)
simultaneous transformation to maintain annotation correctness.

Transforms:
    RandomHorizontalFlip : Mirrors image and flips x-coordinates of boxes.
    ColorJitter          : Randomly perturbs brightness, contrast, saturation, hue.
    MultiScaleResize     : Randomly resizes input to one of several sizes.
    Compose              : Chains multiple transforms together.
    build_transforms     : Factory that builds a transform pipeline from config.
"""

import random
from typing import List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class RandomHorizontalFlip:
    """
    Randomly flip the image and bounding boxes horizontally.

    Boxes are expected in normalized YOLO format: [class_id, cx, cy, w, h].
    After horizontal flip: new_cx = 1.0 - cx.

    Args:
        p: Probability of applying the flip. Default: 0.5.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        image: Image.Image,
        boxes: List[List[float]],
    ) -> Tuple[Image.Image, List[List[float]]]:
        if random.random() < self.p:
            image = TF.hflip(image)
            boxes = [
                [b[0], 1.0 - b[1], b[2], b[3], b[4]]
                for b in boxes
            ]
        return image, boxes


class ColorJitter:
    """
    Randomly perturb image color properties.

    Applies random changes to brightness, contrast, saturation, and hue.
    The order of operations is randomized for stronger augmentation diversity.

    Args:
        brightness: Max brightness change as a fraction. Default: 0.4.
        contrast  : Max contrast change as a fraction. Default: 0.4.
        saturation: Max saturation change as a fraction. Default: 0.4.
        hue       : Max hue shift as a fraction of 0.5. Default: 0.1.
        p         : Probability of applying. Default: 0.8.
    """

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        p: float = 0.8,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(
        self,
        image: Image.Image,
        boxes: List[List[float]],
    ) -> Tuple[Image.Image, List[List[float]]]:
        if random.random() < self.p:
            # Randomize jitter values
            bri = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            con = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            sat = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            hue = random.uniform(-self.hue, self.hue)

            # Apply in random order
            ops = [
                lambda img: TF.adjust_brightness(img, bri),
                lambda img: TF.adjust_contrast(img, con),
                lambda img: TF.adjust_saturation(img, sat),
                lambda img: TF.adjust_hue(img, hue),
            ]
            random.shuffle(ops)
            for op in ops:
                image = op(image)

        return image, boxes


class MultiScaleResize:
    """
    Randomly resize the image to one of several fixed sizes.

    Boxes are in normalized coordinates, so they require no update.
    The final image is always a square (size × size).

    Args:
        sizes: List of possible output sizes. Default: [320, 352, 384, 416, 448].
        p    : Probability of applying a non-default size. Default: 0.5.
    """

    def __init__(
        self,
        sizes: Optional[List[int]] = None,
        p: float = 0.5,
    ) -> None:
        self.sizes = sizes or [320, 352, 384, 416, 448]
        self.p = p

    def __call__(
        self,
        image: Image.Image,
        boxes: List[List[float]],
    ) -> Tuple[Image.Image, List[List[float]]]:
        if random.random() < self.p:
            size = random.choice(self.sizes)
            image = image.resize((size, size), Image.BILINEAR)
        return image, boxes


class Compose:
    """
    Chain multiple transforms in sequence.

    Args:
        transforms: List of transform callables.
    """

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(
        self,
        image: Image.Image,
        boxes: List[List[float]],
    ) -> Tuple[Image.Image, List[List[float]]]:
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes


def build_transforms(aug_cfg: dict, is_train: bool = True) -> Optional[Compose]:
    """
    Build a transform pipeline from the augmentation config dict.

    Args:
        aug_cfg : Augmentation config sub-dict (from YAML).
        is_train: If False, returns None (no augmentation during eval).

    Returns:
        A Compose transform pipeline, or None for validation.
    """
    if not is_train:
        return None

    t_list = []

    if aug_cfg.get("random_hflip", True):
        t_list.append(RandomHorizontalFlip(p=0.5))

    if "color_jitter" in aug_cfg:
        cj = aug_cfg["color_jitter"]
        t_list.append(ColorJitter(
            brightness=cj.get("brightness", 0.4),
            contrast=cj.get("contrast", 0.4),
            saturation=cj.get("saturation", 0.4),
            hue=cj.get("hue", 0.1),
        ))

    if aug_cfg.get("multi_scale", False):
        t_list.append(MultiScaleResize())

    return Compose(t_list) if t_list else None
