"""
scripts/prepare_splits.py

Generates stratified, sequence-level train/val split manifests.

Strategy:
    - Parses filename pattern to extract (scene_type, sequence_id) keys.
    - Groups all image paths by sequence.
    - Performs stratified split: each scene/weather type is proportionally
      represented in both train and val.
    - Saves splits/train.txt and splits/val.txt (one image path per line).

Usage:
    python scripts/prepare_splits.py \\
        --data_dir datasets/obj_det_base \\
        --val_ratio 0.2 \\
        --seed 42

Why sequence-level split matters:
    This dataset consists of video sequences. Adjacent frames within a
    sequence are nearly identical — randomly splitting individual frames
    would place similar frames in both train and val, inflating mAP scores.
    By splitting at the sequence level, we guarantee no temporal leakage.
"""

import argparse
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# Filename pattern (two variants in the dataset):
#   augmented_raw_dataset_{scene}_{scene}_{seq_a}_{seq_b}_sequence.{frame}_step0.camera.{ext}
#   raw_dataset_{scene}_{scene}_{seq_a}_{seq_b}_sequence.{frame}_step0.camera.{ext}
_FILENAME_PATTERN = re.compile(
    r"(?:augmented_)?raw_dataset_"
    r"(?P<scene>[a-z]+_[a-z]+)"    # e.g. city_foggy
    r"_[a-z]+_[a-z]+_"              # repeated scene name (ignored)
    r"(?P<seq_a>\d+)_(?P<seq_b>\d+)_"
    r"sequence\.\d+_step0\.camera\.(png|jpg|jpeg)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stratified sequence-level train/val split manifests."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/obj_det_base",
        help="Directory containing all images and label files.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of sequences to use for validation. Default: 0.2 (80/20 split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="splits",
        help="Output directory for train.txt and val.txt.",
    )
    return parser.parse_args()


def parse_sequence_key(filename: str) -> Tuple[str, str]:
    """
    Extract (scene_type, sequence_id) from a dataset filename.

    Returns:
        scene_type  : e.g. "city_foggy"
        sequence_id : e.g. "0_1"
    """
    match = _FILENAME_PATTERN.match(filename)
    if match is None:
        return ("unknown", "unknown")
    scene = match.group("scene")           # e.g. "city_foggy"
    seq_id = f"{match.group('seq_a')}_{match.group('seq_b')}"  # e.g. "0_1"
    return scene, seq_id


def group_by_sequence(
    data_dir: str,
) -> Dict[Tuple[str, str], List[str]]:
    """
    Group all image paths by their (scene_type, sequence_id) key.

    Returns:
        Dict mapping (scene, seq_id) → list of absolute image paths.
    """
    data_path = Path(data_dir)
    image_extensions = {".png", ".jpg", ".jpeg"}

    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for f in sorted(data_path.iterdir()):
        if f.suffix.lower() not in image_extensions:
            continue
        scene, seq_id = parse_sequence_key(f.name)
        groups[(scene, seq_id)].append(str(f.resolve()))

    return dict(groups)


def stratified_split(
    groups: Dict[Tuple[str, str], List[str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Split sequence groups into train/val with stratification by scene type.

    Args:
        groups   : Dict of (scene, seq_id) → image paths.
        val_ratio: Fraction to allocate to validation.
        seed     : Random seed.

    Returns:
        train_paths, val_paths — flat lists of image file paths.
    """
    rng = random.Random(seed)

    # Group sequences by scene type for stratified splitting
    scene_to_seqs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for scene, seq_id in groups.keys():
        scene_to_seqs[scene].append((scene, seq_id))

    train_keys, val_keys = [], []

    for scene, seq_keys in scene_to_seqs.items():
        rng.shuffle(seq_keys)
        n_val = max(1, round(len(seq_keys) * val_ratio))
        val_keys.extend(seq_keys[:n_val])
        train_keys.extend(seq_keys[n_val:])

    # Flatten to image paths
    train_paths = []
    for key in train_keys:
        train_paths.extend(groups[key])

    val_paths = []
    for key in val_keys:
        val_paths.extend(groups[key])

    return sorted(train_paths), sorted(val_paths)


def write_manifest(paths: List[str], output_path: str) -> None:
    """Write image paths to a manifest text file (one path per line)."""
    with open(output_path, "w") as f:
        for p in paths:
            f.write(p + "\n")


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir.resolve()}")
        sys.exit(1)

    print(f"[*] Scanning dataset directory: {data_dir}")
    groups = group_by_sequence(args.data_dir)

    if not groups:
        print(f"[ERROR] No valid images matching expected patterns found in: {data_dir.resolve()}")
        sys.exit(1)

    n_sequences = len(groups)
    n_images = sum(len(v) for v in groups.values())
    scenes = sorted({k[0] for k in groups.keys()})

    print(f"    Found {n_images} images across {n_sequences} sequences")
    print(f"    Scene types ({len(scenes)}): {', '.join(scenes)}")

    print(f"\n[*] Splitting (val_ratio={args.val_ratio}, seed={args.seed})...")
    train_paths, val_paths = stratified_split(groups, args.val_ratio, args.seed)

    print(f"    Train: {len(train_paths)} images")
    print(f"    Val  : {len(val_paths)} images")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "train.txt"
    val_out = output_dir / "val.txt"

    write_manifest(train_paths, str(train_out))
    write_manifest(val_paths, str(val_out))

    print("")
    print("[OK] Manifests saved:")
    print("    " + str(train_out))
    print("    " + str(val_out))


if __name__ == "__main__":
    main()
