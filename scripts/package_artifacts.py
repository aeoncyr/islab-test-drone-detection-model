"""
scripts/package_artifacts.py

Utility script to package all training checkpoints, evaluation tables,
split manifests, visualization plots, configurations, and source code into
a single, clean, compressed submission archive.

Usage:
    python scripts/package_artifacts.py --output drone_detection_submission_artifacts.zip
"""

import argparse
import os
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package drone detection models, plots, and tables into a submission zip."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Root directory of the project. Default: current directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="drone_detection_submission_artifacts.zip",
        help="Output zip filename. Default: drone_detection_submission_artifacts.zip",
    )
    return parser.parse_args()


def package_artifacts(root_dir: str, output_path: str) -> None:
    root = Path(root_dir).resolve()
    out_file = Path(output_path).resolve()

    artifacts_to_include = []

    # 1. Runs & Checkpoints & Histories
    runs_dir = root / "runs"
    if runs_dir.exists():
        for p in runs_dir.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                artifacts_to_include.append(p)

    # 2. Split manifests
    splits_dir = root / "splits"
    if splits_dir.exists():
        for p in splits_dir.rglob("*"):
            if p.is_file():
                artifacts_to_include.append(p)

    # 3. YAML configs & dataset descriptors
    for yaml_file in list(root.glob("*.yaml")) + list(root.glob("*.yml")) + list((root / "configs").rglob("*.yaml")):
        if yaml_file.is_file():
            artifacts_to_include.append(yaml_file)

    # 4. Plots & Visualizations
    for img_file in list(root.glob("*.png")) + list(root.glob("*.jpg")) + list(root.glob("*.jpeg")):
        if img_file.is_file():
            artifacts_to_include.append(img_file)

    # 5. Tables & Reports
    for doc_file in list(root.glob("*.csv")) + list(root.glob("*.md")) + list(root.glob("*.json")):
        if doc_file.is_file() and doc_file.name != out_file.name:
            artifacts_to_include.append(doc_file)

    # 6. Source code for reproducibility
    src_dir = root / "src"
    if src_dir.exists():
        for p in src_dir.rglob("*.py"):
            if p.is_file() and "__pycache__" not in p.parts:
                artifacts_to_include.append(p)

    # De-duplicate
    artifacts_to_include = sorted(list(set(artifacts_to_include)))

    print(f"[*] Packaging {len(artifacts_to_include)} artifacts into: {out_file.name} ...")

    with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in artifacts_to_include:
            try:
                rel_path = file_path.relative_to(root)
            except ValueError:
                rel_path = file_path.name
            zf.write(file_path, arcname=str(rel_path))

    zip_size_mb = out_file.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 65)
    print("             SUBMISSION ARTIFACTS ARCHIVE READY")
    print("=" * 65)
    print(f"  Archive Name : {out_file.name}")
    print(f"  Archive Size : {zip_size_mb:.2f} MB")
    print(f"  Total Files  : {len(artifacts_to_include)}")
    print("=" * 65)

    ckpts = [p for p in artifacts_to_include if p.suffix in ('.pth', '.pt')]
    plots = [p for p in artifacts_to_include if p.suffix in ('.png', '.jpg', '.jpeg')]
    tables = [p for p in artifacts_to_include if p.suffix in ('.csv', '.md')]
    print(f"  ✓ Checkpoints ({len(ckpts)}): {[c.name for c in ckpts]}")
    print(f"  ✓ Plots & Figures ({len(plots)}): {[p.name for p in plots]}")
    print(f"  ✓ Tables & Manifests ({len(tables)}): {[t.name for t in tables]}")
    print("=" * 65)
    print(f"[✓] Package saved successfully at: {out_file}")


def main() -> None:
    args = parse_args()
    package_artifacts(args.root_dir, args.output)


if __name__ == "__main__":
    main()
