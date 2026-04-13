#!/usr/bin/env python
"""
Create collages from inference output images.

Recursively searches all subdirectories in a samples directory,
groups face/edge image pairs, and creates collages (4x6 grid, 24 samples max per collage).
"""

import argparse
from pathlib import Path

from PIL import Image


def create_collages_for_directory(
    directory: Path, collages_output_dir: Path, dry_run: bool = False
):
    """
    Create collages for all face/edge image pairs in the given directory.
    Saves collages to collages_output_dir with directory name prefix.
    Returns tuple of (num_images, num_collages_created).
    """
    face_images = sorted(directory.glob("*_face.png"))

    if not face_images:
        return 0, 0

    print(f"\n📁 Processing {directory.name} ({len(face_images)} face images)...")

    sample_images = []
    for face_img_path in face_images:
        edge_img_path = directory / face_img_path.name.replace("_face.png", "_edge.png")

        if not edge_img_path.exists():
            print(f"  ⚠️ Missing edge image for {face_img_path.name}")
            continue

        try:
            face_img = Image.open(face_img_path)
            edge_img = Image.open(edge_img_path)

            # Downscale to 60%
            scale = 0.6
            face_img = face_img.resize(
                (int(face_img.width * scale), int(face_img.height * scale)),
                Image.Resampling.LANCZOS,
            )
            edge_img = edge_img.resize(
                (int(edge_img.width * scale), int(edge_img.height * scale)),
                Image.Resampling.LANCZOS,
            )

            # Concatenate horizontally
            concat = Image.new(
                "RGB", (face_img.width + edge_img.width, face_img.height)
            )
            concat.paste(face_img, (0, 0))
            concat.paste(edge_img, (face_img.width, 0))
            sample_images.append(concat)
        except Exception as e:
            print(f"  ⚠️ Error processing {face_img_path.name}: {e}")

    # Create multiple 4x6 collages if necessary
    if not sample_images:
        print("  ⚠️ No images successfully processed for collage")
        return len(face_images), 0

    cols, rows = 4, 6
    samples_per_collage = cols * rows  # 24
    img_w, img_h = sample_images[0].width, sample_images[0].height

    num_collages = (len(sample_images) + samples_per_collage - 1) // samples_per_collage

    for collage_idx in range(num_collages):
        start_idx = collage_idx * samples_per_collage
        end_idx = min(start_idx + samples_per_collage, len(sample_images))
        collage_samples = sample_images[start_idx:end_idx]

        collage = Image.new("RGB", (img_w * cols, img_h * rows))

        for idx, img in enumerate(collage_samples):
            x = (idx % cols) * img_w
            y = (idx // cols) * img_h
            collage.paste(img, (x, y))

        # Construct collage filename with directory name prefix
        dir_name = directory.name
        if num_collages > 1:
            collage_name = f"collage_{dir_name}_{collage_idx:02d}.jpg"
        else:
            collage_name = f"collage_{dir_name}.jpg"

        collage_path = collages_output_dir / collage_name

        # Check if collage already exists
        if collage_path.exists():
            print(f"  ⏭️  Collage {collage_idx + 1}/{num_collages} already exists: {collage_name}")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would save: {collage_path}")
        else:
            collage.save(collage_path, format="JPEG", quality=70, optimize=True)
            print(f"  ✅ Collage {collage_idx + 1}/{num_collages}: {collage_name}")

    return len(sample_images), num_collages


def main():
    parser = argparse.ArgumentParser(
        description="Create collages from inference output images (recursively processes subdirectories)"
    )
    parser.add_argument(
        "samples_dir",
        type=Path,
        help="Root directory containing samples (will process all subdirectories)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating collages",
    )

    args = parser.parse_args()

    samples_dir = args.samples_dir.resolve()

    if not samples_dir.exists():
        print(f"❌ Error: Directory does not exist: {samples_dir}")
        return 1

    if not samples_dir.is_dir():
        print(f"❌ Error: Path is not a directory: {samples_dir}")
        return 1

    # Create collages subdirectory
    collages_dir = samples_dir / "collages"
    if not args.dry_run:
        collages_dir.mkdir(exist_ok=True, parents=True)

    print(f"{'=' * 80}")
    print(f"Creating collages from: {samples_dir}")
    print(f"Output directory: {collages_dir}")
    print(f"Dry-run mode: {args.dry_run}")
    print(f"{'=' * 80}")

    # Recursively process all subdirectories
    total_images = 0
    total_collages = 0

    # First, check root directory (skip if it's empty or only contains the collages dir)
    root_images, root_collages = create_collages_for_directory(
        samples_dir, collages_dir, args.dry_run
    )
    if root_images > 0:
        total_images += root_images
        total_collages += root_collages

    # Then, process subdirectories (but skip the collages directory itself)
    for subdir in sorted(samples_dir.iterdir()):
        if subdir.is_dir() and subdir.name != "collages":
            num_images, num_collages = create_collages_for_directory(
                subdir, collages_dir, args.dry_run
            )
            total_images += num_images
            total_collages += num_collages

    print(f"\n{'=' * 80}")
    print("Summary:")
    print(
        f"  📊 Total images processed: {total_images // 2}"
    )  # Divide by 2 since each sample has face+edge
    print(f"  🎨 Total collages created: {total_collages}")
    print(f"  📁 Collages saved to: {collages_dir}")
    if args.dry_run:
        print("  [DRY RUN MODE - No files were actually created]")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    exit(main())
