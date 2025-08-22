#!/usr/bin/env python3
"""
Copy 100 random images from each class folder into Data_Samples/<Class>_Sample.
Source root: /home/user/Desktop/Tsoutsas/myopia_project/data
Classes: Normal, High_Myopia, Pathological_Myopia
"""

from pathlib import Path
import shutil
import random
import sys

# ------------------ CONFIG ------------------
BASE_DIR = Path("/home/user/Desktop/Tsoutsas/myopia_project/data")
CLASSES = ["Normal", "High_Myopia", "Pathological_Myopia"]
NUM_PER_CLASS = 100
DEST_BASE = BASE_DIR / "Data_Samples"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RANDOM_SEED = 42  # Set None for non-deterministic
# --------------------------------------------

def find_images(folder: Path):
    # Non-recursive: use folder.iterdir(); change to rglob("*") if you need recursive
    return [p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def safe_copy(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    if not target.exists():
        shutil.copy2(src, target)
        return target

    # Avoid collisions by appending _1, _2, ...
    i = 1
    while True:
        candidate = dst_dir / f"{src.stem}_{i}{src.suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        i += 1

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # Validate source folders
    missing = [c for c in CLASSES if not (BASE_DIR / c).exists()]
    if missing:
        print(f"Error: Missing class folders under {BASE_DIR}: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    DEST_BASE.mkdir(parents=True, exist_ok=True)

    summary = []
    for cls in CLASSES:
        src_dir = BASE_DIR / cls
        dst_dir = DEST_BASE / f"{cls}_Sample"

        images = find_images(src_dir)
        if not images:
            print(f"Warning: No images found in {src_dir}")
            summary.append((cls, 0))
            continue

        k = min(NUM_PER_CLASS, len(images))
        if len(images) < NUM_PER_CLASS:
            print(f"Note: {cls} has only {len(images)} images. Copying all of them.")
            chosen = images[:]  # copy all
        else:
            chosen = random.sample(images, k)

        for img in chosen:
            safe_copy(img, dst_dir)

        summary.append((cls, k))
        print(f"Copied {k} images to {dst_dir}")

    print("\nSummary:")
    for cls, k in summary:
        print(f"  {cls}: {k} -> {DEST_BASE / (cls + '_Sample')}")

if __name__ == "__main__":
    main()
