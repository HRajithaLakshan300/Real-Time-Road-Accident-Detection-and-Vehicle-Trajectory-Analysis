from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_classifier import find_split


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    for split_name in ("train", "val", "test"):
        split_path = find_split(args.data_dir, split_name)
        counts: Counter[str] = Counter()
        invalid: list[Path] = []
        for class_directory in [item for item in split_path.iterdir() if item.is_dir()]:
            for image_path in class_directory.rglob("*"):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception:
                    invalid.append(image_path)
                else:
                    counts[class_directory.name] += 1

        print(f"\n{split_name.upper()}: {split_path}")
        for class_name, count in sorted(counts.items()):
            print(f"  {class_name}: {count} valid images")
        if invalid:
            print(f"  Invalid images: {len(invalid)}")
            for path in invalid[:10]:
                print(f"    {path}")


if __name__ == "__main__":
    main()
