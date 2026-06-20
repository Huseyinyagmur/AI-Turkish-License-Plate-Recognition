"""Split the Turkish plate dataset into YOLO train/validation/test folders.

Run from any directory with:
    python src/utils/split_dataset.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT_RATIOS = (0.70, 0.20)  # The remaining samples are assigned to test.
RANDOM_SEED = 42


def is_valid_yolo_label(label_path: Path) -> bool:
    """Return whether every non-empty row follows YOLO bounding-box format."""
    try:
        with label_path.open("r", encoding="utf-8") as label_file:
            for line_number, line in enumerate(label_file, start=1):
                line = line.strip()
                if not line:
                    continue

                values = line.split()
                if len(values) != 5:
                    return False

                try:
                    class_id = int(values[0])
                    coordinates = [float(value) for value in values[1:]]
                except ValueError:
                    return False

                if class_id < 0 or any(not 0.0 <= value <= 1.0 for value in coordinates):
                    return False
    except (OSError, UnicodeDecodeError):
        return False

    return True


def prepare_output_directory(output_root: Path) -> None:
    """Remove a previous generated split and recreate its directory structure."""
    if output_root.exists():
        shutil.rmtree(output_root)

    for data_type in ("images", "labels"):
        for split_name in ("train", "val", "test"):
            (output_root / data_type / split_name).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_root: Path) -> None:
    """Create the YOLO dataset configuration file."""
    data_yaml = """path: datasets/processed/turkish_plate_yolo
train: images/train
val: images/val
test: images/test
names:
  0: license_plate
"""
    (output_root / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source_root = project_root / "datasets" / "raw" / "turkish_plate_dataset"
    source_images = source_root / "images"
    source_labels = source_root / "labels"
    output_root = project_root / "datasets" / "processed" / "turkish_plate_yolo"

    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(
            "Dataset klasörleri bulunamadı: "
            f"{source_images} ve {source_labels} mevcut olmalı."
        )

    valid_pairs: list[tuple[Path, Path]] = []
    missing_label_count = 0
    malformed_label_count = 0

    image_files = sorted(
        (path for path in source_images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )

    for image_path in image_files:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_label_count += 1
            continue

        if not is_valid_yolo_label(label_path):
            malformed_label_count += 1
            continue

        valid_pairs.append((image_path, label_path))

    random.Random(RANDOM_SEED).shuffle(valid_pairs)

    total_pairs = len(valid_pairs)
    train_end = int(total_pairs * SPLIT_RATIOS[0])
    val_end = train_end + int(total_pairs * SPLIT_RATIOS[1])
    splits = {
        "train": valid_pairs[:train_end],
        "val": valid_pairs[train_end:val_end],
        "test": valid_pairs[val_end:],
    }

    prepare_output_directory(output_root)
    for split_name, pairs in splits.items():
        for image_path, label_path in pairs:
            shutil.copy2(image_path, output_root / "images" / split_name / image_path.name)
            shutil.copy2(label_path, output_root / "labels" / split_name / label_path.name)

    write_data_yaml(output_root)

    print(f"Toplam geçerli image-label çifti: {total_pairs}")
    print(f"Train sayısı: {len(splits['train'])}")
    print(f"Val sayısı: {len(splits['val'])}")
    print(f"Test sayısı: {len(splits['test'])}")
    print(f"Eksik label sayısı: {missing_label_count}")
    print(f"Bozuk label sayısı: {malformed_label_count}")


if __name__ == "__main__":
    main()
