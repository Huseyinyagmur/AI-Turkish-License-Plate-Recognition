"""Detect Turkish license plates and save each detected plate as a crop image.

Example:
    python src/detection/crop_plates.py --source path/to/video.mp4
    python src/detection/crop_plates.py --source data/test.mp4 --conf 0.25 --imgsz 640 --frame-step 10
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_CROP_WIDTH = 10
MIN_CROP_HEIGHT = 5


def project_path(path_value: str) -> Path:
    """Interpret relative paths from the project root, not the current directory."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO11s ile plaka tespiti yapar ve plaka crop görüntülerini kaydeder."
    )
    parser.add_argument("--source", required=True, help="Girdi görüntüsü veya video dosyasının yolu.")
    parser.add_argument(
        "--model",
        default="models/detection/best.pt",
        help="YOLO model yolu (varsayılan: models/detection/best.pt).",
    )
    parser.add_argument(
        "--output",
        default="outputs/crops",
        help="Crop görüntülerinin kaydedileceği klasör (varsayılan: outputs/crops).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Minimum tespit güven skoru (varsayılan: 0.25)."
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="YOLO giriş görüntü boyutu (varsayılan: 640)."
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Videoda tespit yapılacak kare aralığı (varsayılan: 10).",
    )
    return parser.parse_args()


def save_plate_crops(frame, frame_number: int, result, output_dir: Path) -> int:
    """Save valid, image-bounded crops in a YOLO result and return their count."""
    if result.boxes is None or len(result.boxes) == 0:
        return 0

    image_height, image_width = frame.shape[:2]
    saved_count = 0

    # The index identifies the detection's order in its source frame.
    for plate_index, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        confidence = float(box.conf[0].cpu().item())

        # Round outward first, then clamp every coordinate to the frame limits.
        left = max(0, min(image_width, math.floor(x1)))
        top = max(0, min(image_height, math.floor(y1)))
        right = max(0, min(image_width, math.ceil(x2)))
        bottom = max(0, min(image_height, math.ceil(y2)))

        crop_width = right - left
        crop_height = bottom - top
        if crop_width < MIN_CROP_WIDTH or crop_height < MIN_CROP_HEIGHT:
            continue

        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            continue

        filename = f"frame_{frame_number:06d}_plate_{plate_index:02d}_conf_{confidence:.2f}.jpg"
        crop_path = output_dir / filename
        if cv2.imwrite(str(crop_path), crop):
            saved_count += 1
        else:
            print(f"Uyarı: Crop kaydedilemedi: {crop_path}", file=sys.stderr)

    return saved_count


def process_frame(frame, frame_number: int, model: YOLO, args: argparse.Namespace, output_dir: Path) -> int:
    """Run model inference for a single frame and save all valid plate crops."""
    results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
    return save_plate_crops(frame, frame_number, results[0], output_dir)


def process_image(
    source_path: Path, model: YOLO, args: argparse.Namespace, output_dir: Path
) -> tuple[int, int, int]:
    """Process one still image; images always use frame number zero."""
    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Görüntü okunamadı: {source_path}")
    crop_count = process_frame(image, frame_number=0, model=model, args=args, output_dir=output_dir)
    # Still images are always processed once; --frame-step does not apply.
    return 1, 1, crop_count


def process_video(
    source_path: Path, model: YOLO, args: argparse.Namespace, output_dir: Path
) -> tuple[int, int, int]:
    """Read every video frame and run detection only at the requested interval."""
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError(f"Video açılamadı: {source_path}")

    total_crops = 0
    frame_number = 0
    processed_frame_count = 0
    detection_frame_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            processed_frame_count += 1
            if frame_number % args.frame_step == 0:
                detection_frame_count += 1
                total_crops += process_frame(frame, frame_number, model, args, output_dir)
            frame_number += 1
    finally:
        capture.release()

    return processed_frame_count, detection_frame_count, total_crops


def main() -> None:
    args = parse_arguments()
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf değeri 0 ile 1 arasında olmalıdır.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz pozitif bir tam sayı olmalıdır.")

    if args.frame_step <= 0:
        raise ValueError("--frame-step pozitif bir tam sayı olmalıdır.")

    source_path = project_path(args.source)
    model_path = project_path(args.model)
    output_dir = project_path(args.output)

    if not source_path.is_file():
        raise FileNotFoundError(f"Kaynak dosya bulunamadı: {source_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        processed_frames, detection_frames, total_crops = process_image(source_path, model, args, output_dir)
    else:
        processed_frames, detection_frames, total_crops = process_video(source_path, model, args, output_dir)

    print(f"Toplam işlenen frame sayısı: {processed_frames}")
    print(f"Detection yapılan frame sayısı: {detection_frames}")

    print(f"Toplam kaydedilen plaka crop sayısı: {total_crops}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Hata: {error}", file=sys.stderr)
        sys.exit(1)
