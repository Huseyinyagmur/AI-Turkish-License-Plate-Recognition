"""YOLO11s ve ByteTrack ile video plakalarını takip edip crop olarak kaydeder."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKING_CROPS_DIR = PROJECT_ROOT / "outputs" / "tracking_crops"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
TRACKING_RESULTS_PATH = LOGS_DIR / "tracking_results.csv"
TRACKING_SUMMARY_PATH = LOGS_DIR / "tracking_summary.csv"
MIN_CROP_WIDTH = 10
MIN_CROP_HEIGHT = 5
LOGGER = logging.getLogger(__name__)


@dataclass
class TrackSummary:
    """Tek bir ByteTrack kimliğine ait özet bilgileri tutar."""

    first_frame: int
    last_frame: int
    frame_count: int
    best_confidence: float
    best_crop_path: str


def parse_arguments() -> argparse.Namespace:
    """Tracking komut satırı seçeneklerini tanımlar."""
    parser = argparse.ArgumentParser(description="YOLO11s ve ByteTrack ile plaka takibi yapar.")
    parser.add_argument("--source", required=True, help="İşlenecek video veya görüntü dosyası.")
    parser.add_argument("--model", default="models/detection/best.pt", help="YOLO model dosyası.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum tespit güven skoru.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO giriş görüntü boyutu.")
    parser.add_argument("--frame-step", type=int, default=1, help="Videoda işlenecek kare aralığı.")
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="İşaretlenmiş takip videosunu kaydeder (varsayılan: True).",
    )
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    """Göreli yolları proje kök dizinine göre çözer."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_arguments(args: argparse.Namespace, source_path: Path, model_path: Path) -> None:
    """Dosya yollarını ve sayısal tracking parametrelerini doğrular."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Kaynak dosya bulunamadı: {source_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf değeri 0 ile 1 arasında olmalıdır.")
    if args.imgsz <= 0 or args.frame_step <= 0:
        raise ValueError("--imgsz ve --frame-step pozitif tam sayı olmalıdır.")


def clamp_bbox(box: list[float], image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    """Bir bounding box'ı görüntü sınırlarına taşır ve geçerliyse döndürür."""
    x1, y1, x2, y2 = box
    left = max(0, min(image_width, math.floor(x1)))
    top = max(0, min(image_height, math.floor(y1)))
    right = max(0, min(image_width, math.ceil(x2)))
    bottom = max(0, min(image_height, math.ceil(y2)))
    if right - left < MIN_CROP_WIDTH or bottom - top < MIN_CROP_HEIGHT:
        return None
    return left, top, right, bottom


def write_summary_csv(track_summaries: dict[int, TrackSummary]) -> None:
    """Track özetlerini CSV dosyasına yazar."""
    with TRACKING_SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["track_id", "first_frame", "last_frame", "frame_count", "best_confidence", "best_crop_path"],
        )
        writer.writeheader()
        for track_id, summary in sorted(track_summaries.items()):
            writer.writerow(
                {
                    "track_id": track_id,
                    "first_frame": summary.first_frame,
                    "last_frame": summary.last_frame,
                    "frame_count": summary.frame_count,
                    "best_confidence": f"{summary.best_confidence:.4f}",
                    "best_crop_path": summary.best_crop_path,
                }
            )


def main() -> None:
    """ByteTrack sonuçlarını crop, ayrıntı CSV'si ve özet CSV'si olarak üretir."""
    args = parse_arguments()
    source_path = project_path(args.source)
    model_path = project_path(args.model)
    validate_arguments(args, source_path, model_path)

    TRACKING_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    track_summaries: dict[int, TrackSummary] = {}

    with TRACKING_RESULTS_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["frame_id", "track_id", "confidence", "x1", "y1", "x2", "y2", "crop_path"],
        )
        writer.writeheader()

        # vid_stride ile frame skipping uygulanır; persist=True track kimliğini korur.
        results = model.track(
            source=str(source_path),
            tracker="bytetrack.yaml",
            persist=True,
            conf=args.conf,
            imgsz=args.imgsz,
            vid_stride=args.frame_step,
            save=args.save_video,
            stream=True,
            verbose=False,
        )
        for processed_index, result in enumerate(results):
            if result.boxes is None or result.boxes.id is None:
                continue

            frame = result.orig_img
            frame_height, frame_width = frame.shape[:2]
            frame_id = processed_index * args.frame_step
            for box in result.boxes:
                if box.id is None:
                    continue
                bbox = clamp_bbox(box.xyxy[0].cpu().tolist(), frame_width, frame_height)
                if bbox is None:
                    continue

                track_id = int(box.id[0].item())
                confidence = float(box.conf[0].item())
                left, top, right, bottom = bbox
                crop = frame[top:bottom, left:right]
                if crop.size == 0:
                    continue

                crop_path = TRACKING_CROPS_DIR / (
                    f"track_{track_id:03d}_frame_{frame_id:06d}_conf_{confidence:.2f}.jpg"
                )
                if not cv2.imwrite(str(crop_path), crop):
                    LOGGER.warning("Crop kaydedilemedi: %s", crop_path)
                    continue

                writer.writerow(
                    {
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "confidence": f"{confidence:.4f}",
                        "x1": left,
                        "y1": top,
                        "x2": right,
                        "y2": bottom,
                        "crop_path": str(crop_path),
                    }
                )

                summary = track_summaries.get(track_id)
                if summary is None:
                    track_summaries[track_id] = TrackSummary(frame_id, frame_id, 1, confidence, str(crop_path))
                else:
                    summary.last_frame = frame_id
                    summary.frame_count += 1
                    if confidence > summary.best_confidence:
                        summary.best_confidence = confidence
                        summary.best_crop_path = str(crop_path)

    write_summary_csv(track_summaries)
    print(f"Takip edilen track sayısı: {len(track_summaries)}")
    print(f"Tracking CSV: {TRACKING_RESULTS_PATH}")
    print(f"Tracking özet CSV: {TRACKING_SUMMARY_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except Exception as error:
        LOGGER.error("Tracking hatası: %s", error)
        sys.exit(1)
