"""ByteTrack crop'larında OCR çalıştırır ve her track için plaka oylaması yapar."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ocr.plate_cleaner import clean_plate_text, is_valid_turkish_plate, normalize_ocr_errors


TRACKING_RESULTS_PATH = PROJECT_ROOT / "outputs" / "logs" / "tracking_results.csv"
FINAL_RESULTS_PATH = PROJECT_ROOT / "outputs" / "logs" / "final_tracked_plates.csv"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrCandidate:
    """Tek crop'tan gelen geçerli OCR adayını temsil eder."""

    plate: str
    confidence: float
    crop_path: str


def parse_arguments() -> argparse.Namespace:
    """Track OCR komut satırı seçeneklerini tanımlar."""
    parser = argparse.ArgumentParser(description="Track crop'larında OCR ve plaka oylaması çalıştırır.")
    parser.add_argument("--input", default=str(TRACKING_RESULTS_PATH), help="Tracking sonuç CSV dosyası.")
    parser.add_argument("--output", default=str(FINAL_RESULTS_PATH), help="Final track-plaka CSV dosyası.")
    parser.add_argument("--lang", default="tr", help="PaddleOCR dil kodu.")
    parser.add_argument("--min-ocr-conf", type=float, default=0.50, help="Minimum OCR güven skoru.")
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    """Göreli yolları proje kök dizinine göre çözer."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_ocr_lines(results: Any) -> list[tuple[str, float]]:
    """PaddleOCR 2.7.x çıktısındaki metin ve skor çiftlerini çıkarır."""
    if not results or not results[0]:
        return []
    lines: list[tuple[str, float]] = []
    for line in results[0]:
        try:
            _box, (text, confidence) = line
            lines.append((str(text), float(confidence)))
        except (TypeError, ValueError):
            LOGGER.warning("Beklenmeyen OCR sonuç satırı atlandı.")
    return lines


def choose_winner(candidates: list[OcrCandidate]) -> tuple[str, float, str] | None:
    """En sık görülen geçerli plakayı; eşitlikte ortalama güveni yüksek olanı seçer."""
    if not candidates:
        return None

    vote_counts = Counter(candidate.plate for candidate in candidates)
    mean_confidences = {
        plate: sum(candidate.confidence for candidate in candidates if candidate.plate == plate) / count
        for plate, count in vote_counts.items()
    }
    winning_plate = max(vote_counts, key=lambda plate: (vote_counts[plate], mean_confidences[plate], plate))
    winning_candidates = [candidate for candidate in candidates if candidate.plate == winning_plate]
    best_candidate = max(winning_candidates, key=lambda candidate: candidate.confidence)
    return winning_plate, best_candidate.confidence, best_candidate.crop_path


def read_tracking_rows(input_path: Path) -> dict[int, list[dict[str, str]]]:
    """Tracking CSV satırlarını track kimliğine göre, dosya sırasını koruyarak gruplar."""
    required_columns = {"frame_id", "track_id", "crop_path"}
    with input_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Tracking CSV gerekli frame_id, track_id ve crop_path kolonlarını içermiyor.")
        grouped_rows: dict[int, list[dict[str, str]]] = {}
        for row in reader:
            try:
                grouped_rows.setdefault(int(row["track_id"]), []).append(row)
            except (TypeError, ValueError):
                LOGGER.warning("Geçersiz track_id içeren CSV satırı atlandı.")
    return grouped_rows


def main() -> None:
    """Track crop'larını OCR ile okuyup track-bazlı oylama sonucu üretir."""
    args = parse_arguments()
    if not 0.0 <= args.min_ocr_conf <= 1.0:
        raise ValueError("--min-ocr-conf değeri 0 ile 1 arasında olmalıdır.")

    input_path = project_path(args.input)
    output_path = project_path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Tracking CSV bulunamadı: {input_path}")

    try:
        from paddleocr import PaddleOCR
    except ImportError as error:
        raise ImportError("PaddleOCR kurulu değil. 'pip install paddleocr==2.7.*' komutunu çalıştırın.") from error

    track_rows = read_tracking_rows(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ocr = PaddleOCR(lang=args.lang, use_angle_cls=False)

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "track_id",
                "plate",
                "plate_confidence",
                "detection_count",
                "first_frame",
                "last_frame",
                "best_crop_path",
            ],
        )
        writer.writeheader()

        for track_id, rows in sorted(track_rows.items()):
            candidates: list[OcrCandidate] = []
            for row in rows:
                crop_path = Path(row["crop_path"])
                if not crop_path.is_file():
                    LOGGER.warning("Crop bulunamadı, atlandı: %s", crop_path)
                    continue
                try:
                    ocr_lines = extract_ocr_lines(ocr.ocr(str(crop_path), cls=False))
                except Exception as error:
                    LOGGER.warning("OCR çalıştırılamadı (%s): %s", crop_path.name, error)
                    continue

                for raw_text, confidence in ocr_lines:
                    cleaned_text = clean_plate_text(normalize_ocr_errors(raw_text))
                    if confidence >= args.min_ocr_conf and is_valid_turkish_plate(cleaned_text):
                        candidates.append(OcrCandidate(cleaned_text, confidence, str(crop_path)))

            winner = choose_winner(candidates)
            if winner is None:
                continue
            plate, plate_confidence, best_crop_path = winner
            writer.writerow(
                {
                    "track_id": track_id,
                    "plate": plate,
                    "plate_confidence": f"{plate_confidence:.4f}",
                    "detection_count": len(rows),
                    "first_frame": rows[0]["frame_id"],
                    "last_frame": rows[-1]["frame_id"],
                    "best_crop_path": best_crop_path,
                }
            )

    print(f"İşlenen track sayısı: {len(track_rows)}")
    print(f"Final tracked plate CSV: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except Exception as error:
        LOGGER.error("Track OCR hatası: %s", error)
        sys.exit(1)
