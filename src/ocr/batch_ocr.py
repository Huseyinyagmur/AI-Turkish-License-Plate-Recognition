"""Plaka crop klasörünü PaddleOCR ile okuyup sonuçları CSV olarak kaydeder.

Örnek:
    python src/ocr/batch_ocr.py --input outputs/crops --output outputs/logs/plate_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

try:
    from .plate_cleaner import clean_plate_text, is_valid_turkish_plate, normalize_ocr_errors
except ImportError:
    # Dosya doğrudan "python src/ocr/batch_ocr.py" ile çalıştırıldığında kullanılır.
    from plate_cleaner import clean_plate_text, is_valid_turkish_plate, normalize_ocr_errors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CSV_FIELDS = [
    "file_name",
    "raw_text",
    "cleaned_text",
    "ocr_confidence",
    "is_valid",
    "image_path",
]


def parse_arguments() -> argparse.Namespace:
    """Komut satırı seçeneklerini tanımlar."""
    parser = argparse.ArgumentParser(description="Plaka crop görüntülerine toplu OCR uygular.")
    parser.add_argument("--input", default="outputs/crops", help="Crop görüntülerinin bulunduğu klasör.")
    parser.add_argument("--output", default="outputs/logs/plate_results.csv", help="Çıktı CSV dosyası.")
    parser.add_argument("--lang", default="tr", help="PaddleOCR dil kodu (varsayılan: tr).")
    parser.add_argument(
        "--min-ocr-conf",
        type=float,
        default=0.50,
        help="Geçerli sayılacak minimum OCR güven skoru (varsayılan: 0.50).",
    )
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    """Göreli bir yolu proje kök dizinine göre çözer."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_crop_images(input_dir: Path) -> list[Path]:
    """Desteklenen uzantılardaki crop görüntülerini adlarına göre sıralar."""
    return sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def extract_ocr_lines(results: Any) -> list[tuple[str, float]]:
    """PaddleOCR 2.7.x sonucundaki [box, (text, confidence)] satırlarını ayıklar."""
    if not results or not results[0]:
        return []

    lines: list[tuple[str, float]] = []
    for line in results[0]:
        try:
            _box, (text, confidence) = line
            lines.append((str(text), float(confidence)))
        except (TypeError, ValueError):
            # Beklenmeyen tek bir sonuç satırı, diğer crop'ların işlenmesini engellemez.
            continue
    return lines


def empty_row(image_path: Path) -> dict[str, object]:
    """OCR sonucu olmayan veya okunamayan görüntü için CSV satırı üretir."""
    return {
        "file_name": image_path.name,
        "raw_text": "",
        "cleaned_text": "",
        "ocr_confidence": 0,
        "is_valid": False,
        "image_path": str(image_path),
    }


def main() -> None:
    args = parse_arguments()
    if not 0.0 <= args.min_ocr_conf <= 1.0:
        raise ValueError("--min-ocr-conf değeri 0 ile 1 arasında olmalıdır.")

    input_dir = project_path(args.input)
    output_path = project_path(args.output)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Girdi klasörü bulunamadı: {input_dir}")

    try:
        from paddleocr import PaddleOCR
    except ImportError as error:
        raise ImportError("PaddleOCR kurulu değil. 'pip install paddleocr==2.7.*' komutunu çalıştırın.") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_paths = find_crop_images(input_dir)
    ocr = PaddleOCR(lang=args.lang, use_angle_cls=False)

    crops_with_ocr = 0
    valid_plate_count = 0
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for image_path in image_paths:
            try:
                # PaddleOCR 2.7.x API: results[0] = [[box, (text, confidence)], ...]
                ocr_lines = extract_ocr_lines(ocr.ocr(str(image_path), cls=False))
            except Exception as error:
                print(f"Uyarı: OCR çalıştırılamadı ({image_path.name}): {error}", file=sys.stderr)
                ocr_lines = []

            if not ocr_lines:
                writer.writerow(empty_row(image_path))
                continue

            crops_with_ocr += 1
            for raw_text, confidence in ocr_lines:
                cleaned_text = clean_plate_text(normalize_ocr_errors(raw_text))
                is_valid = confidence >= args.min_ocr_conf and is_valid_turkish_plate(cleaned_text)
                valid_plate_count += int(is_valid)
                writer.writerow(
                    {
                        "file_name": image_path.name,
                        "raw_text": raw_text,
                        "cleaned_text": cleaned_text,
                        "ocr_confidence": f"{confidence:.4f}",
                        "is_valid": is_valid,
                        "image_path": str(image_path),
                    }
                )

    print(f"Toplam crop sayısı: {len(image_paths)}")
    print(f"OCR sonucu bulunan sayı: {crops_with_ocr}")
    print(f"Geçerli Türk plaka sayısı: {valid_plate_count}")
    print(f"CSV kayıt yolu: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Hata: {error}", file=sys.stderr)
        sys.exit(1)
