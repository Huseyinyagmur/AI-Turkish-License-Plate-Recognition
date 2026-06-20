"""Türk plaka tespit, OCR ve tekilleştirme hattını tek komutta çalıştırır.

Örnekler:
    python src/main.py --source data/test.mp4
    python src/main.py --source data/test.jpg
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from detection import crop_plates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = PROJECT_ROOT / "outputs" / "crops"
OCR_CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "plate_results.csv"
UNIQUE_CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "unique_plates.csv"


def parse_arguments() -> argparse.Namespace:
    """Komut satırı seçeneklerini tanımlar."""
    parser = argparse.ArgumentParser(description="Türk plaka tespit ve OCR pipeline'ını çalıştırır.")
    parser.add_argument("--source", required=True, help="İşlenecek görüntü veya video dosyası.")
    parser.add_argument("--model", default="models/detection/best.pt", help="YOLO model dosyası.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum tespit güven skoru.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO giriş görüntü boyutu.")
    parser.add_argument("--frame-step", type=int, default=10, help="Videoda işlenecek kare aralığı.")
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    """Göreli yolları proje kök dizinine göre çözer."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_arguments(args: argparse.Namespace, source_path: Path, model_path: Path) -> None:
    """Pipeline başlamadan önce dosya ve sayısal parametreleri doğrular."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Kaynak görüntü veya video bulunamadı: {source_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model dosyası bulunamadı: {model_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf değeri 0 ile 1 arasında olmalıdır.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz pozitif bir tam sayı olmalıdır.")
    if args.frame_step <= 0:
        raise ValueError("--frame-step pozitif bir tam sayı olmalıdır.")


def extract_plate_crops(source_path: Path, model_path: Path, args: argparse.Namespace) -> int:
    """Mevcut detection modülünü kullanarak kaynak dosyadan plaka crop'ları üretir."""
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    model = crop_plates.YOLO(str(model_path))
    detection_args = argparse.Namespace(conf=args.conf, imgsz=args.imgsz, frame_step=args.frame_step)

    if source_path.suffix.lower() in crop_plates.IMAGE_EXTENSIONS:
        _processed_frames, _detection_frames, crop_count = crop_plates.process_image(
            source_path, model, detection_args, CROPS_DIR
        )
    else:
        _processed_frames, _detection_frames, crop_count = crop_plates.process_video(
            source_path, model, detection_args, CROPS_DIR
        )
    return crop_count


def run_script(script_path: Path, *arguments: str) -> None:
    """Bir pipeline betiğini mevcut Python yorumlayıcısıyla çalıştırır."""
    command = [sys.executable, str(script_path), *arguments]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def count_nonempty_ocr_results(csv_path: Path) -> int:
    """Ham OCR metni bulunan CSV satırlarının sayısını döndürür."""
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        return sum(bool((row.get("raw_text") or "").strip()) for row in csv.DictReader(csv_file))


def count_unique_plates(csv_path: Path) -> int:
    """Tekilleştirilmiş CSV'deki plaka satırlarını sayar."""
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        return sum(1 for _ in csv.DictReader(csv_file))


def print_summary(crop_count: int, ocr_count: int, unique_plate_count: int) -> None:
    """Pipeline sonucunu istenen kısa özet biçiminde yazdırır."""
    print("==================================")
    print("Turkish License Plate Pipeline")
    print("==================================")
    print(f"Toplam Crop: {crop_count}")
    print(f"OCR Sonucu: {ocr_count}")
    print(f"Benzersiz Plaka: {unique_plate_count}")
    print("CSV: outputs/logs/unique_plates.csv")


def main() -> None:
    args = parse_arguments()
    source_path = project_path(args.source)
    model_path = project_path(args.model)
    validate_arguments(args, source_path, model_path)

    # Adım 1: Kaynaktaki plaka bölgelerini crop olarak kaydet.
    crop_count = extract_plate_crops(source_path, model_path, args)

    # Adım 2: Tüm crop'lar için OCR, metin temizleme ve doğrulama çalıştır.
    run_script(
        PROJECT_ROOT / "src" / "ocr" / "batch_ocr.py",
        "--input",
        str(CROPS_DIR),
        "--output",
        str(OCR_CSV_PATH),
    )

    # Adım 3: Geçerli OCR sonuçlarını plaka metnine göre tekilleştir.
    run_script(
        PROJECT_ROOT / "src" / "logging" / "duplicate_filter.py",
        "--input",
        str(OCR_CSV_PATH),
        "--output",
        str(UNIQUE_CSV_PATH),
    )

    print_summary(crop_count, count_nonempty_ocr_results(OCR_CSV_PATH), count_unique_plates(UNIQUE_CSV_PATH))
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"Hata: Pipeline adımı başarısız oldu (çıkış kodu: {error.returncode}).", file=sys.stderr)
        sys.exit(error.returncode or 1)
    except Exception as error:
        print(f"Hata: {error}", file=sys.stderr)
        sys.exit(1)
