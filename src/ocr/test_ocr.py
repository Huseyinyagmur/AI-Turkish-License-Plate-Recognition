"""Bir plaka crop görüntüsündeki metni PaddleOCR ile okuyun.

Örnek:
    python src/ocr/test_ocr.py --image outputs/crops/sample.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini alır."""
    parser = argparse.ArgumentParser(description="Tek bir plaka crop görüntüsünde OCR çalıştırır.")
    parser.add_argument("--image", required=True, help="Okunacak plaka crop görüntüsünün yolu.")
    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    """Göreli yolları, komutun çalıştırıldığı yerden bağımsız olarak proje köküne göre çözer."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_arguments()
    image_path = resolve_project_path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")

    try:
        from paddleocr import PaddleOCR
    except ImportError as error:
        raise ImportError("PaddleOCR kurulu değil. 'pip install paddleocr' komutunu çalıştırın.") from error

    # "tr" modeli Türkçe karakterleri de içeren Latin alfabesi için yapılandırılır.
    ocr = PaddleOCR(
        lang="tr",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    results = ocr.predict(str(image_path))

    found_text = False
    for result in results:
        # PaddleOCR 3.x, her metin satırı için paralel metin ve skor listeleri döndürür.
        for text, confidence in zip(result["rec_texts"], result["rec_scores"]):
            found_text = True
            print(f"text: {text}")
            print(f"confidence: {float(confidence):.4f}")

    if not found_text:
        print("Metin bulunamadı.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, OSError, RuntimeError) as error:
        print(f"Hata: {error}", file=sys.stderr)
        sys.exit(1)
