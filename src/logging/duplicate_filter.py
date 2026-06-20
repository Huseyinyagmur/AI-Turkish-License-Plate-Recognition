"""OCR sonuçlarındaki tekrar eden Türk plakalarını tekilleştirir.

Örnek:
    python src/logging/duplicate_filter.py --input outputs/logs/plate_results.csv --output outputs/logs/unique_plates.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {
    "file_name",
    "raw_text",
    "cleaned_text",
    "ocr_confidence",
    "is_valid",
    "image_path",
}
OUTPUT_COLUMNS = [
    "plate",
    "best_confidence",
    "detection_count",
    "first_seen_file",
    "last_seen_file",
    "best_image_path",
    "raw_variants",
]


def parse_arguments() -> argparse.Namespace:
    """Komut satırı seçeneklerini tanımlar."""
    parser = argparse.ArgumentParser(description="OCR sonuçlarındaki tekrar eden plakaları tekilleştirir.")
    parser.add_argument("--input", default="outputs/logs/plate_results.csv", help="Girdi OCR sonuç CSV dosyası.")
    parser.add_argument("--output", default="outputs/logs/unique_plates.csv", help="Tekilleştirilmiş çıktı CSV dosyası.")
    parser.add_argument(
        "--valid-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Yalnızca is_valid=True satırlarını kullanır (varsayılan: True).",
    )
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    """Göreli yolları proje kök dizinine göre çözer."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_true(value: Any) -> bool:
    """CSV'den gelen bool veya metin değerini güvenle bool'a dönüştürür."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def text_value(value: Any) -> str:
    """Eksik CSV hücrelerini boş metin olarak döndürür."""
    return "" if value is None or str(value).lower() == "nan" else str(value)


def unique_raw_variants(values: Any) -> str:
    """Boş olmayan ham OCR metinlerini ilk görülme sırasıyla birleştirir."""
    variants: list[str] = []
    for value in values:
        raw_text = text_value(value).strip()
        if raw_text and raw_text not in variants:
            variants.append(raw_text)
    return " | ".join(variants)


def main() -> None:
    args = parse_arguments()
    input_path = project_path(args.input)
    output_path = project_path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Girdi CSV dosyası bulunamadı: {input_path}")

    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("Pandas kurulu değil. 'pip install pandas' komutunu çalıştırın.") from error

    dataframe = pd.read_csv(input_path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Girdi CSV dosyasında zorunlu kolonlar eksik: {missing}")

    total_ocr_rows = len(dataframe)
    dataframe["cleaned_text"] = dataframe["cleaned_text"].fillna("").astype(str).str.strip()
    dataframe = dataframe[dataframe["cleaned_text"] != ""].copy()

    if args.valid_only:
        dataframe = dataframe[dataframe["is_valid"].map(is_true)].copy()

    # Bozuk ya da boş güven değerleri, gruplama sırasında en düşük değer kabul edilir.
    dataframe["ocr_confidence"] = pd.to_numeric(dataframe["ocr_confidence"], errors="coerce").fillna(0.0)

    unique_rows: list[dict[str, object]] = []
    for plate, group in dataframe.groupby("cleaned_text", sort=False):
        best_row = group.loc[group["ocr_confidence"].idxmax()]
        unique_rows.append(
            {
                "plate": plate,
                "best_confidence": round(float(best_row["ocr_confidence"]), 4),
                "detection_count": len(group),
                "first_seen_file": text_value(group["file_name"].iloc[0]),
                "last_seen_file": text_value(group["file_name"].iloc[-1]),
                "best_image_path": text_value(best_row["image_path"]),
                "raw_variants": unique_raw_variants(group["raw_text"]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unique_rows, columns=OUTPUT_COLUMNS).to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Toplam OCR satırı: {total_ocr_rows}")
    print(f"Kullanılan geçerli satır sayısı: {len(dataframe)}")
    print(f"Benzersiz plaka sayısı: {len(unique_rows)}")
    print(f"Çıktı CSV yolu: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Hata: {error}", file=sys.stderr)
        sys.exit(1)
