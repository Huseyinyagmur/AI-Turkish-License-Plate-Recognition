"""PaddleOCR çıktılarındaki Türk plaka metinlerini temizleme yardımcıları."""

from __future__ import annotations

import re


# Türkçe harfler, plakalarda kullanılan Latin karakter karşılıklarına dönüştürülür.
TURKISH_CHARACTER_MAP = str.maketrans(
    {
        "Ç": "C",
        "Ğ": "G",
        "İ": "I",
        "Ö": "O",
        "Ş": "S",
        "Ü": "U",
    }
)

# İl kodu 01-81, ardından 1-3 harf ve 2-4 rakam gelmelidir.
TURKISH_PLATE_PATTERN = re.compile(r"^(?:0[1-9]|[1-7][0-9]|8[0-1])[A-Z]{1,3}\d{2,4}$")


def normalize_ocr_errors(text: str) -> str:
    """Türkçe karakterleri Latin karşılıklarına çevirip boşlukları kaldırır.

    O/0 ve I/1 gibi belirsiz karakter çiftleri bilerek dönüştürülmez. Bu tür
    düzeltmeler plakanın harf ve rakam bölümleri değerlendirilerek yapılmalıdır.
    """
    return re.sub(r"\s+", "", text.upper().translate(TURKISH_CHARACTER_MAP))


def clean_plate_text(text: str) -> str:
    """Metinden yalnızca büyük A-Z harfleri ile 0-9 rakamlarını bırakır."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_turkish_plate(text: str) -> bool:
    """Metnin 01-81 il kodlu standart Türk plaka biçiminde olup olmadığını döndürür."""
    return TURKISH_PLATE_PATTERN.fullmatch(text) is not None


if __name__ == "__main__":
    samples = ["03ACU 808", "34 AB-1234", "06 A8 1234", "35A1234", "99XYZ999"]

    for raw_text in samples:
        cleaned_text = clean_plate_text(normalize_ocr_errors(raw_text))
        print(f"Raw: {raw_text}")
        print(f"Cleaned: {cleaned_text}")
        print(f"Valid: {is_valid_turkish_plate(cleaned_text)}")
        print()
