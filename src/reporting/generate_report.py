"""Tracking sonuçlarından profesyonel bir PDF plaka raporu üretir."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
DEFAULT_FINAL_CSV = LOGS_DIR / "final_tracked_plates.csv"
DEFAULT_TRACKING_SUMMARY_CSV = LOGS_DIR / "tracking_summary.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "license_plate_report.pdf"


def parse_arguments() -> argparse.Namespace:
    """PDF raporu için komut satırı seçeneklerini tanımlar."""
    parser = argparse.ArgumentParser(description="Tracking CSV sonuçlarından PDF raporu oluşturur.")
    parser.add_argument("--input", default=str(DEFAULT_FINAL_CSV), help="Final tracked plate CSV dosyası.")
    parser.add_argument(
        "--tracking-summary",
        default=str(DEFAULT_TRACKING_SUMMARY_CSV),
        help="Opsiyonel tracking özet CSV dosyası.",
    )
    parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH), help="Üretilecek PDF rapor dosyası.")
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    """Göreli bir yolu proje kök dizinine göre çözer."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dataframes(final_csv_path: Path, tracking_summary_path: Path) -> tuple[Any, Any | None]:
    """Final plaka sonuçlarını ve varsa tracking özetini pandas ile yükler."""
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("Pandas kurulu değil. 'pip install pandas' komutunu çalıştırın.") from error

    if not final_csv_path.is_file():
        raise FileNotFoundError(f"Final tracked plate CSV bulunamadı: {final_csv_path}")

    final_dataframe = pd.read_csv(final_csv_path)
    required_columns = {"track_id", "plate", "plate_confidence", "detection_count", "best_crop_path"}
    missing_columns = required_columns - set(final_dataframe.columns)
    if missing_columns:
        raise ValueError(f"Final CSV zorunlu kolonları içermiyor: {', '.join(sorted(missing_columns))}")

    for column_name in ("track_id", "plate_confidence", "detection_count"):
        final_dataframe[column_name] = pd.to_numeric(final_dataframe[column_name], errors="coerce").fillna(0)

    summary_dataframe = pd.read_csv(tracking_summary_path) if tracking_summary_path.is_file() else None
    if summary_dataframe is not None and "frame_count" in summary_dataframe.columns:
        summary_dataframe["frame_count"] = pd.to_numeric(summary_dataframe["frame_count"], errors="coerce").fillna(0)
    return final_dataframe, summary_dataframe


def make_table(data: list[list[str]], column_widths: list[float], header_color: Any) -> Any:
    """Rapor tabloları için ortak görsel biçimlendirme uygular."""
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(data, colWidths=column_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def format_number(value: Any) -> str:
    """Eksik sayısal değerleri raporda sıfır olarak gösterir."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "0"


def build_report(final_dataframe: Any, summary_dataframe: Any | None, output_path: Path) -> None:
    """Rapor sayfalarını, tablolarını ve isteğe bağlı crop görsellerini oluşturur."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer
    except ImportError as error:
        raise ImportError("ReportLab kurulu değil. 'pip install reportlab' komutunu çalıştırın.") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
    )
    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#1E3A5F"),
        spaceBefore=12,
        spaceAfter=7,
    )
    body_style = ParagraphStyle("ReportBody", parent=styles["BodyText"], fontSize=9, leading=13)
    primary_color = colors.HexColor("#1E3A5F")

    final_dataframe = final_dataframe.copy()
    final_dataframe["detection_count"] = final_dataframe["detection_count"].fillna(0)
    final_dataframe["plate_confidence"] = final_dataframe["plate_confidence"].fillna(0)
    total_tracks = len(summary_dataframe) if summary_dataframe is not None else final_dataframe["track_id"].nunique()
    unique_plates = final_dataframe["plate"].dropna().nunique()
    if summary_dataframe is not None and "frame_count" in summary_dataframe.columns:
        total_detections = sum(int(value) for value in summary_dataframe["frame_count"].fillna(0))
    else:
        total_detections = sum(int(float(value)) for value in final_dataframe["detection_count"])

    story: list[Any] = [
        Paragraph("AI-Based Turkish License Plate Recognition Report", title_style),
        Spacer(1, 0.15 * cm),
        Paragraph("Generated Automatically by YOLO11 + ByteTrack + PaddleOCR Pipeline", subtitle_style),
        Spacer(1, 0.55 * cm),
        Paragraph("1. Pipeline Summary", section_style),
        make_table(
            [
                ["Metric", "Value"],
                ["Total Tracks", str(total_tracks)],
                ["Unique Plates", str(unique_plates)],
                ["Total Detections", str(total_detections)],
            ],
            [8.5 * cm, 8.5 * cm],
            primary_color,
        ),
        Paragraph("2. Detected Plates", section_style),
    ]

    plates_table = [["Track ID", "Plate", "Confidence", "Detection Count"]]
    for _, row in final_dataframe.iterrows():
        plates_table.append(
            [
                format_number(row["track_id"]),
                str(row["plate"]),
                f"{float(row['plate_confidence']):.4f}",
                format_number(row["detection_count"]),
            ]
        )
    if len(plates_table) == 1:
        plates_table.append(["-", "No valid plate detected", "-", "0"])
    story.append(make_table(plates_table, [3.2 * cm, 5.2 * cm, 4.0 * cm, 4.6 * cm], primary_color))

    story.append(Paragraph("3. Top Detected Plates", section_style))
    top_dataframe = final_dataframe.sort_values("detection_count", ascending=False).head(10)
    top_table = [["Plate", "Track ID", "Detection Count", "Confidence"]]
    for _, row in top_dataframe.iterrows():
        top_table.append(
            [
                str(row["plate"]),
                format_number(row["track_id"]),
                format_number(row["detection_count"]),
                f"{float(row['plate_confidence']):.4f}",
            ]
        )
    if len(top_table) == 1:
        top_table.append(["No valid plate detected", "-", "0", "-"])
    story.append(make_table(top_table, [5.2 * cm, 3.2 * cm, 4.6 * cm, 4.0 * cm], primary_color))

    story.extend(
        [
            Paragraph("4. System Information", section_style),
            Paragraph(
                "<b>Model:</b> YOLO11s<br/>"
                "<b>OCR:</b> PaddleOCR<br/>"
                "<b>Tracker:</b> ByteTrack<br/>"
                "<b>Validation:</b> Turkish Plate Regex",
                body_style,
            ),
            Paragraph("5. Execution Timestamp", section_style),
            Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), body_style),
        ]
    )

    image_rows = final_dataframe.head(5)
    valid_images = []
    for _, row in image_rows.iterrows():
        image_path = Path(str(row.get("best_crop_path", "")))
        if image_path.is_file():
            valid_images.append((str(row["plate"]), image_path))
    if valid_images:
        story.append(Paragraph("Best Plate Crop Samples", section_style))
        for plate, image_path in valid_images:
            story.append(Paragraph(f"<b>{plate}</b>", body_style))
            try:
                image = Image(str(image_path))
                image._restrictSize(7.0 * cm, 3.5 * cm)
                story.extend([image, Spacer(1, 0.2 * cm)])
            except Exception:
                # Bozuk bir görüntü rapor üretimini durdurmamalıdır.
                story.append(Paragraph("Crop image could not be rendered.", body_style))

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="AI-Based Turkish License Plate Recognition Report",
        author="Turkish License Plate Recognition Pipeline",
    )
    document.build(story)


def main() -> None:
    """Komut satırı girdilerinden PDF raporu üretir."""
    args = parse_arguments()
    final_csv_path = project_path(args.input)
    tracking_summary_path = project_path(args.tracking_summary)
    output_path = project_path(args.output)
    final_dataframe, summary_dataframe = load_dataframes(final_csv_path, tracking_summary_path)
    build_report(final_dataframe, summary_dataframe, output_path)

    print("==================================")
    print("Report Generated Successfully")
    print("=============================")
    print(f"PDF: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Hata: Rapor oluşturulamadı: {error}", file=sys.stderr)
        sys.exit(1)
