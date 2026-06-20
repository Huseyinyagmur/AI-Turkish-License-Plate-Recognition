"""Türk plaka tanıma pipeline çıktıları için Streamlit dashboard'u.

README komutu:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
FINAL_CSV_PATH = LOGS_DIR / "final_tracked_plates.csv"
TRACKING_SUMMARY_PATH = LOGS_DIR / "tracking_summary.csv"
UNIQUE_PLATES_PATH = LOGS_DIR / "unique_plates.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "license_plate_report.pdf"


@st.cache_data(show_spinner=False)
def load_csv(csv_path: str) -> pd.DataFrame:
    """CSV dosyasını önbelleğe alarak yükler."""
    return pd.read_csv(csv_path)


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    """Mevcutsa yardımcı CSV dosyasını okur, bozuksa kullanıcıyı bilgilendirir."""
    if not path.is_file():
        return None
    try:
        return load_csv(str(path))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.warning(f"{path.name} okunamadı: {error}")
        return None


def numeric_series(dataframe: pd.DataFrame, column_name: str) -> pd.Series:
    """Bir kolonu eksik ve bozuk değerleri sıfır kabul ederek sayısal seriye çevirir."""
    if column_name not in dataframe.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(dataframe[column_name], errors="coerce").fillna(0.0)


def apply_dashboard_style() -> None:
    """Koyu tema ile uyumlu, kurumsal görünümlü hafif bir dashboard stili uygular."""
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: #0B1220; }
        [data-testid="stHeader"] { background: rgba(11, 18, 32, 0.88); }
        [data-testid="stSidebar"] { background: #111C2E; }
        [data-testid="stMetric"] {
            background: #17243A;
            border: 1px solid #2A3A55;
            border-radius: 12px;
            padding: 18px;
        }
        [data-testid="stMetricValue"] { font-size: 2rem; }
        .dashboard-subtitle { color: #94A3B8; font-size: 1.05rem; margin-bottom: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_system_information() -> None:
    """Sidebar'da kullanılan model ve doğrulama bileşenlerini gösterir."""
    st.sidebar.header("System Information")
    st.sidebar.markdown(
        """
        **Model:** YOLO11s  
        **Tracker:** ByteTrack  
        **OCR:** PaddleOCR  
        **Validation:** Turkish Regex
        """
    )
    st.sidebar.divider()
    st.sidebar.caption("Pipeline çıktıları yerel `outputs/` dizininden okunur.")


def display_metrics(final_dataframe: pd.DataFrame, summary_dataframe: pd.DataFrame | None) -> None:
    """Üst bölümdeki pipeline özet metriklerini gösterir."""
    total_tracks = len(summary_dataframe) if summary_dataframe is not None else final_dataframe["track_id"].nunique()
    unique_plates = final_dataframe["plate"].dropna().nunique()
    if summary_dataframe is not None and "frame_count" in summary_dataframe.columns:
        total_detections = int(numeric_series(summary_dataframe, "frame_count").sum())
    else:
        total_detections = int(numeric_series(final_dataframe, "detection_count").sum())
    average_confidence = numeric_series(final_dataframe, "plate_confidence").mean()

    total_tracks_card, unique_plates_card, total_detections_card, confidence_card = st.columns(4)
    total_tracks_card.metric("Total Tracks", total_tracks)
    unique_plates_card.metric("Unique Plates", unique_plates)
    total_detections_card.metric("Total Detections", total_detections)
    confidence_card.metric("Average OCR Confidence", f"{average_confidence:.4f}" if pd.notna(average_confidence) else "N/A")


def resolve_crop_path(path_value: Any) -> Path | None:
    """CSV'deki crop yolunu güvenli, mevcut bir dosya yoluna dönüştürür."""
    if path_value is None or pd.isna(path_value):
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.is_file() else None


def display_crop_gallery(final_dataframe: pd.DataFrame) -> None:
    """Mevcut en iyi crop görsellerini, en fazla beş plaka için gösterir."""
    if "best_crop_path" not in final_dataframe.columns:
        return

    gallery_rows = final_dataframe.sort_values("plate_confidence", ascending=False).head(5)
    images = []
    captions = []
    for _, row in gallery_rows.iterrows():
        crop_path = resolve_crop_path(row["best_crop_path"])
        if crop_path is not None:
            images.append(str(crop_path))
            captions.append(f"{row['plate']} | confidence: {float(row['plate_confidence']):.4f}")

    if images:
        st.subheader("Best Plate Crop Samples")
        st.image(images, caption=captions, width=220)
    else:
        st.info("Gösterilebilecek crop görüntüsü bulunamadı.")


def display_downloads() -> None:
    """Final CSV ve varsa PDF rapor için indirme düğmeleri sunar."""
    st.subheader("Downloads")
    st.download_button(
        "Final CSV'yi indir",
        data=FINAL_CSV_PATH.read_bytes(),
        file_name="final_tracked_plates.csv",
        mime="text/csv",
    )
    if REPORT_PATH.is_file():
        st.download_button(
            "PDF raporunu indir",
            data=REPORT_PATH.read_bytes(),
            file_name="license_plate_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("PDF raporu henüz üretilmedi. Pipeline'ı --tracking --report ile çalıştırın.")


def main() -> None:
    """Dashboard sayfasını oluşturur ve pipeline çıktılarını görselleştirir."""
    st.set_page_config(page_title="Turkish License Plate Recognition", page_icon="🚗", layout="wide")
    st.title("AI-Based Turkish License Plate Recognition Dashboard")
    st.caption("YOLO11s + ByteTrack + PaddleOCR pipeline sonuçları")

    if not FINAL_CSV_PATH.is_file():
        st.warning("Önce pipeline çalıştırılmalıdır.")
        st.stop()

    try:
        final_dataframe = load_csv(str(FINAL_CSV_PATH))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Final plaka CSV dosyası okunamadı: {error}")
        st.stop()

    required_columns = {"track_id", "plate", "plate_confidence", "detection_count"}
    missing_columns = required_columns - set(final_dataframe.columns)
    if missing_columns:
        st.error(f"Final plaka CSV dosyasında eksik kolonlar var: {', '.join(sorted(missing_columns))}")
        st.stop()

    # Sıralama ve kart hesaplamalarının CSV türlerinden etkilenmemesi için normalleştir.
    final_dataframe["plate_confidence"] = numeric_series(final_dataframe, "plate_confidence")
    final_dataframe["detection_count"] = numeric_series(final_dataframe, "detection_count")

    summary_dataframe = read_optional_csv(TRACKING_SUMMARY_PATH)
    unique_dataframe = read_optional_csv(UNIQUE_PLATES_PATH)
    display_metrics(final_dataframe, summary_dataframe)

    detected_tab, rankings_tab, gallery_tab = st.tabs(["Detected Plates", "Rankings", "Crops & Downloads"])
    with detected_tab:
        st.subheader("Detected Plates")
        st.dataframe(final_dataframe, use_container_width=True, hide_index=True)
        if unique_dataframe is not None:
            st.caption(f"Ek tekilleştirme CSV'sinde {len(unique_dataframe)} kayıt bulunuyor.")

    with rankings_tab:
        st.subheader("Highest Confidence Plates")
        st.dataframe(
            final_dataframe.sort_values("plate_confidence", ascending=False).head(10),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Most Frequently Detected Plates")
        st.dataframe(
            final_dataframe.sort_values("detection_count", ascending=False).head(10),
            use_container_width=True,
            hide_index=True,
        )

    with gallery_tab:
        display_crop_gallery(final_dataframe)
        display_downloads()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        st.error(f"Dashboard yüklenirken hata oluştu: {error}")
