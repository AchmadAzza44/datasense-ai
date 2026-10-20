"""Konfigurasi dan konstanta DataSense AI."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
REPORTS_DIR = DATA_DIR / "reports"

ALLOWED_DATA_EXTENSIONS = (".csv", ".xlsx")
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

DATASENSE_NAME = "DataSense AI"

DISCLAIMER_ANALYTICS = (
    "Hasil analisis bersifat interpretasi berbasis data dan konfigurasi yang dipilih. "
    "Hasil tersebut bukan bukti hubungan sebab-akibat dan perlu diperiksa oleh manusia "
    "sebelum digunakan untuk mengambil keputusan."
)


def ensure_runtime_dirs() -> None:
    """Create runtime directories used by the applications."""
    for directory in (DATA_DIR, SAMPLES_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
