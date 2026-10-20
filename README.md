# DataSense AI

Aplikasi analisis data dan pendukung keputusan (Streamlit). Materi kuliah: **Make Sense of Data with Analytics and AI**. PRD: bagian I dokumen "Implementasi Dua Materi Kuliah Praktisi Informatika".

Alur: unggah CSV/XLSX -> profil kualitas -> pembersihan eksplisit -> EDA -> model regresi/klasifikasi -> interpretasi -> laporan PDF. Semua diproses lokal; **tidak ada layanan AI generatif eksternal** (interpretasi berbasis aturan dari statistik dan hasil model).

## Fitur (dipetakan ke PRD)
| PRD | Fitur | Lokasi |
|---|---|---|
| FR-01 Upload | CSV/XLSX, batas 50 MB, validasi format dan struktur | `datasense/io.py` |
| FR-02 Kualitas data | baris/kolom, kosong per kolom, duplikat, tipe, statistik numerik, skor kualitas | `datasense/profiling.py` |
| FR-03 Pembersihan | hapus duplikat, strategi nilai kosong (isi mean/median/modus/konstanta, hapus baris/kolom), konversi tipe, pilih kolom; **setiap tindakan dicatat dan ditampilkan** | `datasense/cleaning.py` |
| FR-04 EDA | distribusi, kategori, korelasi (Pearson/Spearman), agregasi | `datasense/eda.py` |
| FR-05 Model | regresi (Linear, Ridge, Random Forest) dan klasifikasi (Logistic Regression, Random Forest); split latih/uji, cross-validation, peringatan kolom identifier | `datasense/modeling.py` |
| FR-06 Interpretasi | ringkasan otomatis + rekomendasi + keterbatasan; ditandai perlu diperiksa manusia | `datasense/interpretation.py` |
| FR-07 Laporan | PDF: dataset, kualitas, pembersihan, grafik, metrik model, keterbatasan; unduh dataset bersih dan prediksi | `datasense/reporting.py` |

## Struktur folder
```text
app.py                # titik masuk Streamlit
datasense/            # logika inti: io, profiling, cleaning, eda, modeling, interpretation, reporting
ui/                   # tampilan 6 tab (datasense_app.py) + gaya bersama (shared.py)
common/               # konfigurasi dan utilitas (hash, JSON)
data/samples/         # dataset contoh SINTETIS (churn, penjualan CSV/XLSX)
scripts/              # generate_samples.py
tests/                # 84 tes pytest
docs/                 # DATA_FORMAT, ARCHITECTURE, TEST_PLAN, PROGRESS, LAPORAN, screenshots/
.streamlit/           # config (batas unggah 50 MB)
```

## Menjalankan
```bash
pip install -r requirements.txt
python -m streamlit run app.py
python -m pytest -q          # 84 tes
python scripts/generate_samples.py   # buat ulang dataset contoh (SINTETIS, seed tetap)
```
Batas unggah 50 MB diatur di `.streamlit/config.toml`.

## Dataset contoh (SINTETIS, bukan data nyata)
`data/samples/customer_churn.csv` (klasifikasi), `sales_regression.csv` / `.xlsx` (regresi). Format: `docs/DATA_FORMAT.md`.

## Dokumen
`docs/DATA_FORMAT.md` - `docs/ARCHITECTURE.md` - `docs/TEST_PLAN.md` - `docs/PROGRESS.md` - `docs/LAPORAN.md` - `docs/screenshots/`.

## Batasan
Dataset contoh sintetis; analisis diagnostik bukan bukti sebab-akibat; skor model bergantung pada split acak tunggal (disertai cross-validation); tidak ada pemilihan model otomatis lintas banyak algoritma atau riwayat eksperimen (Tahap 3 PRD).
