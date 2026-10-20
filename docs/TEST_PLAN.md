# Test Plan
`python -m pytest -q` -> **84 lulus** (Python 3.13, pandas 3.0.5, scikit-learn 1.9, Streamlit 1.66; juga diuji dengan Streamlit 1.50.0 pada versi gabungan sebelumnya).
| Area | Yang diuji | Berkas |
|---|---|---|
| Upload | format ditolak, ukuran melebihi batas, file kosong, XLSX, hash | test_datasense_io.py |
| Profil | tipe, kosong, duplikat, skor kualitas, kolom konstan/identifier | test_datasense_profiling.py |
| Pembersihan | duplikat, tiap strategi kosong, konversi tipe, strip spasi (regresi pandas 3), tindakan tercatat, data asli tidak berubah | test_datasense_cleaning.py |
| EDA | korelasi, agregasi (termasuk kunci grup kosong; regresi pandas 3) | test_datasense_eda.py |
| Model | regresi dan klasifikasi, split, metrik, penolakan target tidak layak, peringatan identifier | test_datasense_modeling.py |
| Interpretasi dan laporan | bagian wajib interpretasi, tindakan pembersihan dan metrik model ikut tercantum, Markdown, PDF valid (tanpa model, regresi, klasifikasi) | test_datasense_report.py |
| UI | halaman tampil dengan 6 tab; dataset contoh dapat dimuat | test_ui_smoke.py |

## Verifikasi alur di browser (Playwright/Chromium, data contoh)
Muat `customer_churn.csv` -> profil (608 baris, skor kualitas 99,2) -> pembersihan -> EDA -> model klasifikasi `churn` -> hasil: **akurasi 0,758, F1 weighted 0,751**, cross-validation F1 0,706 +/- 0,060, peringatan identifier `id_pelanggan` dan keterbatasan tampil -> tab laporan. Tidak ada galat JavaScript. Screenshot: `docs/screenshots/`.

## Hasil model pada dataset contoh (pemrosesan skrip, tanpa duplikat; kolom identifier tidak dipakai)
| Kasus | Model | Hasil (data uji) |
|---|---|---|
| Regresi `penjualan_juta` (384 latih / 96 uji) | Ridge | R² 0,906; MAE 5,73; RMSE 7,32; MAPE 7,4% |
| Klasifikasi `churn` (480 / 120) | Logistic Regression | akurasi 0,750; F1 weighted 0,747; ROC-AUC 0,794 |
Data sintetis: angka hanya membuktikan alur berjalan, bukan kemampuan prediksi pada data nyata.

Belum diuji: uji pengguna, performa pada file mendekati 50 MB, pembaca layar.
