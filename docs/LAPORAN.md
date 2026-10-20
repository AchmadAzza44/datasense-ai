# Laporan Singkat - DataSense AI
**Materi kuliah:** Make Sense of Data with Analytics and AI. PRD: bagian I (DataSense AI).

## Hubungan fitur dengan materi
| Konsep materi | Implementasi | Bukti |
|---|---|---|
| Data -> informasi -> insight | alur profil -> EDA -> interpretasi -> laporan | screenshot 02-06 |
| Pembersihan dan persiapan data | rencana pembersihan eksplisit; tiap tindakan dicatat; data asli utuh | `test_datasense_cleaning.py` |
| Analitik deskriptif (apa yang terjadi?) | statistik, distribusi, agregasi | tab Profil, EDA |
| Analitik diagnostik (faktor apa berkaitan?) | korelasi dan fitur penting, dengan catatan "bukan sebab-akibat" | laporan PDF |
| Analitik prediktif (apa yang mungkin terjadi?) | regresi dan klasifikasi dengan data uji terpisah + cross-validation | `test_datasense_modeling.py` |
| Analitik preskriptif (tindakan apa dipertimbangkan?) | rekomendasi otomatis (mis. buang kolom identifier) | tab Laporan |
| Evaluasi model dan validasi manusia | metrik sesuai jenis masalah; interpretasi ditandai perlu diperiksa | UI + PDF |

## Hasil nyata
- 84 tes lulus. Pada data contoh sintetis: regresi Ridge R² 0,906 (96 uji); klasifikasi churn akurasi 0,750-0,758 (120 uji; hasil bergantung pada split).
- Dua regresi pandas 3 ditemukan dan diperbaiki: pembersihan spasi tidak berjalan (dtype `str`) dan agregasi membuang kelompok kosong.

## Keterbatasan
Data sintetis; satu split acak; interpretasi berbasis aturan (bukan AI generatif); korelasi bukan kausalitas; belum ada uji pengguna, pemilihan model otomatis, atau riwayat eksperimen (PRD Tahap 3).

## Demonstrasi
`python -m streamlit run app.py` -> pilih `customer_churn.csv` -> Muat -> Profil -> Pembersihan -> EDA -> Model (target `churn`, klasifikasi) -> Laporan.
