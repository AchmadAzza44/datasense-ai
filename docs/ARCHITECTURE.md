# Arsitektur
```text
app.py (Streamlit) -> ui/datasense_app.py (6 tab) -> datasense/
   io -> profiling -> cleaning -> eda -> modeling -> interpretation -> reporting (PDF)
common/ (konfigurasi, utilitas hash/JSON)   data/samples/   tests/
```
| Modul | Tanggung jawab |
|---|---|
| io | baca CSV/XLSX, batas ukuran/format, hash SHA-256 untuk penelusuran |
| profiling | profil kualitas: tipe, kosong, duplikat, statistik, skor kualitas |
| cleaning | `CleaningPlan` eksplisit -> `CleaningResult` dengan daftar tindakan; data asli tidak diubah |
| eda | agregat dan korelasi untuk grafik Plotly |
| modeling | pra-pemrosesan (imputasi, one-hot, skala), split, metrik regresi/klasifikasi, cross-validation, peringatan |
| interpretation | teks otomatis berbasis aturan dari profil dan hasil model |
| reporting | PDF (ReportLab + matplotlib) |

Prinsip: tiap modul fungsi murni yang diuji tanpa UI; UI hanya merangkai. Reproduksibilitas: seed tetap, hash dataset, konfigurasi pembersihan dan model ditampilkan di laporan.
