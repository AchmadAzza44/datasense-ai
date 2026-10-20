# Format Data

## Umum
- Format: CSV (UTF-8) atau XLSX (lembar pertama). Maksimum 50 MB. Baris pertama = nama kolom. File kosong, tanpa kolom, atau dengan ekstensi lain ditolak dengan pesan yang menyebut penyebabnya.
- Tipe kolom dideteksi otomatis (numerik, kategorikal, tanggal, teks, boolean); konversi hanya terjadi bila pengguna memintanya pada langkah Pembersihan.

## `customer_churn.csv` (SINTETIS; 608 baris x 9 kolom; target klasifikasi `churn`)
| Kolom | Tipe | Keterangan |
|---|---|---|
| id_pelanggan | teks | identifier unik; aplikasi memperingatkan agar tidak dipakai sebagai fitur |
| usia | numerik | tahun |
| masa_berlangganan_bulan | numerik | bulan |
| tagihan_bulanan | numerik | ada nilai kosong disengaja |
| jumlah_panggilan_support | numerik | |
| jenis_kontrak | kategorikal | Bulanan / Tahunan / Dua Tahun |
| jenis_internet | kategorikal | DSL / Fiber / Tidak Ada |
| skor_kepuasan | numerik | 0-10 |
| churn | kategorikal | Ya / Tidak |

Disengaja: nilai kosong (0,55% sel) dan 8 baris duplikat untuk menguji alur pembersihan.

## `sales_regression.csv` / `.xlsx` (SINTETIS; 484 baris x 8 kolom; target regresi `penjualan_juta`)
`tanggal`, `toko`, `jenis_toko`, `biaya_iklan_juta`, `harga_rata_rata`, `jumlah_pengunjung`, `promosi`, `penjualan_juta`. Disengaja: 4 baris duplikat dan sedikit nilai kosong (0,13% sel).

## Data Anda sendiri
Tidak ada skema wajib; pilih kolom target dan jenis masalah (regresi untuk nilai numerik, klasifikasi untuk kategori). Klasifikasi memerlukan target dengan beberapa kelas yang masing-masing punya cukup contoh; model menolak target yang tidak layak dengan pesan jelas.
