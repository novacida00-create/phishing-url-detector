# Dataset

## asal data

LegitPhish Dataset versi 2 diunduh dari Mendeley Data:

- DOI: `10.17632/hx4m73v2sf.2`
- URL: https://data.mendeley.com/datasets/hx4m73v2sf/2
- Lisensi: CC BY 4.0
- Cakupan: seluruh dunia, bukan dataset khusus Indonesia

Berkas aslinya adalah `raw/LegitPhish_v2.zip`, berisi
`url_features_extracted1.csv`. CSV ini tidak dimasukkan ke repository publik.

## hasil audit

| pemeriksaan | hasil |
|---|---|
| jumlah baris di CSV | 101.219 |
| label 0 (phishing) | 63.678 |
| label 1 (legitimate) | 37.540 |
| baris dengan label kosong | 1 |
| URL duplikat | 346 |
| jumlah baris setelah label kosong dan duplikat dibuang | 100.872 |
| phishing setelah dibersihkan | 63.492 |
| legitimate setelah dibersihkan | 37.380 |
| jumlah grup domain unik | 53.739 |

Selisih antara total 101.219 dan jumlah dua kelas 101.218 ternyata berasal dari **satu baris
dengan `ClassLabel` kosong**. Baris ini dibuang saat preprocessing.

## fitur dan pengecekan ulang

CSV asli menyediakan 16 kolom fitur numerik. Nilainya dihitung ulang dari kolom `URL`
memakai `app/feature_extraction.py`, lalu dibandingkan dengan kolom asli. Sebagian besar
fitur cocok 100%; beberapa di antaranya memiliki sedikit perbedaan karena cara memotong
netloc pada URL yang tidak lazim:

- `has_ip_address`: 22 baris beda dari 101.218
- `path_length`: 1 baris beda
- `suspicious_file_extension`: 72 baris beda
- `domain_name_length`: 8 baris beda

Karena training dan website sama-sama memakai fungsi di `app/feature_extraction.py`,
perbedaan kecil itu tidak menimbulkan ketidaksesuaian antara data latih dan input web.

## file hasil olahan

- `processed/data_bersih.csv` — data setelah label kosong dan duplikat dibuang
- `processed/fitur.csv` — 16 fitur hasil hitungan ulang + label
- `processed/data_latih_contoh.csv` — contoh data latih
- `processed/data_uji.csv` — data uji yang dipakai menghitung metrik
