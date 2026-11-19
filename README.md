# Deteksi URL Phishing dengan Machine Learning dan SHAP

Aplikasi untuk mengklasifikasikan sebuah URL menjadi **phishing** atau **legitimate**.
Model dilatih dengan dataset LegitPhish v2, lalu hasil prediksinya dijelaskan memakai SHAP.
Antarmuka web dibuat dengan Flask.

## ringkasan

- Dataset: LegitPhish Dataset Version 2, DOI `10.17632/hx4m73v2sf.2`, lisensi CC BY 4.0, cakupan global.
- Fitur: 16 atribut leksikal/struktural yang dihitung dari string URL. Fungsi yang sama dipakai saat training dan saat pengguna mengetik URL di web.
- Model yang dibandingkan: Random Forest, XGBoost, SVM. Model terpilih: **Random Forest**.
- Penjelasan: `TreeExplainer` dari SHAP, menampilkan lima fitur dengan kontribusi terbesar.
- Web: tidak pernah mengunjungi URL yang dimasukkan pengguna; hanya menganalisis stringnya.

## struktur folder

```
phishing-url-detector/
├── dataset/
│   ├── raw/                     # CSV asli hasil unduhan
│   ├── processed/               # data bersih, fitur, dan data uji
│   └── README.md
├── notebooks/
│   ├── 01_explorasi_dataset.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_training_model.ipynb
│   ├── 05_evaluasi_model.ipynb
│   └── 06_explainable_ai.ipynb
├── models/
│   ├── phishing_pipeline.joblib # pipeline + data latih ringkas untuk SHAP
│   ├── metadata.json
│   └── metrics.csv
├── app/
│   ├── templates/
│   ├── static/
│   ├── feature_extraction.py
│   └── web.py
├── app.py
├── train_model.py
├── requirements.txt
└── Procfile
```

## menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:5000`, masukkan URL, lalu tekan **Periksa URL**.

Kalau model ingin dilatih ulang dari CSV asli:

```bash
python train_model.py
```

Perintah ini menulis ulang `models/phishing_pipeline.joblib`, `models/metadata.json`,
`models/metrics.csv`, dan beberapa file di `dataset/processed/`.

## hasil singkat

Evaluasi memakai 8.946 URL uji yang tidak dipakai saat training. Pembagian data dilakukan
berdasarkan dua label terakhir domain supaya URL dari domain yang sama tidak bocor ke split lain.

| metrik | nilai |
|---|---|
| accuracy | 0.9992 |
| precision (phishing) | 0.9989 |
| recall (phishing) | 0.9998 |
| F1 (phishing) | 0.9994 |
| ROC-AUC | 0.999997 |
| false negative rate | 0.00018 |

Dari 5.602 URL phishing di data uji, satu di antaranya terklasifikasi sebagai legitimate.
Angka lengkap ada di `models/metadata.json` dan `models/metrics.csv`.

## deployment

`Procfile` sudah berisi:

```
web: gunicorn app:app
```

`waitress` dipakai sebagai server WSGI produksi. Server ini murni Python, jadi bisa dijalankan
baik di Linux (Render) maupun di Windows saat pengujian lokal.

Langkah singkat: daftarkan repository ke Render, pilih Web Service, gunakan Python,
isi build command dengan `pip install -r requirements.txt` dan start command dengan
`waitress-serve --port=$PORT app:app`. Setelah aktif, coba beberapa URL untuk memastikan form dan hasil prediksi tampil.

Dataset mentah sengaja tidak dimasukkan ke repository publik. Berkas model berukuran sekitar 2,4 MB,
masih wajar untuk paket gratis, tetapi perlu diperiksa kembali terhadap batas layanan hosting.

## sitasi dataset

Potpelwar, R. S., Kulkarni, U. V., & Waghmare, J. M. (2025). *LegitPhish: A large-scale annotated
dataset for URL-based phishing detection.* Data in Brief, 63, 111972.
DOI dataset: `10.17632/hx4m73v2sf.2` · DOI artikel: `10.1016/j.dib.2025.111972`.
