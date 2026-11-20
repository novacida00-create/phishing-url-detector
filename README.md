# Deteksi URL Phishing dengan Machine Learning dan SHAP

Aplikasi untuk mengklasifikasikan sebuah URL menjadi **phishing** atau **legitimate**.
Model dilatih dengan dataset LegitPhish v2, lalu hasil prediksinya dijelaskan memakai SHAP.
Antarmuka web dibuat dengan Flask.

## ringkasan

- Dataset: LegitPhish Dataset Version 2, DOI `10.17632/hx4m73v2sf.2`, lisensi CC BY 4.0, cakupan global.
- Fitur: 16 atribut leksikal/struktural yang dihitung dari string URL. Fungsi yang sama dipakai saat training dan saat pengguna mengetik URL di web.
- Model yang dibandingkan: Random Forest, XGBoost, SVM. Model terpilih: **XGBoost**.
- Augmentasi: 27 URL legitimate terverifikasi ditambahkan hanya ke train/train_val; `https://career.astra.co.id/` kini diprediksi legitimate (99,90%).
- Penjelasan: kontribusi Tree SHAP (setara `pred_contribs`) dihitung langsung dari `models/xgb_booster.json`, menampilkan lima fitur terbesar; analisis global memakai SHAP `TreeExplainer`.
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
├── run_local.py
├── server.py
├── train_model.py
├── requirements.txt
└── Procfile
```

## menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python run_local.py
```

`requirements-dev.txt` memuat semua paket untuk training dan notebook. Untuk deployment yang hanya menjalankan website, cukup `requirements.txt`—ini sengaja dirampingkan supaya ukuran bundle Vercel tetap di bawah batas.

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
| accuracy | 0.9993 |
| precision (phishing) | 0.9991 |
| recall (phishing) | 0.9998 |
| F1 (phishing) | 0.9995 |
| ROC-AUC | 0.999997 |
| false negative rate | 0.00018 |

Dari 5.602 URL phishing di data uji, satu di antaranya terklasifikasi sebagai legitimate. Matriks kebingungan lengkap: TP 5.601, FN 1, FP 5, TN 3.339.
Angka lengkap ada di `models/metadata.json` dan `models/metrics.csv`.

## deployment

Aplikasi ini sudah live di https://phishing-url-detector-rust.vercel.app

Cara paling mudah untuk deploy ulang adalah **Vercel**:

1. Buka https://vercel.com/new lalu import repository GitHub ini.
2. Biarkan pengaturan default. Vercel mendeteksi Flask lewat `server.py`.
3. Klik **Deploy**. Tidak perlu environment variable.

Dependensi runtime dideklarasikan di `pyproject.toml` dan `requirements.txt`. Berkas
`.python-version` menentukan Python 3.12, sedangkan `vercel.json` membuang dataset,
notebook, dan cache dari bundle supaya ukurannya tetap di bawah batas function Vercel.

Kalau tetap memakai Render, `Procfile` berisi:

```
web: waitress-serve --port=$PORT app.web:app
```

Pada Render pilih compute plan **Free** dan env var `PYTHON_VERSION=3.12.0`. `waitress` dipakai
sebagai server WSGI karena murni Python dan juga bisa diuji di Windows.

Dataset mentah sengaja tidak dimasukkan ke repository publik. Berkas model berukuran sekitar 2,4 MB,
masih wajar untuk paket gratis, tetapi perlu diperiksa kembali terhadap batas layanan hosting.

## sitasi dataset

Potpelwar, R. S., Kulkarni, U. V., & Waghmare, J. M. (2025). *LegitPhish: A large-scale annotated
dataset for URL-based phishing detection.* Data in Brief, 63, 111972.
DOI dataset: `10.17632/hx4m73v2sf.2` · DOI artikel: `10.1016/j.dib.2025.111972`.
