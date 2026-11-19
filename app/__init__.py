"""Paket aplikasi Flask untuk deteksi URL phishing.

Mengekspos `app` sehingga `gunicorn app:app` (Procfile) bekerja, karena `app`
di sini merujuk ke paket ini, bukan berkas app.py di akar proyek.
"""
from app.web import app  # noqa: F401

__all__ = ["app"]
