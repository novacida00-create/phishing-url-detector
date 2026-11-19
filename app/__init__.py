"""Paket aplikasi Flask untuk deteksi URL phishing.

Mengekspos `app` supaya `waitress-serve app:app` bisa dipakai. Nama paket ini
sengaja `app`, sehingga route tetap berada di modul `app.web`.
"""
from app.web import app  # noqa: F401

__all__ = ["app"]
