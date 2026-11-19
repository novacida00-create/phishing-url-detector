"""Entrypoint khusus untuk Vercel.

File ini sengaja dipisah dari app.py supaya tidak tertukar dengan paket `app/`
saat Vercel memuat aplikasi.
"""
from app.web import app

__all__ = ["app"]
