"""Entrypoint untuk Vercel.

Vercel mencari objek Flask bernama `app` di berkas ini. Import memakai
`app.web` supaya tidak tertukar dengan berkas di akar proyek.
"""
from app.web import app

__all__ = ["app"]
