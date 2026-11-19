"""Entrypoint untuk Vercel.

Vercel mencari objek Flask bernama `app` di berkas ini. Import memakai
`app.web` supaya tidak tertukar dengan berkas app.py di akar proyek.
"""
try:
    from app.web import app
except Exception:
    # Sementara membantu membaca penyebab error saat deployment pertama.
    import traceback

    _detail = traceback.format_exc()

    def app(environ, start_response):
        start_response(
            "500 Internal Server Error",
            [("Content-Type", "text/plain; charset=utf-8")],
        )
        return [_detail.encode("utf-8")]
