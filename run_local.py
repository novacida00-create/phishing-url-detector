"""Entry point untuk menjalankan aplikasi secara lokal: python run_local.py.

Definisi route ada di `app/web.py`. Berkas ini sengaja tidak bernama app.py
supaya tidak tertukar dengan paket `app/` saat diimpor oleh server.
"""
from app.web import app

if __name__ == "__main__":
    from waitress import serve

    print("Aplikasi siap di http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000)
