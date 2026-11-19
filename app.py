"""Entry point untuk menjalankan aplikasi secara lokal: python app.py.

Definisi route ada di `app/web.py`; paket `app` mengekspos objek `app`
sehingga perintah `waitress-serve app:app` di Procfile juga bekerja.
"""
from app import app

if __name__ == "__main__":
    from waitress import serve

    print("Aplikasi siap di http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000)
