"""Ekstraksi fitur URL.

Fitur-fitur di sini mengikuti daftar atribut LegitPhish Dataset v2. Cara
hitungnya juga dipakai waktu training, jadi tidak ada beda antara data yang
dipakai model dan URL yang diketik di website.

Fungsi ini hanya membaca string URL. Tidak ada koneksi keluar sama sekali.
"""
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse

MAX_URL_LENGTH = 2048

POPULAR_TLD = {"com", "org", "net", "edu", "gov"}
SUSPICIOUS_EXT = {"exe", "zip", "bin"}

FEATURE_NAMES = [
    "url_length",
    "has_ip_address",
    "dot_count",
    "https_flag",
    "url_entropy",
    "token_count",
    "subdomain_count",
    "query_param_count",
    "tld_length",
    "path_length",
    "has_hyphen_in_domain",
    "number_of_digits",
    "tld_popularity",
    "suspicious_file_extension",
    "domain_name_length",
    "percentage_numeric_chars",
]

FEATURE_LABELS = {
    "url_length": "Panjang URL",
    "has_ip_address": "Alamat IP pada URL",
    "dot_count": "Jumlah titik",
    "https_flag": "Menggunakan HTTPS",
    "url_entropy": "Entropi URL",
    "token_count": "Jumlah token",
    "subdomain_count": "Jumlah subdomain",
    "query_param_count": "Jumlah parameter query",
    "tld_length": "Panjang TLD",
    "path_length": "Panjang path",
    "has_hyphen_in_domain": "Tanda hubung pada domain",
    "number_of_digits": "Jumlah digit",
    "tld_popularity": "TLD umum",
    "suspicious_file_extension": "Ekstensi berkas mencurigakan",
    "domain_name_length": "Panjang nama domain",
    "percentage_numeric_chars": "Persentase karakter angka",
}


def _parse(url):
    try:
        return urlparse(url)
    except ValueError:
        return urlparse("")


def _entropy(text):
    if not text:
        return 0.0
    n = len(text)
    return -sum((count / n) * math.log2(count / n) for count in Counter(text).values())


def _is_ip(netloc):
    host = netloc.split("@")[-1].split(":")[0].strip("[]")
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def extract_features(url):
    """Hitung 16 fitur untuk satu URL dan kembalikan sebagai dictionary."""
    text = str(url).strip()
    parsed = _parse(text)
    netloc = parsed.netloc
    net_parts = netloc.split(".")
    tld = net_parts[-1].lower()

    digits = sum(char.isdigit() for char in text)
    tail = text.lower().split("?")[0].split("#")[0].rsplit(".", 1)[-1]

    return {
        "url_length": len(text),
        "has_ip_address": _is_ip(netloc),
        "dot_count": text.count("."),
        "https_flag": int(parsed.scheme == "https"),
        "url_entropy": _entropy(text),
        "token_count": len(re.split(r"[./?&=]", text)),
        "subdomain_count": netloc.count(".") - 1,
        "query_param_count": parsed.query.count("&") + 1,
        "tld_length": len(tld),
        "path_length": len(parsed.path),
        "has_hyphen_in_domain": int("-" in netloc),
        "number_of_digits": digits,
        "tld_popularity": int(tld in POPULAR_TLD),
        "suspicious_file_extension": int(tail in SUSPICIOUS_EXT),
        "domain_name_length": len(net_parts[-2]) if len(net_parts) >= 2 else 0,
        "percentage_numeric_chars": 100 * digits / len(text) if text else 0.0,
    }


def extract_feature_dataframe(urls):
    """Ekstraksi banyak URL sekaligus dengan urutan kolom yang tetap."""
    import pandas as pd

    rows = [extract_features(url) for url in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


__all__ = [
    "FEATURE_LABELS",
    "FEATURE_NAMES",
    "MAX_URL_LENGTH",
    "extract_features",
    "extract_feature_dataframe",
]
