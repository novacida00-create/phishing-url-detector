"""Halaman web untuk mengecek URL phishing."""
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, render_template, request

from app.feature_extraction import FEATURE_LABELS, FEATURE_NAMES, MAX_URL_LENGTH, extract_features
from app.xgb_model import XGBModel

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xgb_booster.json"
LABEL_MAP = {0: "Phishing", 1: "Legitimate"}

MODEL = None

if MODEL_PATH.exists():
    try:
        MODEL = XGBModel(MODEL_PATH)
    except Exception as error:
        print("Gagal memuat model:", error)


def validate_url(raw):
    """Cek panjang, skema, dan host. URL tidak pernah dikunjungi."""
    text = raw.strip()
    if not text:
        return None, "URL tidak boleh kosong."
    if len(text) > MAX_URL_LENGTH:
        return None, f"URL terlalu panjang (maksimal {MAX_URL_LENGTH} karakter)."

    candidate = text if "://" in text else "http://" + text
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None, "Format URL tidak valid."

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None, "Gunakan URL dengan skema http/https dan nama host yang valid."
    return candidate, None


def feature_values(url):
    """Kembalikan 16 fitur dalam urutan yang sama dengan urutan saat training."""
    extracted = extract_features(url)
    return [float(extracted[name]) for name in FEATURE_NAMES]


def explain_prediction(values):
    """Ambil lima kontribusi Tree SHAP terbesar."""
    try:
        impacts = MODEL.shap_values(values)[:-1]
        items = [
            {
                "name": FEATURE_LABELS.get(name, name),
                "value": round(float(value), 2),
                "impact": float(impacts[index]),
            }
            for index, (name, value) in enumerate(zip(FEATURE_NAMES, values))
        ]
        items.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return items[:5]
    except Exception as error:
        print("Gagal membuat penjelasan SHAP:", error)
        return []


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", url="", error=None)


@app.route("/predict", methods=["POST"])
def predict():
    raw = request.form.get("url", "")
    normalized, error = validate_url(raw)
    if error:
        return render_template("index.html", url=raw, error=error), 400

    if MODEL is None:
        return render_template(
            "result.html",
            url=raw,
            label=None,
            confidence=None,
            shap_items=[],
            error="Model belum tersedia. Jalankan python train_model.py terlebih dahulu.",
        )

    values = feature_values(normalized)
    probability = MODEL.predict_proba(values)
    prediction = 1 if probability >= 0.5 else 0
    confidence = probability if prediction == 1 else 1 - probability

    return render_template(
        "result.html",
        url=raw,
        label=LABEL_MAP.get(prediction, "Unknown"),
        confidence=confidence,
        shap_items=explain_prediction(values),
        error=None,
    )
