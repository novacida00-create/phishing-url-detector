"""Halaman web untuk mengecek URL phishing."""
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
from flask import Flask, render_template, request
from xgboost import DMatrix

from app.feature_extraction import FEATURE_LABELS, MAX_URL_LENGTH, extract_feature_dataframe

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "phishing_pipeline.joblib"
LABEL_MAP = {0: "Phishing", 1: "Legitimate"}

ARTIFACT = None
PIPELINE = None

if MODEL_PATH.exists():
    try:
        ARTIFACT = joblib.load(MODEL_PATH)
        PIPELINE = ARTIFACT["pipeline"]
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


def explain_prediction(features, prediction):
    """Ambil lima kontribusi Tree SHAP terbesar menggunakan pred_contribs XGBoost."""
    try:
        if "clf" not in PIPELINE.named_steps:
            return []
        booster = PIPELINE.named_steps["clf"].get_booster()
        values = booster.predict(DMatrix(features), pred_contribs=True)
        array = np.asarray(values)
        if array.ndim == 2:
            impacts = array[0, : len(features.columns)]
        else:
            impacts = array[: len(features.columns)]
        row = features.iloc[0]

        items = [
            {
                "name": FEATURE_LABELS.get(name, name),
                "value": round(float(row[name]), 2),
                "impact": float(impacts[i]),
            }
            for i, name in enumerate(features.columns)
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

    if PIPELINE is None:
        return render_template(
            "result.html",
            url=raw,
            label=None,
            confidence=None,
            shap_items=[],
            error="Model belum tersedia. Jalankan python train_model.py terlebih dahulu.",
        )

    features = extract_feature_dataframe([normalized])
    prediction = int(PIPELINE.predict(features)[0])
    confidence = None
    if hasattr(PIPELINE, "predict_proba"):
        probabilities = PIPELINE.predict_proba(features)[0]
        confidence = float(probabilities[list(PIPELINE.classes_).index(prediction)])

    return render_template(
        "result.html",
        url=raw,
        label=LABEL_MAP.get(prediction, "Unknown"),
        confidence=confidence,
        shap_items=explain_prediction(features, prediction),
        error=None,
    )
