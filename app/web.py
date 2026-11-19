"""Halaman web untuk mengecek URL phishing."""
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import shap
from flask import Flask, render_template, request

from app.feature_extraction import FEATURE_LABELS, MAX_URL_LENGTH, extract_feature_dataframe

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "phishing_pipeline.joblib"
LABEL_MAP = {0: "Phishing", 1: "Legitimate"}

ARTIFACT = None
PIPELINE = None
BACKGROUND = None
_EXPLAINER = None

if MODEL_PATH.exists():
    try:
        ARTIFACT = joblib.load(MODEL_PATH)
        PIPELINE = ARTIFACT["pipeline"]
        BACKGROUND = ARTIFACT.get("background")
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


def get_explainer():
    global _EXPLAINER
    if _EXPLAINER is not None:
        return _EXPLAINER

    if "clf" in PIPELINE.named_steps and hasattr(PIPELINE.named_steps["clf"], "estimators_"):
        _EXPLAINER = shap.TreeExplainer(PIPELINE.named_steps["clf"])
    else:
        _EXPLAINER = shap.Explainer(lambda values: PIPELINE.predict_proba(values)[:, 1], BACKGROUND)
    return _EXPLAINER


def explain_prediction(features, prediction):
    """Ambil beberapa fitur yang paling berpengaruh pada satu prediksi."""
    try:
        values = get_explainer().shap_values(features)
        if isinstance(values, list):
            array = np.asarray(values[prediction])
        else:
            array = np.asarray(values)
            if array.ndim == 3:
                array = array[0, :, prediction] if array.shape[-1] == 2 else array[0, :, 0]
            elif array.ndim == 2 and array.shape[0] == 2:
                array = array[prediction]

        array = np.asarray(array).reshape(-1)
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
