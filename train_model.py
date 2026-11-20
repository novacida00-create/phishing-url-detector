"""Latih model klasifikasi URL phishing dan simpan hasilnya.

 Jalankan: python train_model.py

Data mentah dibaca dari dataset/raw, fitur dihitung ulang dari string URL
memakai modul app/feature_extraction.py, lalu model dievaluasi pada data uji
yang dipisahkan berdasarkan domain.
"""
import json
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from app.feature_extraction import FEATURE_NAMES, extract_feature_dataframe

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "dataset" / "raw" / "LegitPhish Dataset" / "url_features_extracted1.csv"
PROCESSED_DIR = ROOT / "dataset" / "processed"
MODEL_DIR = ROOT / "models"

# URL legitimate terverifikasi yang ditambahkan ke data latih. Dataset LegitPhish
# tidak memuat domain korporat Indonesia seperti astra.co.id, sehingga model
# sebelumnya memberi false positive pada URL resmi tersebut. Penambahan ini hanya
# masuk ke train/train_val, tidak ke data validasi maupun data uji.
TAMBAHAN_LEGITIMATE_URLS = [
    "https://career.astra.co.id/",
    "https://career.astra.co.id",
    "http://career.astra.co.id/",
    "https://www.astra.co.id/",
    "https://astra.co.id/",
    "https://career.astra.co.id/jobs",
    "https://career.astra.co.id/candidate",
    "https://www.astra.co.id/career",
    "https://www.danamon.co.id/",
    "https://www.bca.co.id/",
    "https://www.bni.co.id/",
    "https://www.bri.co.id/",
    "https://www.telkom.co.id/",
    "https://www.pln.co.id/",
    "https://www.kemenkeu.go.id/",
    "https://www.kominfo.go.id/",
    "https://www.bps.go.id/",
    "https://www.tokopedia.com/",
    "https://www.blibli.com/",
    "https://www.bukalapak.com/",
    "https://www.gojek.com/",
    "https://www.truemoney.co.id/",
    "https://www.jagajaga.co.id/",
    "https://www.erlangga.co.id/",
    "https://www.mazda.co.id/",
    "https://www.kisi.co.id/",
    "https://www.kubuku.co.id/",
]


def domain_group(url):
    """Ambil dua label terakhir dari host sebagai pengelompok domain."""
    try:
        netloc = urlparse(url).netloc
    except ValueError:
        netloc = ""
    host = netloc.split("@")[-1].split(":")[0].lower()
    labels = [part for part in host.split(".") if part]
    if len(labels) >= 2:
        return ".".join(labels[-2:])
    return host or "unknown"


def load_data():
    df = pd.read_csv(RAW_CSV)
    audit = {
        "baris_mentah": int(len(df)),
        "label_kosong": int(df["ClassLabel"].isna().sum()),
        "url_duplikat": int(df["URL"].duplicated().sum()),
        "kolom": list(df.columns),
    }

    df = df.dropna(subset=["ClassLabel"]).drop_duplicates(subset=["URL"]).reset_index(drop=True).copy()
    audit["baris_setelah_pembersihan"] = int(len(df))
    audit["phishing"] = int((df["ClassLabel"] == 0).sum())
    audit["legitimate"] = int((df["ClassLabel"] == 1).sum())

    X = extract_feature_dataframe(df["URL"])
    y = df["ClassLabel"].astype(int)
    groups = df["URL"].map(domain_group)
    audit["jumlah_domain"] = int(groups.nunique())
    return X, y, groups, audit


def split_data(X, y, groups):
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(split1.split(X, y, groups))

    split2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_rel, test_rel = next(split2.split(X.iloc[temp_idx], y.iloc[temp_idx], groups.iloc[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "train_val": np.concatenate([train_idx, val_idx]),
    }


def build_models():
    return {
        "Random Forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=120,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                )
            ]
        ),
        "XGBoost": Pipeline(
            [
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=42,
                    ),
                )
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
            ]
        ),
    }


def evaluate(model, X, y):
    prediction = model.predict(X)
    proba_phishing = model.predict_proba(X)[:, list(model.classes_).index(0)]
    cm = confusion_matrix(y, prediction, labels=[0, 1])

    # Positive class = phishing (label 0)
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score((y == 0).astype(int), proba_phishing)),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion_matrix": {
            "true_phishing_predicted_phishing": int(tp),
            "true_phishing_predicted_legitimate": int(fn),
            "true_legitimate_predicted_phishing": int(fp),
            "true_legitimate_predicted_legitimate": int(tn),
        },
    }


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups, audit = load_data()
    idx = split_data(X, y, groups)

    # Tambahkan contoh legitimate terverifikasi hanya ke data latih.
    X_tambahan = extract_feature_dataframe(pd.Series(TAMBAHAN_LEGITIMATE_URLS))
    y_tambahan = pd.Series([1] * len(TAMBAHAN_LEGITIMATE_URLS), dtype=int)
    groups_tambahan = pd.Series([domain_group(url) for url in TAMBAHAN_LEGITIMATE_URLS])
    offset = len(X)
    indeks_tambahan = np.arange(offset, offset + len(X_tambahan))

    X_gabung = pd.concat([X, X_tambahan], ignore_index=True)
    y_gabung = pd.concat([y.reset_index(drop=True), y_tambahan], ignore_index=True)
    groups_gabung = pd.concat([groups.reset_index(drop=True), groups_tambahan], ignore_index=True)

    idx_latih = dict(idx)
    idx_latih["train"] = np.concatenate([idx["train"], indeks_tambahan])
    idx_latih["train_val"] = np.concatenate([idx["train_val"], indeks_tambahan])

    audit["augmentasi_legitimate"] = len(TAMBAHAN_LEGITIMATE_URLS)
    print("Audit:", json.dumps(audit, indent=2))
    print("Ukuran split:", {k: len(v) for k, v in idx_latih.items()})

    X_train, y_train = X_gabung.iloc[idx_latih["train"]], y_gabung.iloc[idx_latih["train"]]
    X_val, y_val = X.iloc[idx["val"]], y.iloc[idx["val"]]

    results = {}
    for name, model in build_models().items():
        if name == "SVM":
            # SVM RBF dengan probability=True cukup berat pada 80 ribu baris.
            # Latih pada 30 ribu sampel training yang dipilih secara acak.
            sample = X_train.sample(n=min(30000, len(X_train)), random_state=42).index
            fit_X, fit_y = X_train.loc[sample], y_train.loc[sample]
        else:
            fit_X, fit_y = X_train, y_train

        print(f"\nMelatih {name} ...")
        model.fit(fit_X, fit_y)
        results[name] = evaluate(model, X_val, y_val)
        print(f"  F1={results[name]['f1']:.4f}  ROC-AUC={results[name]['roc_auc']:.4f}")

        if name == "SVM":
            results[name]["catatan"] = "Dilatih pada 30.000 sampel data training."

    best_name = max(results, key=lambda name: results[name]["f1"])
    print(f"\nModel terpilih berdasarkan F1: {best_name}")

    final_model = build_models()[best_name]
    if best_name == "SVM":
        sample = X_gabung.iloc[idx_latih["train_val"]].sample(
            n=min(30000, len(idx_latih["train_val"])), random_state=42
        ).index
        final_model.fit(X_gabung.loc[sample], y_gabung.loc[sample])
    else:
        final_model.fit(X_gabung.iloc[idx_latih["train_val"]], y_gabung.iloc[idx_latih["train_val"]])

    test_metrics = evaluate(final_model, X.iloc[idx["test"]], y.iloc[idx["test"]])
    print("Hasil data uji:", json.dumps(test_metrics, indent=2))

    background = X_gabung.iloc[idx_latih["train_val"]].sample(
        n=min(100, len(idx_latih["train_val"])), random_state=42
    ).reset_index(drop=True)
    artifact = {
        "pipeline": final_model,
        "background": background,
        "feature_names": FEATURE_NAMES,
        "model_name": best_name,
        "label_map": {0: "Phishing", 1: "Legitimate"},
    }
    joblib.dump(artifact, MODEL_DIR / "phishing_pipeline.joblib")

    processed = X.copy()
    processed["ClassLabel"] = y
    processed["domain_group"] = groups
    processed.iloc[idx["test"]].to_csv(PROCESSED_DIR / "data_uji.csv", index=False)
    processed.iloc[idx["train_val"]].sample(n=min(5000, len(idx["train_val"])), random_state=42).to_csv(
        PROCESSED_DIR / "data_latih_contoh.csv", index=False
    )

    tambahan_simpan = X_tambahan.copy()
    tambahan_simpan.insert(0, "URL", TAMBAHAN_LEGITIMATE_URLS)
    tambahan_simpan["ClassLabel"] = 1
    tambahan_simpan["domain_group"] = groups_tambahan.values
    tambahan_simpan.to_csv(PROCESSED_DIR / "data_tambahan_legitimate.csv", index=False)

    metadata = {
        "model": best_name,
        "fitur": FEATURE_NAMES,
        "label": {"0": "phishing", "1": "legitimate"},
        "pembagian_data": {
            "metode": "GroupShuffleSplit berdasarkan dua label terakhir domain",
            "jumlah": {
                "train": len(idx_latih["train"]),
                "val": len(idx["val"]),
                "test": len(idx["test"]),
                "train_val": len(idx_latih["train_val"]),
            },
            "random_state": 42,
        },
        "audit_dataset": audit,
        "metrik_validasi": results,
        "metrik_uji": test_metrics,
        "dataset": {
            "nama": "LegitPhish Dataset, Version 2",
            "doi": "10.17632/hx4m73v2sf.2",
            "lisensi": "CC BY 4.0",
            "cakupan": "global",
            "augmentasi_latih": {
                "jumlah": len(TAMBAHAN_LEGITIMATE_URLS),
                "label": 1,
                "keterangan": "URL legitimate terverifikasi, hanya ditambahkan ke train dan train_val",
            },
        },
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for name, metrics in results.items():
        rows.append({"model": name, "data": "validasi", **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
    rows.append({"model": best_name, "data": "uji", **{k: v for k, v in test_metrics.items() if k != "confusion_matrix"}})
    pd.DataFrame(rows).to_csv(MODEL_DIR / "metrics.csv", index=False)

    print("\nSelesai. Model disimpan di", MODEL_DIR / "phishing_pipeline.joblib")


if __name__ == "__main__":
    main()
