import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.joblib"
FEATURES_PATH = BASE_DIR / "feature_names.json"
METADATA_PATH = BASE_DIR / "model_metadata.json"
MONITORING_DIR = BASE_DIR / "monitoring"
LOG_PATH = MONITORING_DIR / "prediction_logs.csv"

# The model was trained on numeric / label-encoded columns in the notebook.
# Incoming values must use the same feature names, order, encoding, and units.
model = joblib.load(MODEL_PATH)
with FEATURES_PATH.open("r", encoding="utf-8") as f:
    FEATURE_NAMES = json.load(f)
if not isinstance(FEATURE_NAMES, list) or not FEATURE_NAMES:
    raise RuntimeError("feature_names.json harus berisi daftar nama fitur yang tidak kosong.")

METADATA = {}
if METADATA_PATH.exists():
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        METADATA = json.load(f)

MODEL_NAME = METADATA.get("model_name", "Random Forest")
MONITORING_DIR.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)


def log_prediction(prediction: int, probability: float) -> None:
    """Log hanya metrik prediksi; jangan simpan payload pelanggan mentah."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": int(prediction),
        "churn_probability": float(probability),
    }
    has_header = LOG_PATH.exists() and LOG_PATH.stat().st_size > 0
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


@app.get("/")
def home():
    return jsonify({
        "service": "Telecom Customer Churn API",
        "model": MODEL_NAME,
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "monitoring": "GET /monitoring/summary",
        },
    }), 200


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "expected_feature_count": len(FEATURE_NAMES),
    }), 200


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Kirim JSON object dengan Content-Type application/json."}), 400

    features = body.get("features", body)
    if not isinstance(features, dict):
        return jsonify({"error": "Key 'features' harus berupa JSON object."}), 400

    missing = [name for name in FEATURE_NAMES if name not in features]
    extra = [name for name in features if name not in FEATURE_NAMES]
    if missing or extra:
        return jsonify({
            "error": "Nama fitur harus sama persis dengan feature_names.json.",
            "missing_features": missing,
            "unexpected_features": extra,
        }), 400

    try:
        # Keep exactly the training order, enforce numeric inputs, and reject NaN/Inf.
        row = pd.DataFrame([{name: features[name] for name in FEATURE_NAMES}], columns=FEATURE_NAMES)
        row = row.apply(pd.to_numeric, errors="raise").astype(float)
        if not all(math.isfinite(float(v)) for v in row.iloc[0].tolist()):
            return jsonify({"error": "Semua nilai fitur harus berupa angka finite, bukan NaN/Infinity."}), 400

        raw_prediction = model.predict(row)[0]
        prediction = int(raw_prediction)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            classes = list(model.classes_)
            positive_index = next((i for i, label in enumerate(classes) if str(label) == "1"), None)
            if positive_index is None:
                return jsonify({"error": "Model tidak memiliki kelas positif 1 untuk label churn."}), 500
            probability = float(probabilities[positive_index])
        else:
            probability = float(prediction)
    except (TypeError, ValueError) as exc:
        return jsonify({
            "error": "Input harus berupa fitur numerik yang sudah di-encode seperti saat training.",
            "detail": str(exc),
        }), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediksi gagal. Periksa kecocokan model, versi scikit-learn, dan input."}), 500

    try:
        log_prediction(prediction, probability)
    except OSError:
        app.logger.exception("Gagal menulis log monitoring")

    return jsonify({
        "model": MODEL_NAME,
        "prediction": prediction,
        "label": "Churn" if prediction == 1 else "Tidak Churn",
        "churn_probability": round(probability, 6),
    }), 200


@app.get("/monitoring/summary")
def monitoring_summary():
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return jsonify({
            "status": "no_data",
            "prediction_requests": 0,
            "message": "Belum ada prediksi yang tercatat.",
        }), 200

    try:
        logs = pd.read_csv(LOG_PATH)
        if logs.empty:
            return jsonify({"status": "no_data", "prediction_requests": 0}), 200
        predictions = pd.to_numeric(logs["prediction"], errors="coerce").dropna()
        probabilities = pd.to_numeric(logs["churn_probability"], errors="coerce").dropna()
        return jsonify({
            "status": "ok",
            "prediction_requests": int(len(logs)),
            "churn_predictions": int((predictions == 1).sum()),
            "non_churn_predictions": int((predictions == 0).sum()),
            "churn_prediction_rate": round(float((predictions == 1).mean()), 4) if not predictions.empty else None,
            "average_churn_probability": round(float(probabilities.mean()), 4) if not probabilities.empty else None,
            "high_risk_predictions_probability_ge_0_70": int((probabilities >= 0.70).sum()),
            "last_prediction_at": str(logs["timestamp"].iloc[-1]),
            "note": "Ringkasan ini memantau prediksi. Accuracy/recall membutuhkan label aktual.",
        }), 200
    except Exception:
        app.logger.exception("Monitoring summary failed")
        return jsonify({"error": "Gagal membaca ringkasan monitoring."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
