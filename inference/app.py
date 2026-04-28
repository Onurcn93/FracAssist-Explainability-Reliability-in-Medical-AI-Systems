"""
inference/app.py — FracAssist Flask server.

Run from repo root:
    python inference/app.py

Routes:
    GET  /         → serve index.html
    GET  /health   → {"status": "ok", "models_loaded": true}
    POST /predict  → multipart/form-data with 'image' field → inference JSON
"""

import csv
import datetime
import os
import sys
import tempfile

from flask import Flask, jsonify, make_response, request, send_file
from flask_cors import CORS
from PIL import Image

# Ensure both inference/ and repo root are on path so all imports resolve
_INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT     = os.path.dirname(_INFERENCE_DIR)
sys.path.insert(0, _INFERENCE_DIR)
sys.path.insert(0, _REPO_ROOT)

from config import CONFIG
from predict import load_models, predict as run_predict

_ROOT = _REPO_ROOT

app = Flask(__name__, static_folder=_ROOT, static_url_path='')
CORS(app)  # Required: index.html may be opened as file:// or cross-origin

_ALLOWED = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/favicon.ico")
def favicon():
    return make_response("", 204)


@app.route("/")
def index():
    return send_file(CONFIG["index_html"])


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "device": CONFIG["device"],
    })


_FRACTATLAS_DIRS   = [
    os.path.join(_ROOT, "FracAtlas", "images", "Fractured"),
    os.path.join(_ROOT, "FracAtlas", "images", "Non_fractured"),
]
_REVIEW_CSV        = os.path.join(_ROOT, "review", "expert_review.csv")
_REVIEW_IMAGES_DIR = os.path.join(_ROOT, "review", "images")
_CSV_FIELDS        = ["image_id", "gel_probability", "gel_label",
                      "resnet_probability", "densenet_probability",
                      "efficientnet_probability", "true_label", "status", "timestamp"]


@app.route("/fractatlas/<filename>")
def fractatlas_image(filename):
    for folder in _FRACTATLAS_DIRS:
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            return send_file(path)
    return make_response("Not found", 404)


@app.route("/review-queue", methods=["GET"])
def review_queue():
    rows = []
    if os.path.exists(_REVIEW_CSV):
        with open(_REVIEW_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return jsonify(rows)


@app.route("/send-review", methods=["POST"])
def send_review():
    data = request.json or {}
    image_id = data.get("image_id", "").strip()
    if not image_id:
        return jsonify({"error": "image_id required"}), 400

    # Reject duplicate
    if os.path.exists(_REVIEW_CSV):
        with open(_REVIEW_CSV, newline="", encoding="utf-8") as f:
            existing = [r["image_id"] for r in csv.DictReader(f)]
        if image_id in existing:
            return jsonify({"error": "already in review queue"}), 409

    # Generate thumbnail from FracAtlas source; derive true label from folder
    os.makedirs(_REVIEW_IMAGES_DIR, exist_ok=True)
    thumb_path = os.path.join(_REVIEW_IMAGES_DIR, image_id)
    true_label = ""
    for folder in _FRACTATLAS_DIRS:
        src = os.path.join(folder, image_id)
        if os.path.isfile(src):
            true_label = os.path.basename(folder)  # "Fractured" or "Non_fractured"
            if not os.path.exists(thumb_path):
                img = Image.open(src).convert("RGB")
                img.thumbnail((96, 96), Image.LANCZOS)
                img.save(thumb_path, quality=88)
            break

    # Append row to CSV
    row = {
        "image_id":               image_id,
        "gel_probability":        data.get("gel_probability", ""),
        "gel_label":              data.get("gel_label", ""),
        "resnet_probability":     data.get("resnet_probability", ""),
        "densenet_probability":   data.get("densenet_probability", ""),
        "efficientnet_probability": data.get("efficientnet_probability", ""),
        "true_label":             true_label,
        "status":                 "pending",
        "timestamp":              datetime.datetime.now().isoformat(timespec="seconds"),
    }
    write_header = not os.path.exists(_REVIEW_CSV)
    with open(_REVIEW_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # --- Validate upload ---
    if "image" not in request.files:
        return jsonify({"error": "No 'image' field in request"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in _ALLOWED:
        return jsonify({"error": "Invalid file type. Accepted: JPG, PNG"}), 400

    tmp_path = None
    try:
        # Save to a temp file; delete after inference
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            tmp_path = f.name
            file.save(tmp_path)

        inference_mode = request.form.get("inference_mode", "gel")
        if inference_mode not in {"gel", "yolo", "resnet"}:
            inference_mode = "gel"

        result = run_predict(tmp_path, CONFIG, inference_mode=inference_mode)
        return jsonify(result)

    except Exception as e:
        # Strip filesystem paths from error messages before returning
        msg = str(e)
        if os.sep in msg:
            msg = msg.split(os.sep)[-1]
        return jsonify({"error": msg}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[INFO] Loading models...")
    try:
        load_models(CONFIG)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"[INFO] Device: {CONFIG['device']}")
    print("[INFO] FracAssist running → http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
