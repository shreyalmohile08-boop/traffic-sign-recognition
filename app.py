"""
app.py
Traffic Sign Recognition System - Flask Web Application
Provides:
- GET  /                : Main Recognition Web Interface
- POST /predict          : REST API for sign classification (File or Base64)
- GET  /dashboard        : Telemetry & Model Analytics Dashboard
- GET  /about            : Architecture, Dataset Taxonomy & Info
- GET  /api/stats        : Model accuracy, class breakdown, and parameters
- GET  /api/history      : Recent inference history
"""

import os
import io
import time
import json
import base64
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
from werkzeug.utils import secure_filename
from PIL import Image

# Import inference engine
from predict import predict_traffic_sign, load_resources

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_test_images')
SAMPLES_STATIC = os.path.join(BASE_DIR, 'static', 'img', 'samples')
HISTORY_FILE = os.path.join(BASE_DIR, 'prediction_history.json')
METRICS_FILE = os.path.join(BASE_DIR, 'model_metrics.json')
LABELS_CSV = os.path.join(BASE_DIR, 'traffic_sign_labels.csv')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_STATIC, exist_ok=True)

# Copy sample images to static/img/samples for clean frontend serving
if os.path.exists(SAMPLE_DIR):
    for fn in os.listdir(SAMPLE_DIR):
        src = os.path.join(SAMPLE_DIR, fn)
        dst = os.path.join(SAMPLES_STATIC, fn)
        if os.path.isfile(src) and not os.path.exists(dst):
            try:
                import shutil
                shutil.copyfile(src, dst)
            except Exception:
                pass

# Pre-load resources into memory at app startup
try:
    load_resources()
    print("TensorFlow model & metadata loaded successfully.")
except Exception as e:
    print(f"Notice: Model resources not fully initialized: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(entry):
    history = load_history()
    history.append(entry)
    # Keep last 50 entries
    if len(history) > 50:
        history = history[-50:]
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print("Failed to persist prediction history:", e)

# =========================================================================
# Web Routes
# =========================================================================

@app.route('/')
def index():
    """Renders the main recognition interface."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Renders the telemetry dashboard with charts and logs."""
    return render_template('dashboard.html')

@app.route('/about')
def about():
    """Renders system architecture, dataset, and hackathon details."""
    return render_template('about.html')

@app.route('/sample_test_images/<path:filename>')
def serve_sample_image(filename):
    """Serves test images from sample_test_images directory."""
    return send_from_directory(SAMPLE_DIR, filename)

# =========================================================================
# Prediction API
# =========================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts:
    1. 'file' : Multipart form upload
    2. 'image_data' : Base64 data URL string from webcam snapshot
    Returns JSON prediction response.
    """
    saved_filepath = None
    web_image_url = None

    # Case A: File upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected",
                "message": "Please select an image file to upload."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Unsupported file format",
                "message": "Unsupported file format. Please upload PNG, JPG, JPEG, or WEBP."
            }), 400

        filename = secure_filename(file.filename)
        unique_name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{filename}"
        saved_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(saved_filepath)
        web_image_url = f"/static/uploads/{unique_name}"

    # Case B: Base64 webcam capture
    elif 'image_data' in request.form:
        image_data = request.form['image_data']
        try:
            # Handle data:image/jpeg;base64, header
            if ',' in image_data:
                header, encoded = image_data.split(',', 1)
            else:
                encoded = image_data

            decoded = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(decoded))
            unique_name = f"webcam_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
            saved_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            img.convert('RGB').save(saved_filepath, 'JPEG', quality=95)
            web_image_url = f"/static/uploads/{unique_name}"
        except Exception as e:
            return jsonify({
                "success": False,
                "error": "Invalid camera data",
                "message": f"Failed to process webcam image: {str(e)}"
            }), 400
    else:
        return jsonify({
            "success": False,
            "error": "No image payload found",
            "message": "No image was provided. Please upload a file or capture via camera."
        }), 400

    # Run Prediction
    try:
        prediction_result = predict_traffic_sign(saved_filepath)
        
        if not prediction_result.get('success', False):
            return jsonify(prediction_result), 500

        prediction_result['image_url'] = web_image_url

        # Log entry for dashboard history
        log_entry = {
            "id": uuid.uuid4().hex[:8],
            "timestamp": datetime.now().isoformat(),
            "sign_name": prediction_result.get('sign_name'),
            "category": prediction_result.get('category'),
            "confidence": prediction_result.get('confidence'),
            "recommended_action": prediction_result.get('recommended_action'),
            "image_url": web_image_url
        }
        save_history(log_entry)

        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "An error occurred during sign classification."
        }), 500

# =========================================================================
# Statistics & Telemetry APIs
# =========================================================================

@app.route('/api/stats')
def get_stats():
    """Returns dataset counts, model test accuracy, and class taxonomy."""
    metrics = {}
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                metrics = json.load(f)
        except Exception:
            pass

    labels = []
    if os.path.exists(LABELS_CSV):
        try:
            df = pd.read_csv(LABELS_CSV)
            labels = df.to_dict(orient='records')
        except Exception:
            pass

    return jsonify({
        "total_classes": len(labels) if labels else 20,
        "metrics": metrics,
        "labels": labels
    })

@app.route('/api/history')
def get_history():
    """Returns recent prediction logs."""
    return jsonify(load_history())

if __name__ == '__main__':
    # Run on port 5005 (customizable via PORT env variable)
    port = int(os.environ.get('PORT', 5005))
    app.run(host='0.0.0.0', port=port, debug=False)