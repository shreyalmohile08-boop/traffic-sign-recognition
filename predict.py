"""
predict.py
Traffic Sign Recognition - Inference Module
Supports both module import and direct command line execution:
    python predict.py <path_to_image>
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'traffic_sign_model.h5')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names.json')
LABELS_CSV_PATH = os.path.join(BASE_DIR, 'traffic_sign_labels.csv')

IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 50.0  # Percentage threshold for low-confidence flag

# Global cache for model and metadata
_MODEL = None
_CLASS_NAMES = None
_LABELS_DF = None

def load_resources():
    """Safely loads model, class names, and metadata CSV into memory."""
    global _MODEL, _CLASS_NAMES, _LABELS_DF
    
    # 1. Load Class Names
    if _CLASS_NAMES is None:
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r') as f:
                _CLASS_NAMES = json.load(f)
        else:
            raise FileNotFoundError(f"Class names file missing at: {CLASS_NAMES_PATH}")

    # 2. Load Metadata CSV
    if _LABELS_DF is None:
        if os.path.exists(LABELS_CSV_PATH):
            _LABELS_DF = pd.read_csv(LABELS_CSV_PATH)
        else:
            raise FileNotFoundError(f"Labels CSV file missing at: {LABELS_CSV_PATH}")

    # 3. Load Trained Model
    if _MODEL is None:
        if os.path.exists(MODEL_PATH):
            _MODEL = load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file missing at: {MODEL_PATH}. Please run train_model.py first.")

    return _MODEL, _CLASS_NAMES, _LABELS_DF

def preprocess_image(image_input):
    """
    Accepts a filepath, numpy array, or bytes and outputs a (1, 64, 64, 3) float32 array.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Unable to read image file at: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
        # If received in BGR or grayscale, convert
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        # Note: if it's already RGB (like from PIL or Gradio/webcam), caller should ensure RGB.
    else:
        raise TypeError("Unsupported image input type. Provide filepath or numpy array.")

    resized = cv2.resize(img, IMG_SIZE)
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)

def predict_traffic_sign(image_input):
    """
    Predicts traffic sign from image input.
    Returns a comprehensive dict containing metadata, confidence, and low-confidence handling.
    """
    try:
        model, class_names, labels_df = load_resources()
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Model or configuration could not be loaded."
        }

    try:
        tensor = preprocess_image(image_input)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to preprocess the provided image."
        }

    # Run inference
    predictions = model.predict(tensor, verbose=0)[0]
    best_idx = int(np.argmax(predictions))
    best_confidence = float(predictions[best_idx] * 100.0)
    best_class_key = class_names[best_idx]

    # Retrieve metadata from traffic_sign_labels.csv
    # Matching either by sign_id (best_idx) or sign_name
    row = None
    if best_idx < len(labels_df):
        row = labels_df.iloc[best_idx]

    if row is not None:
        sign_name = str(row.get('sign_name', best_class_key.replace('_', ' ').title()))
        category = str(row.get('category', 'General'))
        shape = str(row.get('shape', 'Standard'))
        color = str(row.get('color', 'Standard'))
        speed_limit = int(row.get('speed_limit', 0))
        description = str(row.get('description', 'Traffic sign detected.'))
        recommended_action = str(row.get('recommended_action', 'Proceed with caution.'))
    else:
        sign_name = best_class_key.replace('_', ' ').title()
        category = "General"
        shape = "Standard"
        color = "Standard"
        speed_limit = 0
        description = "Traffic sign identified."
        recommended_action = "Observe local traffic rules."

    # Top-5 probability breakdown
    top_indices = np.argsort(predictions)[::-1][:5]
    top_predictions = []
    for idx in top_indices:
        c_name = class_names[idx]
        c_label = labels_df.iloc[idx]['sign_name'] if idx < len(labels_df) else c_name.replace('_', ' ').title()
        c_cat = labels_df.iloc[idx]['category'] if idx < len(labels_df) else "General"
        top_predictions.append({
            "class_id": int(idx),
            "class_key": c_name,
            "sign_name": c_label,
            "category": c_cat,
            "confidence": round(float(predictions[idx] * 100.0), 2)
        })

    is_low_confidence = best_confidence < CONFIDENCE_THRESHOLD

    result = {
        "success": True,
        "class_id": best_idx,
        "class_key": best_class_key,
        "sign_name": sign_name,
        "category": category,
        "shape": shape,
        "color": color,
        "speed_limit": speed_limit,
        "meaning": description,
        "recommended_action": recommended_action,
        "confidence": round(best_confidence, 2),
        "is_low_confidence": is_low_confidence,
        "top_predictions": top_predictions
    }

    if is_low_confidence:
        result["warning_message"] = "Unable to confidently recognize this traffic sign. Please upload a clearer image."

    return result

def main():
    if len(sys.argv) < 2:
        # If no arguments provided, test on a sample image
        sample_img = os.path.join(BASE_DIR, 'sample_test_images', 'stop.png')
        if os.path.exists(sample_img):
            print(f"No image argument passed. Testing on default sample: {sample_img}")
            res = predict_traffic_sign(sample_img)
            print(json.dumps(res, indent=2))
        else:
            print("Usage: python predict.py <path_to_image>")
        return

    image_path = sys.argv[1]
    res = predict_traffic_sign(image_path)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()