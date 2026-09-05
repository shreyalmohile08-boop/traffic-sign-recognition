"""
predict.py
Traffic Sign Recognition - Inference Module
Supports both module import and direct command line execution:
    python predict.py <path_to_image>
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_KERAS_PATH = BASE_DIR / 'traffic_sign_model.keras'
MODEL_H5_PATH = BASE_DIR / 'traffic_sign_model.h5'
CLASS_NAMES_PATH = BASE_DIR / 'class_names.json'
LABELS_CSV_PATH = BASE_DIR / 'traffic_sign_labels.csv'

IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 50.0  # Percentage threshold for low-confidence flag

# Global cache for model and metadata
_MODEL = None
_CLASS_NAMES = None
_LABELS_DF = None

def _load_keras_model(model_path):
    """
    Loads Keras model with compile=False and backward/forward compatibility fallback.
    """
    try:
        # Preferred modern loader with compile=False
        return tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as primary_error:
        error_str = str(primary_error)
        # Check if the error is related to InputLayer batch_shape / batch_input_shape deserialization
        if 'batch_shape' in error_str or 'InputLayer' in error_str:
            try:
                from tensorflow.keras.layers import InputLayer as _OrigInputLayer
                class PatchedInputLayer(_OrigInputLayer):
                    def __init__(self, *args, **kwargs):
                        if 'batch_shape' in kwargs and 'batch_input_shape' not in kwargs:
                            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                        super().__init__(*args, **kwargs)
                return tf.keras.models.load_model(
                    str(model_path), 
                    compile=False, 
                    custom_objects={'InputLayer': PatchedInputLayer}
                )
            except Exception:
                pass
        raise primary_error

def load_resources():
    """Safely loads model, class names, and metadata CSV into memory."""
    global _MODEL, _CLASS_NAMES, _LABELS_DF
    
    # 1. Load Class Names
    if _CLASS_NAMES is None:
        if CLASS_NAMES_PATH.exists():
            with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
                _CLASS_NAMES = json.load(f)
        else:
            raise FileNotFoundError(f"Class names file missing at: {CLASS_NAMES_PATH}")

    # 2. Load Metadata CSV
    if _LABELS_DF is None:
        if LABELS_CSV_PATH.exists():
            _LABELS_DF = pd.read_csv(LABELS_CSV_PATH)
        else:
            raise FileNotFoundError(f"Labels CSV file missing at: {LABELS_CSV_PATH}")

    # 3. Load Trained Model (try modern .keras first, then fallback to .h5)
    if _MODEL is None:
        model_path = None
        if MODEL_KERAS_PATH.exists():
            model_path = MODEL_KERAS_PATH
        elif MODEL_H5_PATH.exists():
            model_path = MODEL_H5_PATH
        else:
            raise FileNotFoundError(
                f"Model file missing. Looked for {MODEL_KERAS_PATH} and {MODEL_H5_PATH}. "
                "Please run train_model.py first."
            )

        print(f"Loading traffic sign model from: {model_path} (compile=False)...")
        _MODEL = _load_keras_model(model_path)
        # Warm up graph execution so live inferences are instant
        try:
            _dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
            _MODEL(_dummy, training=False)
        except Exception:
            pass
        print("Model loaded and warmed up successfully.")

    return _MODEL, _CLASS_NAMES, _LABELS_DF

def preprocess_image(image_input):
    """
    Accepts a filepath, numpy array, or bytes and outputs a (1, 64, 64, 3) float32 array.
    """
    if isinstance(image_input, (str, Path)):
        path_obj = Path(image_input)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image not found at: {image_input}")
        img = cv2.imread(str(path_obj))
        if img is None:
            raise ValueError(f"Unable to read image file at: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
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

    # Run inference via direct tensor forward pass (avoids tf.data generator overhead)
    preds_tensor = model(tensor, training=False)
    predictions = np.array(preds_tensor[0])
    best_idx = int(np.argmax(predictions))
    best_confidence = float(predictions[best_idx] * 100.0)
    best_class_key = class_names[best_idx]

    # Retrieve metadata from traffic_sign_labels.csv
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
        sample_img = BASE_DIR / 'sample_test_images' / 'stop.png'
        if sample_img.exists():
            print(f"No image argument passed. Testing on default sample: {sample_img}")
            res = predict_traffic_sign(str(sample_img))
            print(json.dumps(res, indent=2))
        else:
            print("Usage: python predict.py <path_to_image>")
        return

    image_path = sys.argv[1]
    res = predict_traffic_sign(image_path)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()