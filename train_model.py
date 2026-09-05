"""
train_model.py
Trains a deep Convolutional Neural Network (CNN) for 20-Class Traffic Sign Recognition.
Saves:
- Model weights and architecture: traffic_sign_model.h5
- Training history and metrics: model_metrics.json
- Performance loss/accuracy curves: static/img/training_metrics.png
"""

import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "traffic_sign_dataset")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "traffic_sign_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
STATIC_IMG_DIR = os.path.join(BASE_DIR, "static", "img")
PLOT_PATH = os.path.join(STATIC_IMG_DIR, "training_metrics.png")

IMG_SIZE = (64, 64)
NUM_CLASSES = 20
BATCH_SIZE = 32
EPOCHS = 10

def load_split(split_name, class_names):
    """Loads and preprocesses images from a split folder."""
    split_dir = os.path.join(DATASET_DIR, split_name)
    images = []
    labels = []
    
    print(f"Loading '{split_name}' data from {split_dir}...")
    for idx, class_name in enumerate(class_names):
        class_folder = os.path.join(split_dir, class_name)
        if not os.path.exists(class_folder):
            continue
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                file_path = os.path.join(class_folder, filename)
                img = cv2.imread(file_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(idx)

    X = np.array(images, dtype='float32') / 255.0
    y = np.array(labels, dtype='int32')
    print(f"Loaded {len(X)} samples for '{split_name}' across {len(class_names)} classes.")
    return X, y

def build_cnn_model(input_shape=(64, 64, 3), num_classes=20):
    """
    Builds a robust CNN architecture with Convolution, ReLU, MaxPooling,
    Dropout, Flatten, Dense layers and Softmax output.
    """
    model = models.Sequential([
        # Data Augmentation layer inside the model
        layers.RandomRotation(0.08, input_shape=input_shape),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.08, 0.08),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.30),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.35),

        # Fully Connected Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_and_save_curves(history, output_path):
    """Plots training & validation accuracy and loss curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], 'o-', color='#6366f1', label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], 's-', color='#10b981', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], 'o-', color='#f43f5e', label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], 's-', color='#f59e0b', label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved training curve plot to {output_path}")

def main():
    print("=" * 60)
    print("TRAFFIC SIGN RECOGNITION - CNN TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load class names
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    print(f"Target classes ({len(class_names)}): {class_names}")

    # 2. Load dataset splits
    X_train, y_train = load_split("train", class_names)
    X_val, y_val = load_split("validation", class_names)
    X_test, y_test = load_split("test", class_names)

    # 3. Build model
    model = build_cnn_model(input_shape=(64, 64, 3), num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 4. Callbacks
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # 5. Train
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list
    )

    # Ensure best model is saved
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved successfully to {MODEL_SAVE_PATH}")

    # 6. Evaluate on Test Split
    print("\nEvaluating on unseen test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Generate predictions for classification report
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report Summary:")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")

    # 7. Plot and save training history
    plot_and_save_curves(history, PLOT_PATH)

    # 8. Save metrics to JSON for Dashboard
    metrics = {
        "model_architecture": "Deep Convolutional Neural Network (3 Conv Blocks + BatchNorm + Dropout + Dense)",
        "num_classes": len(class_names),
        "input_shape": [64, 64, 3],
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "total_samples": len(X_train) + len(X_val) + len(X_test),
        "epochs_completed": len(history.history['accuracy']),
        "history": {
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']],
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']]
        },
        "per_class_f1": {c: float(report[c]['f1-score']) for c in class_names if c in report}
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON to {METRICS_PATH}")
    print("=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
