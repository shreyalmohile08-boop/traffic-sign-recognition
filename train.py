import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

IMG_SIZE = (64, 64)
NUM_CLASSES = 43
DATA_DIR = 'dataset'

def load_data(csv_path, double_augment=True):
    """
    Loads images listed in GTSRB CSV file.
    If double_augment is True, loads BOTH the cropped ROI image and the full uncropped image.
    This trains the model to recognize traffic signs in both tightly cropped and raw padded formats.
    """
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    print(f"Loading images from {csv_path} (Double Augmentation={double_augment})...")
    for _, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, str(row['Path']))
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Convert OpenCV BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Load cropped ROI image
        x1, y1, x2, y2 = int(row['Roi.X1']), int(row['Roi.Y1']), int(row['Roi.X2']), int(row['Roi.Y2'])
        if x2 > x1 and y2 > y1:
            img_cropped = img[y1:y2, x1:x2]
            img_cropped = cv2.resize(img_cropped, IMG_SIZE)
            images.append(img_cropped)
            labels.append(int(row['ClassId']))
            
        # 2. Also load the full uncropped image (if double_augment is enabled)
        if double_augment:
            img_full = cv2.resize(img, IMG_SIZE)
            images.append(img_full)
            labels.append(int(row['ClassId']))
        
    X = np.array(images, dtype='float32') / 255.0
    y = np.array(labels, dtype='int32')
    return X, y

def build_model(input_shape=(64, 64, 3), num_classes=43):
    """
    Builds a robust, deep CNN architecture for GTSRB Traffic Sign Recognition.
    """
    model = models.Sequential([
        # Data Augmentation Layer (runs on GPU/CPU during training)
        layers.RandomRotation(0.08, input_shape=input_shape),
        layers.RandomZoom(0.12),
        layers.RandomTranslation(0.08, 0.08),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.35),
        
        # Classification Head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    train_csv = os.path.join(DATA_DIR, 'Train.csv')
    test_csv = os.path.join(DATA_DIR, 'Test.csv')
    
    # Load Training data with double augmentation (cropped + uncropped)
    X_train_full, y_train_full = load_data(train_csv, double_augment=True)
    print(f"Loaded {len(X_train_full)} training samples (augmented).")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )
    
    model = build_model(input_shape=(64, 64, 3), num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint('traffic_sign_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    print("\nTraining CNN Model on both cropped and uncropped data...")
    # Training for 5 epochs to ensure it finishes quickly while achieving top validation metrics
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=64,
        callbacks=cb_list
    )
    
    # Evaluate on GTSRB Test Set
    if os.path.exists(test_csv):
        print("\nEvaluating on GTSRB Test Set (Cropped)...")
        X_test_crop, y_test_crop = load_data(test_csv, double_augment=False)
        crop_loss, crop_acc = model.evaluate(X_test_crop, y_test_crop, verbose=0)
        print(f"Test Accuracy on Cropped Images: {crop_acc * 100:.2f}%")
        
        print("\nEvaluating on GTSRB Test Set (Uncropped)...")
        # Load test set without cropping
        df_test = pd.read_csv(test_csv)
        images_uncrop = []
        labels_uncrop = []
        for _, row in df_test.iterrows():
            img_path = os.path.join(DATA_DIR, str(row['Path']))
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                images_uncrop.append(img)
                labels_uncrop.append(int(row['ClassId']))
        X_test_uncrop = np.array(images_uncrop, dtype='float32') / 255.0
        y_test_uncrop = np.array(labels_uncrop, dtype='int32')
        
        uncrop_loss, uncrop_acc = model.evaluate(X_test_uncrop, y_test_uncrop, verbose=0)
        print(f"Test Accuracy on Uncropped Images: {uncrop_acc * 100:.2f}%")
        
    print("\nModel saved to 'traffic_sign_model.h5'. Training pipeline execution complete.")

if __name__ == "__main__":
    main()