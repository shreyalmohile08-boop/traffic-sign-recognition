"""
prepare_dataset.py
Prepares the 20-class Traffic Sign dataset:
1. Extracts real-world benchmark images from GTSRB for matching classes.
2. Procedurally synthesizes photorealistic training samples with photometric & geometric
   variations for non-GTSRB classes (no_horn, railway_crossing, hospital, parking).
3. Partitions images into train (70%), validation (15%), and test (15%) splits.
4. Generates a set of clean sample test images in sample_test_images/ for instant UI testing.
"""

import os
import shutil
import random
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_OUT = os.path.join(BASE_DIR, "traffic_sign_dataset")
SAMPLE_TEST_DIR = os.path.join(BASE_DIR, "sample_test_images")
GTSRB_DIR = os.path.join(BASE_DIR, "dataset")

# 20 Target Classes
CLASSES_20 = [
    "stop",
    "no_entry",
    "speed_limit_30",
    "speed_limit_50",
    "speed_limit_60",
    "speed_limit_80",
    "speed_limit_100",
    "speed_limit_120",
    "no_overtaking",
    "no_horn",
    "turn_left",
    "turn_right",
    "straight_ahead",
    "pedestrian_crossing",
    "school_ahead",
    "slippery_road",
    "railway_crossing",
    "speed_breaker",
    "hospital",
    "parking"
]

# Mapping to GTSRB Class IDs
GTSRB_MAPPING = {
    "stop": 14,
    "no_entry": 17,
    "speed_limit_30": 1,
    "speed_limit_50": 2,
    "speed_limit_60": 3,
    "speed_limit_80": 5,
    "speed_limit_100": 7,
    "speed_limit_120": 8,
    "no_overtaking": 9,
    "turn_left": 34,
    "turn_right": 33,
    "straight_ahead": 35,
    "pedestrian_crossing": 27,
    "school_ahead": 28,
    "slippery_road": 23,
    "speed_breaker": 22  # Bumpy road / speed bump
}

SYNTHETIC_CLASSES = ["no_horn", "railway_crossing", "hospital", "parking"]

def create_synthetic_base_image(class_name, size=(128, 128)):
    """Creates a clean high-resolution canonical sign graphic for synthetic classes."""
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    w, h = size
    margin = 8

    if class_name == "parking":
        # Blue rectangle/square with white 'P'
        draw.rounded_rectangle([margin, margin, w - margin, h - margin], radius=14, fill=(10, 80, 200), outline=(240, 240, 240), width=4)
        # Draw white P
        # Stem
        draw.rectangle([42, 28, 56, 100], fill=(255, 255, 255))
        # Loop
        draw.rounded_rectangle([42, 28, 88, 68], radius=14, fill=(255, 255, 255))
        draw.rounded_rectangle([56, 38, 76, 58], radius=8, fill=(10, 80, 200))

    elif class_name == "hospital":
        # Blue rectangle/square with white 'H'
        draw.rounded_rectangle([margin, margin, w - margin, h - margin], radius=14, fill=(12, 90, 210), outline=(240, 240, 240), width=4)
        # Left bar
        draw.rectangle([36, 30, 50, 98], fill=(255, 255, 255))
        # Right bar
        draw.rectangle([78, 30, 92, 98], fill=(255, 255, 255))
        # Cross bar
        draw.rectangle([50, 57, 78, 71], fill=(255, 255, 255))

    elif class_name == "no_horn":
        # White circle, thick red border, black horn symbol, red slash
        draw.ellipse([margin, margin, w - margin, h - margin], fill=(250, 250, 250), outline=(215, 25, 30), width=12)
        # Draw black horn/trumpet silhouette
        # Horn body
        draw.polygon([(36, 68), (56, 60), (74, 52), (74, 76), (56, 68)], fill=(30, 30, 30))
        # Horn bell
        draw.ellipse([68, 46, 82, 82], fill=(30, 30, 30))
        # Sound bulb
        draw.ellipse([26, 58, 38, 74], fill=(30, 30, 30))
        # Red diagonal prohibition bar
        draw.line([(28, 34), (100, 94)], fill=(215, 25, 30), width=10)

    elif class_name == "railway_crossing":
        # White warning triangle with red border, steam engine / cross tracks symbol
        # Triangle points: top, bottom-right, bottom-left
        pt1 = (w // 2, margin + 4)
        pt2 = (w - margin - 4, h - margin - 6)
        pt3 = (margin + 4, h - margin - 6)
        draw.polygon([pt1, pt2, pt3], fill=(250, 250, 250), outline=(215, 25, 30), width=12)
        # Train engine silhouette inside
        draw.rectangle([44, 70, 84, 94], fill=(35, 35, 35))
        draw.rectangle([48, 56, 66, 70], fill=(35, 35, 35))
        draw.rectangle([72, 60, 78, 70], fill=(35, 35, 35))
        # Cowcatcher / cow bars
        draw.polygon([(40, 94), (46, 94), (44, 88)], fill=(35, 35, 35))
        draw.polygon([(88, 94), (82, 94), (84, 88)], fill=(35, 35, 35))
        # Wheels
        draw.ellipse([48, 90, 60, 102], fill=(35, 35, 35))
        draw.ellipse([68, 90, 80, 102], fill=(35, 35, 35))

    # Convert to RGB numpy array
    rgb_bg = Image.new("RGB", size, (220, 225, 230))
    rgb_bg.paste(img, mask=img.split()[3])
    return np.array(rgb_bg)

def augment_synthetic_image(base_img):
    """Applies realistic photometric and geometric transformations to simulate camera feed."""
    h, w = base_img.shape[:2]
    
    # 1. Random rotation (-14 to 14 deg)
    angle = random.uniform(-14, 14)
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, random.uniform(0.88, 1.05))
    transformed = cv2.warpAffine(base_img, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 2. Random perspective warp
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    jitter = random.uniform(2, 9)
    pts2 = np.float32([
        [random.uniform(0, jitter), random.uniform(0, jitter)],
        [w - random.uniform(0, jitter), random.uniform(0, jitter)],
        [random.uniform(0, jitter), h - random.uniform(0, jitter)],
        [w - random.uniform(0, jitter), h - random.uniform(0, jitter)]
    ])
    M_persp = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(transformed, M_persp, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 3. Brightness and Contrast
    alpha = random.uniform(0.7, 1.3) # contrast
    beta = random.uniform(-25, 25)    # brightness
    transformed = cv2.convertScaleAbs(transformed, alpha=alpha, beta=beta)
    
    # 4. Occasional Gaussian Blur (simulate camera defocus/motion)
    if random.random() > 0.4:
        ksize = random.choice([3, 5])
        transformed = cv2.GaussianBlur(transformed, (ksize, ksize), 0)
        
    # 5. Add subtle noise
    noise = np.random.normal(0, random.uniform(3, 8), transformed.shape).astype(np.float32)
    transformed = np.clip(transformed.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return transformed

def prepare_data():
    print("=" * 60)
    print("Preparing 20-Class Traffic Sign Dataset...")
    print("=" * 60)

    # Re-create directory structure
    for split in ["train", "validation", "test"]:
        for c in CLASSES_20:
            os.makedirs(os.path.join(DATASET_OUT, split, c), exist_ok=True)
    os.makedirs(SAMPLE_TEST_DIR, exist_ok=True)

    manifest = {
        "dataset_type": "Hybrid Real GTSRB Benchmark + Synthetically Augmented Demo",
        "total_classes": len(CLASSES_20),
        "classes": CLASSES_20,
        "splits": {"train": {}, "validation": {}, "test": {}},
        "class_sources": {}
    }

    # 1. Process GTSRB Classes
    for class_name, gtsrb_id in GTSRB_MAPPING.items():
        src_train_folder = os.path.join(GTSRB_DIR, "Train", str(gtsrb_id))
        all_images = []
        
        if os.path.exists(src_train_folder):
            files = [f for f in os.listdir(src_train_folder) if f.lower().endswith(('.png', '.ppm', '.jpg', '.jpeg'))]
            # Shuffle deterministically
            random.shuffle(files)
            # Sample up to 250 images per class for balanced, efficient training
            selected_files = files[:250]
            for f in selected_files:
                all_images.append(os.path.join(src_train_folder, f))
                
        manifest["class_sources"][class_name] = {
            "source": f"GTSRB Real Benchmark (Class ID {gtsrb_id})",
            "total_extracted": len(all_images)
        }
        print(f"Extracted {len(all_images)} real images for '{class_name}' from GTSRB Class {gtsrb_id}.")

        # Split 70% train, 15% val, 15% test
        n = len(all_images)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_imgs = all_images[:n_train]
        val_imgs = all_images[n_train:n_train + n_val]
        test_imgs = all_images[n_train + n_val:]

        for split_name, img_list in [("train", train_imgs), ("validation", val_imgs), ("test", test_imgs)]:
            dest_dir = os.path.join(DATASET_OUT, split_name, class_name)
            for idx, img_path in enumerate(img_list):
                ext = os.path.splitext(img_path)[1]
                dest_path = os.path.join(dest_dir, f"{class_name}_{idx:04d}{ext}")
                shutil.copyfile(img_path, dest_path)
            manifest["splits"][split_name][class_name] = len(img_list)

        # Save a clean sample image in sample_test_images
        if len(test_imgs) > 0:
            sample_src = test_imgs[0]
            sample_dest = os.path.join(SAMPLE_TEST_DIR, f"{class_name}.png")
            img = cv2.imread(sample_src)
            if img is not None:
                cv2.imwrite(sample_dest, img)

    # 2. Process Synthetic / Demo Classes
    for class_name in SYNTHETIC_CLASSES:
        base_img = create_synthetic_base_image(class_name, size=(128, 128))
        total_samples = 220
        n_train = int(total_samples * 0.70)
        n_val = int(total_samples * 0.15)
        n_test = total_samples - n_train - n_val

        manifest["class_sources"][class_name] = {
            "source": "Photometrically & Geometrically Augmented Synthetic (Demo Mode)",
            "total_extracted": total_samples
        }
        print(f"Synthesizing {total_samples} augmented samples for '{class_name}' (Demo Mode)...")

        # Train
        train_dir = os.path.join(DATASET_OUT, "train", class_name)
        for i in range(n_train):
            aug = augment_synthetic_image(base_img)
            cv2.imwrite(os.path.join(train_dir, f"{class_name}_train_{i:04d}.png"), aug)
        manifest["splits"]["train"][class_name] = n_train

        # Validation
        val_dir = os.path.join(DATASET_OUT, "validation", class_name)
        for i in range(n_val):
            aug = augment_synthetic_image(base_img)
            cv2.imwrite(os.path.join(val_dir, f"{class_name}_val_{i:04d}.png"), aug)
        manifest["splits"]["validation"][class_name] = n_val

        # Test
        test_dir = os.path.join(DATASET_OUT, "test", class_name)
        for i in range(n_test):
            aug = augment_synthetic_image(base_img)
            cv2.imwrite(os.path.join(test_dir, f"{class_name}_test_{i:04d}.png"), aug)
        manifest["splits"]["test"][class_name] = n_test

        # Save sample test image
        clean_sample = cv2.resize(base_img, (96, 96))
        cv2.imwrite(os.path.join(SAMPLE_TEST_DIR, f"{class_name}.png"), clean_sample)

    # Save manifest
    manifest_path = os.path.join(DATASET_OUT, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_train = sum(manifest["splits"]["train"].values())
    total_val = sum(manifest["splits"]["validation"].values())
    total_test = sum(manifest["splits"]["test"].values())

    print("-" * 60)
    print(f"Dataset generated successfully in: {DATASET_OUT}")
    print(f"Total Train Images: {total_train}")
    print(f"Total Validation Images: {total_val}")
    print(f"Total Test Images: {total_test}")
    print(f"Grand Total: {total_train + total_val + total_test}")
    print(f"Sample images saved to: {SAMPLE_TEST_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    prepare_data()
