# 🚦 Traffic Sign Recognition System

An intelligent, web-based Deep Learning application that classifies traffic signs in real-time from uploaded images or live camera video streams. Built using **Python**, **TensorFlow/Keras**, **OpenCV**, and **Flask**, with an ultra-modern glassmorphic UI, audio feedback, and telemetry dashboard.

---

## 📌 Project Overview & Objectives

In autonomous driving systems and Advanced Driver Assistance Systems (ADAS), automated traffic sign recognition (TSR) is critical for vehicle safety, speed compliance, and accident prevention. This system provides:

- **Instant Sign Recognition**: Real-time identification of 20 standard road traffic signs.
- **Context-Aware Driving Guidance**: Immediate display of sign category, shape, color, speed limit, full meaning, and recommended driving action.
- **Confidence Scoring & Low-Confidence Guardrails**: Displays exact prediction probability with visual meters and friendly warning prompts when confidence falls below 50%.
- **Live Camera & Drag-and-Drop Capture**: Flexible input via image upload or live webcam frames.
- **Audio Voice Announcements**: Natural voice feedback via Web Speech API announcing sign names and driving instructions.
- **Analytics & Telemetry Dashboard**: Live KPI tracking, Chart.js loss & accuracy curves, category distribution, and prediction logs.

---

## 🎯 20 Traffic Sign Classes & Taxonomy

| Sign ID | Sign Name | Category | Shape | Color | Speed Limit | Description | Recommended Action |
|:---:|:---|:---|:---|:---|:---:|:---|:---|
| **0** | Stop | Regulatory | Octagon | Red | 0 | Stop the vehicle | Stop completely |
| **1** | No Entry | Prohibitory | Circle | Red | 0 | Entry prohibited | Do not enter |
| **2** | Speed Limit 30 | Regulatory | Circle | Red/White | 30 | Maximum speed 30 km/h | Maintain speed below 30 km/h |
| **3** | Speed Limit 50 | Regulatory | Circle | Red/White | 50 | Maximum speed 50 km/h | Maintain speed below 50 km/h |
| **4** | Speed Limit 60 | Regulatory | Circle | Red/White | 60 | Maximum speed 60 km/h | Maintain speed below 60 km/h |
| **5** | Speed Limit 80 | Regulatory | Circle | Red/White | 80 | Maximum speed 80 km/h | Maintain speed below 80 km/h |
| **6** | Speed Limit 100 | Regulatory | Circle | Red/White | 100 | Maximum speed 100 km/h | Maintain speed below 100 km/h |
| **7** | Speed Limit 120 | Regulatory | Circle | Red/White | 120 | Maximum speed 120 km/h | Maintain speed below 120 km/h |
| **8** | No Overtaking | Prohibitory | Circle | Red/White | 0 | Overtaking prohibited | Do not overtake |
| **9** | No Horn | Prohibitory | Circle | Red | 0 | Horn prohibited | Do not use horn |
| **10** | Turn Left | Mandatory | Circle | Blue | 0 | Turn left | Turn left |
| **11** | Turn Right | Mandatory | Circle | Blue | 0 | Turn right | Turn right |
| **12** | Straight Ahead | Mandatory | Circle | Blue | 0 | Continue straight | Continue straight |
| **13** | Pedestrian Crossing | Warning | Triangle | Red/White | 0 | Pedestrian crossing ahead | Slow down and watch for pedestrians |
| **14** | School Ahead | Warning | Triangle | Red/White | 0 | School zone ahead | Slow down |
| **15** | Slippery Road | Warning | Triangle | Red/White | 0 | Slippery road ahead | Reduce speed |
| **16** | Railway Crossing | Warning | Triangle | Red/White | 0 | Railway crossing ahead | Slow down and check for trains |
| **17** | Speed Breaker | Warning | Triangle | Red/White | 0 | Speed breaker ahead | Slow down |
| **18** | Hospital | Informational | Rectangle | Blue | 0 | Hospital nearby | Drive carefully |
| **19** | Parking | Informational | Rectangle | Blue | 0 | Parking area | Parking available |

---

## 🧠 Deep Learning Model Architecture

The model is a deep Convolutional Neural Network (CNN) engineered for spatial feature extraction, lighting invariance, and rapid CPU/GPU inference:

```
Input Image (64x64x3 RGB, Normalized [0, 1])
   │
   ├── [Data Augmentation] RandomRotation (±8°), RandomZoom (±10%), RandomTranslation (±8%)
   │
   ├── [Conv Block 1] 2x Conv2D (32 filters, 3x3, ReLU) + BatchNorm + MaxPool (2x2) + Dropout (0.25)
   │
   ├── [Conv Block 2] 2x Conv2D (64 filters, 3x3, ReLU) + BatchNorm + MaxPool (2x2) + Dropout (0.30)
   │
   ├── [Conv Block 3] 2x Conv2D (128 filters, 3x3, ReLU) + BatchNorm + MaxPool (2x2) + Dropout (0.35)
   │
   ├── [Flatten Layer]
   │
   ├── [Dense Layer] 256 units (ReLU) + BatchNorm + Dropout (0.50)
   │
   └── [Output Layer] Dense 20 units (Softmax Activation)
```

- **Optimizer**: Adam ($\text{lr}=0.001$)
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Validation Accuracy**: **96.68%**
- **Test Accuracy**: **95.66%**

---

## 📁 Repository Directory Structure

```
traffic_sign_recogination/
├── app.py                      # Flask backend application with all routes and REST APIs
├── train_model.py              # CNN model training script with callbacks & curve generation
├── predict.py                  # Standalone CLI and importable prediction module
├── prepare_dataset.py          # Dataset partitioning and benchmark extraction script
├── requirements.txt            # Python package dependencies
├── traffic_sign_labels.csv     # Metadata CSV for 20 classes
├── class_names.json            # Ordered JSON list of 20 class labels
├── traffic_sign_model.h5       # Trained 20-class CNN Keras model
├── model_metrics.json          # Saved training history and evaluation metrics
├── README.md                   # Complete documentation
│
├── templates/
│   ├── index.html              # Main recognition web interface
│   ├── dashboard.html          # Performance telemetry and analytics dashboard
│   └── about.html              # System architecture, pipeline & reference table
│
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphism dark-mode responsive stylesheet
│   ├── js/
│   │   ├── script.js           # Main UI logic (drag & drop, webcam, speech API)
│   │   └── dashboard.js        # Dashboard Chart.js graphs, telemetry, and catalog
│   ├── img/
│   │   ├── training_metrics.png# Matplotlib loss & accuracy curves
│   │   └── samples/            # 20 benchmark sample test images for quick demo
│   └── uploads/                # Directory for user-uploaded images
│
├── sample_test_images/         # Ready-to-use sample images for each of the 20 classes
│   ├── stop.png
│   ├── speed_limit_50.png
│   ├── pedestrian_crossing.png
│   └── ... (20 images)
│
└── traffic_sign_dataset/
    ├── train/                  # 20 class folders (3,409 images)
    ├── validation/             # 20 class folders (723 images)
    ├── test/                   # 20 class folders (738 images)
    └── dataset_manifest.json   # Dataset provenance and count breakdown
```

---

## 🚀 Installation & Quick Start Guide

### Step 1: Clone or Navigate to the Workspace
Open terminal or PowerShell in the project directory:
```bash
cd path/to/traffic_sign_recogination
```

### Step 2: Install Required Dependencies
Ensure you have Python 3.10+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Prepare Dataset
The dataset is already pre-partitioned into `traffic_sign_dataset/`. To re-generate or refresh it:
```bash
python prepare_dataset.py
```

### Step 4: Train the Machine Learning Model
To train the CNN model from scratch and generate performance plots:
```bash
python train_model.py
```
This will train the CNN on the training set, validate against the validation set, evaluate on unseen test images, and output:
- `traffic_sign_model.h5`
- `model_metrics.json`
- `static/img/training_metrics.png`

### Step 5: Test Inference via Command Line
You can test any image directly from the terminal using `predict.py`:
```bash
python predict.py sample_test_images/stop.png
```

### Step 6: Launch the Web Application
Start the Flask development server:
```bash
python app.py
```
The application will launch at:
👉 **`http://localhost:5000`**

---

## 🌐 Web Interface Walkthrough

### 1. Main Recognition Page (`/`)
- **Quick-Test Strip**: Click any of the sample signs (Stop, Speed Limit 50, No Entry, Parking, etc.) for instant 1-click classification.
- **Dual Input Modes**: Drag-and-drop any road sign image or switch to the Live Camera tab to capture video frames directly.
- **Scanner Animation**: Sleek holographic scanning beam while analyzing image features.
- **Comprehensive Results Display**:
  - Sign Name & Category Badge (Regulatory, Warning, Prohibitory, Mandatory, Informational)
  - Confidence Percentage with color-coded gauge
  - Recommended Action Callout: High-visibility driving guidance
  - Sign Shape, Color, Speed Limit, and Meaning
  - Top-5 Alternative Probabilities distribution bars
  - **Voice Announcement**: Click "Announce Sign (Voice)" to hear speech synthesis.
  - **Low-Confidence Safeguard**: Flags confidence below 50% with:
    *"Unable to confidently recognize this traffic sign. Please upload a clearer image."*

### 2. Analytics Dashboard (`/dashboard`)
- **4 Real-Time KPI Cards**: Total Sign Classes (20), Model Accuracy (95.66%), Test Image Count (738), and Inferences Logged.
- **Interactive Accuracy & Loss Curves**: Visualizing training progression across all 10 epochs using Chart.js.
- **Category & Frequency Distributions**: Visual breakdown of detection frequency and class categories.
- **Recent Inferences Table**: Searchable history with image thumbnails, timestamps, and confidence ratings.
- **Interactive 20-Class Library**: Filterable reference catalog with category filters.

### 3. About & Architecture Page (`/about`)
- In-depth review of the convolutional pipeline, feature extraction, and classification head.
- Technical stack breakdown.
- Complete 20-class taxonomy reference table.

---

## 📡 REST API Documentation

### Classify Traffic Sign
- **Endpoint**: `POST /predict`
- **Body**: 
  - `file`: Multipart image file (`.png`, `.jpg`, `.jpeg`, `.webp`), **OR**
  - `image_data`: Base64 data URL string from webcam snapshot.
- **Response**:
```json
{
  "success": true,
  "class_id": 0,
  "class_key": "stop",
  "sign_name": "Stop",
  "category": "Regulatory",
  "shape": "Octagon",
  "color": "Red",
  "speed_limit": 0,
  "meaning": "Stop the vehicle",
  "recommended_action": "Stop completely",
  "confidence": 99.99,
  "is_low_confidence": false,
  "top_predictions": [
    { "class_id": 0, "class_key": "stop", "sign_name": "Stop", "category": "Regulatory", "confidence": 99.99 },
    { "class_id": 5, "class_key": "speed_limit_80", "sign_name": "Speed Limit 80", "category": "Regulatory", "confidence": 0.00 }
  ],
  "image_url": "/static/uploads/1715000000_abc123_stop.png"
}
```

### System Telemetry Stats
- **Endpoint**: `GET /api/stats`
- **Response**: Returns total classes count, model test metrics, and class taxonomy array.

### Detection History
- **Endpoint**: `GET /api/history`
- **Response**: Returns list of recent predictions with timestamps and thumbnail URLs.

---

## 🏆 College Hackathon Demonstration Highlights

- **Aesthetics**: Polished glassmorphism with dark-mode color harmony (deep indigo, cyan, emerald), subtle glows, and responsive typography.
- **Complete End-to-End**: Data extraction, augmentation, CNN model training, REST API, webcam capture, and Chart.js telemetry.
- **Real-World Impact**: Directly maps detected signs to driver actions (e.g. "Maintain speed below 50 km/h", "Stop completely before proceeding").
- **Accessibility**: Includes Web Speech API voice announcements for hands-free driver assistance.

---

## 📜 License & Acknowledgments
Built for educational, academic, and hackathon presentation purposes. Real-world benchmark training samples sourced from the **German Traffic Sign Recognition Benchmark (GTSRB)** and augmented with standard Vienna Convention road signage.
