/**
 * script.js
 * TrafficSign.AI - Frontend Interaction Engine
 * Handles image upload, webcam frame capture, real-time prediction,
 * scanner animations, and speech synthesis voice output.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const tabUpload = document.getElementById('tab-upload');
  const tabWebcam = document.getElementById('tab-webcam');
  const panelUpload = document.getElementById('panel-upload');
  const panelWebcam = document.getElementById('panel-webcam');

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browse-btn');

  const webcamFeed = document.getElementById('webcam-feed');
  const webcamCanvas = document.getElementById('webcam-canvas');
  const startCamBtn = document.getElementById('start-cam-btn');
  const snapCamBtn = document.getElementById('snap-cam-btn');

  const previewContainer = document.getElementById('preview-container');
  const previewImg = document.getElementById('preview-img');
  const clearBtn = document.getElementById('clear-btn');
  const scannerBeam = document.getElementById('scanner-beam');

  const predictBtn = document.getElementById('predict-btn');
  const predictBtnText = document.getElementById('predict-btn-text');

  const resultPlaceholder = document.getElementById('result-placeholder');
  const resultDisplay = document.getElementById('result-display');

  // Result fields
  const resIcon = document.getElementById('res-icon');
  const resSignName = document.getElementById('res-sign-name');
  const resCategory = document.getElementById('res-category');
  const resConfidence = document.getElementById('res-confidence');
  const resAction = document.getElementById('res-action');
  const resShape = document.getElementById('res-shape');
  const resColor = document.getElementById('res-color');
  const resSpeed = document.getElementById('res-speed');
  const resMeaning = document.getElementById('res-meaning');
  const topPredsList = document.getElementById('top-preds-list');
  const speakBtn = document.getElementById('speak-btn');
  const lowConfAlert = document.getElementById('low-conf-alert');
  const lowConfMsg = document.getElementById('low-conf-msg');

  // State
  let currentFile = null;
  let currentBase64 = null;
  let webcamStream = null;
  let lastPrediction = null;

  // Icon mapping for 20 classes
  const SIGN_EMOJIS = {
    "stop": "🛑",
    "no_entry": "⛔",
    "speed_limit_30": "3️⃣0️⃣",
    "speed_limit_50": "5️⃣0️⃣",
    "speed_limit_60": "6️⃣0️⃣",
    "speed_limit_80": "8️⃣0️⃣",
    "speed_limit_100": "🔟0️⃣",
    "speed_limit_120": "1️⃣2️⃣0️⃣",
    "no_overtaking": "🚫🚗",
    "no_horn": "🔇",
    "turn_left": "⬅️",
    "turn_right": "➡️",
    "straight_ahead": "⬆️",
    "pedestrian_crossing": "🚶",
    "school_ahead": "🚸",
    "slippery_road": "⚠️",
    "railway_crossing": "🚂",
    "speed_breaker": "〰️",
    "hospital": "🏥",
    "parking": "🅿️"
  };

  // 1. Tab Switching
  tabUpload.addEventListener('click', () => {
    tabUpload.classList.add('active');
    tabWebcam.classList.remove('active');
    panelUpload.style.display = 'block';
    panelWebcam.classList.remove('active');
    stopWebcam();
  });

  tabWebcam.addEventListener('click', () => {
    tabWebcam.classList.add('active');
    tabUpload.classList.remove('active');
    panelUpload.style.display = 'none';
    panelWebcam.classList.add('active');
    startWebcam();
  });

  // 2. File Upload & Drag-and-Drop
  browseBtn.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('click', (e) => {
    if (e.target !== browseBtn) fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (PNG, JPG, JPEG, WEBP).');
      return;
    }
    currentFile = file;
    currentBase64 = null;
    const reader = new FileReader();
    reader.onload = (e) => {
      showPreview(e.target.result);
    };
    reader.readAsDataURL(file);
  }

  // 3. Webcam Operations
  async function startWebcam() {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
      });
      webcamFeed.srcObject = webcamStream;
      snapCamBtn.disabled = false;
      startCamBtn.disabled = true;
    } catch (err) {
      console.warn("Webcam access error:", err);
      alert("Unable to access camera. Please allow camera permissions or use image upload.");
    }
  }

  function stopWebcam() {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      webcamStream = null;
    }
    snapCamBtn.disabled = true;
    startCamBtn.disabled = false;
  }

  startCamBtn.addEventListener('click', startWebcam);

  snapCamBtn.addEventListener('click', () => {
    if (!webcamFeed.videoWidth) return;
    webcamCanvas.width = webcamFeed.videoWidth;
    webcamCanvas.height = webcamFeed.videoHeight;
    const ctx = webcamCanvas.getContext('2d');
    
    // Draw mirrored to match video display
    ctx.translate(webcamCanvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(webcamFeed, 0, 0, webcamCanvas.width, webcamCanvas.height);
    
    const dataUrl = webcamCanvas.toDataURL('image/jpeg', 0.92);
    currentBase64 = dataUrl;
    currentFile = null;
    showPreview(dataUrl);
  });

  // 4. Preview Management
  function showPreview(src) {
    previewImg.src = src;
    previewContainer.style.display = 'block';
    predictBtn.disabled = false;
    dropzone.style.display = 'none';
  }

  clearBtn.addEventListener('click', () => {
    resetInput();
  });

  function resetInput() {
    currentFile = null;
    currentBase64 = null;
    fileInput.value = '';
    previewImg.src = '';
    previewContainer.style.display = 'none';
    dropzone.style.display = 'flex';
    predictBtn.disabled = true;
    scannerBeam.style.display = 'none';
    resultDisplay.style.display = 'none';
    resultPlaceholder.style.display = 'flex';
  }

  // 5. Quick Sample Click
  document.querySelectorAll('.sample-card').forEach(card => {
    card.addEventListener('click', () => {
      const sampleFileName = card.getAttribute('data-sample');
      const sampleUrl = `/sample_test_images/${sampleFileName}`;
      
      // Fetch sample image as blob
      fetch(sampleUrl)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], sampleFileName, { type: 'image/png' });
          handleSelectedFile(file);
          // Automatically trigger prediction for instant demo wow factor!
          setTimeout(() => {
            predictTrafficSign();
          }, 300);
        })
        .catch(err => {
          console.error("Error loading sample image:", err);
          showPreview(sampleUrl);
          currentBase64 = null;
          currentFile = null;
        });
    });
  });

  // 6. Predict Execution
  predictBtn.addEventListener('click', predictTrafficSign);

  async function predictTrafficSign() {
    if (!currentFile && !currentBase64 && !previewImg.src) {
      alert("Please upload or capture an image first.");
      return;
    }

    // UI loading state
    predictBtn.disabled = true;
    predictBtnText.textContent = "Analyzing Sign...";
    scannerBeam.style.display = 'block';

    const formData = new FormData();
    if (currentFile) {
      formData.append('file', currentFile);
    } else if (currentBase64) {
      formData.append('image_data', currentBase64);
    } else if (previewImg.src) {
      // If loaded from sample URL
      try {
        const resp = await fetch(previewImg.src);
        const blob = await resp.blob();
        formData.append('file', blob, 'sample.png');
      } catch (e) {
        formData.append('image_url', previewImg.src);
      }
    }

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || data.message || "Recognition failed.");
      }

      renderPredictionResult(data);

    } catch (err) {
      console.error("Prediction Error:", err);
      alert("Error recognizing traffic sign: " + err.message);
    } finally {
      predictBtn.disabled = false;
      predictBtnText.textContent = "Recognize Sign";
      scannerBeam.style.display = 'none';
    }
  }

  // 7. Render Result
  function renderPredictionResult(data) {
    lastPrediction = data;
    resultPlaceholder.style.display = 'none';
    resultDisplay.style.display = 'block';

    // Sign Name & Icon
    resSignName.textContent = data.sign_name;
    const emoji = SIGN_EMOJIS[data.class_key] || "🚦";
    resIcon.textContent = emoji;

    // Category Badge
    resCategory.textContent = data.category;
    resCategory.className = 'category-badge badge-' + data.category.toLowerCase();

    // Confidence
    resConfidence.textContent = data.confidence + '%';
    if (data.confidence >= 85) {
      resConfidence.style.color = 'var(--accent-emerald)';
    } else if (data.confidence >= 50) {
      resConfidence.style.color = 'var(--accent-amber)';
    } else {
      resConfidence.style.color = 'var(--accent-rose)';
    }

    // Low confidence alert handling
    if (data.is_low_confidence) {
      lowConfAlert.style.display = 'flex';
      lowConfMsg.textContent = data.warning_message || "Unable to confidently recognize this traffic sign. Please upload a clearer image.";
    } else {
      lowConfAlert.style.display = 'none';
    }

    // Action & Details
    resAction.textContent = data.recommended_action;
    resShape.textContent = data.shape;
    resColor.textContent = data.color;
    resSpeed.textContent = data.speed_limit > 0 ? `${data.speed_limit} km/h` : 'None';
    resMeaning.textContent = data.meaning;

    // Top Predictions Bars
    topPredsList.innerHTML = '';
    if (data.top_predictions && data.top_predictions.length > 0) {
      data.top_predictions.forEach(item => {
        const div = document.createElement('div');
        div.className = 'pred-bar-item';
        div.innerHTML = `
          <div class="pred-bar-info">
            <span><strong>${item.sign_name}</strong> <span style="font-size:0.75rem; color:var(--text-dim);">(${item.category})</span></span>
            <span>${item.confidence}%</span>
          </div>
          <div class="pred-progress">
            <div class="pred-fill" style="width: ${Math.min(100, Math.max(2, item.confidence))}%;"></div>
          </div>
        `;
        topPredsList.appendChild(div);
      });
    }

    // Smooth scroll to result on mobile
    if (window.innerWidth < 960) {
      document.getElementById('result-card').scrollIntoView({ behavior: 'smooth' });
    }
  }

  // 8. Voice Announcement (Web Speech API)
  speakBtn.addEventListener('click', () => {
    if (!lastPrediction) return;
    if (!('speechSynthesis' in window)) {
      alert("Speech synthesis is not supported in this browser.");
      return;
    }

    window.speechSynthesis.cancel(); // Stop any pending speech
    const text = `${lastPrediction.sign_name} detected. Category: ${lastPrediction.category}. Recommended action: ${lastPrediction.recommended_action}`;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  });

});
