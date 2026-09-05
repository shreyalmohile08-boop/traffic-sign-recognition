/**
 * dashboard.js
 * TrafficSign.AI - Dashboard Analytics & Telemetry
 * Renders Chart.js graphs (Accuracy, Loss, Category & Detection Frequency),
 * updates live KPI cards, lists detection logs, and manages the sign catalog.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Chart instances
  let accChart = null;
  let lossChart = null;
  let freqChart = null;
  let catChart = null;

  // Sign Emojis
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

  // Load dashboard data
  loadDashboardData();

  const refreshBtn = document.getElementById('refresh-history-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', loadDashboardData);
  }

  async function loadDashboardData() {
    try {
      const [statsRes, historyRes] = await Promise.all([
        fetch('/api/stats').then(r => r.json()),
        fetch('/api/history').then(r => r.json())
      ]);

      updateKPIs(statsRes, historyRes);
      renderPerformanceCharts(statsRes);
      renderFrequencyAndCategoryCharts(statsRes, historyRes);
      renderHistoryTable(historyRes);
      renderCatalog(statsRes.labels || []);

    } catch (err) {
      console.error("Failed to load dashboard telemetry:", err);
    }
  }

  // 1. Update KPI Cards
  function updateKPIs(stats, history) {
    const kpiClasses = document.getElementById('kpi-classes');
    const kpiAccuracy = document.getElementById('kpi-accuracy');
    const kpiTestCount = document.getElementById('kpi-test-count');
    const kpiTotalPreds = document.getElementById('kpi-total-preds');

    if (kpiClasses) kpiClasses.textContent = (stats.total_classes || 20) + " Classes";
    if (kpiAccuracy && stats.metrics) {
      const acc = (stats.metrics.test_accuracy * 100).toFixed(2);
      kpiAccuracy.textContent = acc + "%";
    }
    if (kpiTestCount && stats.metrics) {
      kpiTestCount.textContent = (stats.metrics.test_samples || 738) + " Samples";
    }
    if (kpiTotalPreds) {
      kpiTotalPreds.textContent = history.length || 0;
    }
  }

  // 2. Render Accuracy & Loss Curves via Chart.js
  function renderPerformanceCharts(stats) {
    if (!stats.metrics || !stats.metrics.history) return;

    const hist = stats.metrics.history;
    const epochs = hist.accuracy.map((_, idx) => `Epoch ${idx + 1}`);

    // Chart styling constants
    Chart.defaults.color = '#9ca3af';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    // Accuracy Chart
    const ctxAcc = document.getElementById('accuracyChart');
    if (ctxAcc) {
      if (accChart) accChart.destroy();
      accChart = new Chart(ctxAcc, {
        type: 'line',
        data: {
          labels: epochs,
          datasets: [
            {
              label: 'Train Accuracy',
              data: hist.accuracy.map(v => (v * 100).toFixed(2)),
              borderColor: '#6366f1',
              backgroundColor: 'rgba(99, 102, 241, 0.1)',
              tension: 0.35,
              fill: true,
              borderWidth: 2.5,
              pointRadius: 4
            },
            {
              label: 'Val Accuracy',
              data: hist.val_accuracy.map(v => (v * 100).toFixed(2)),
              borderColor: '#10b981',
              backgroundColor: 'transparent',
              borderDash: [5, 5],
              tension: 0.35,
              borderWidth: 2.5,
              pointRadius: 4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top', labels: { boxWidth: 12 } },
            tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}%` } }
          },
          scales: {
            y: { min: 40, max: 100, grid: { color: 'rgba(255,255,255,0.06)' } },
            x: { grid: { color: 'rgba(255,255,255,0.04)' } }
          }
        }
      });
    }

    // Loss Chart
    const ctxLoss = document.getElementById('lossChart');
    if (ctxLoss) {
      if (lossChart) lossChart.destroy();
      lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
          labels: epochs,
          datasets: [
            {
              label: 'Train Loss',
              data: hist.loss.map(v => v.toFixed(3)),
              borderColor: '#f43f5e',
              backgroundColor: 'rgba(244, 63, 94, 0.1)',
              tension: 0.35,
              fill: true,
              borderWidth: 2.5,
              pointRadius: 4
            },
            {
              label: 'Val Loss',
              data: hist.val_loss.map(v => v.toFixed(3)),
              borderColor: '#f59e0b',
              backgroundColor: 'transparent',
              borderDash: [5, 5],
              tension: 0.35,
              borderWidth: 2.5,
              pointRadius: 4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top', labels: { boxWidth: 12 } }
          },
          scales: {
            y: { grid: { color: 'rgba(255,255,255,0.06)' } },
            x: { grid: { color: 'rgba(255,255,255,0.04)' } }
          }
        }
      });
    }
  }

  // 3. Category & Frequency Charts
  function renderFrequencyAndCategoryCharts(stats, history) {
    // Frequency calculation from history or initial baseline
    const freqCounts = {};
    if (history.length > 0) {
      history.forEach(item => {
        const sign = item.sign_name || "Unknown";
        freqCounts[sign] = (freqCounts[sign] || 0) + 1;
      });
    } else {
      // Default demo distribution
      freqCounts["Stop"] = 6;
      freqCounts["Speed Limit 50"] = 4;
      freqCounts["No Entry"] = 3;
      freqCounts["Pedestrian Crossing"] = 3;
      freqCounts["Parking"] = 2;
    }

    const freqLabels = Object.keys(freqCounts).slice(0, 6);
    const freqData = freqLabels.map(k => freqCounts[k]);

    const ctxFreq = document.getElementById('frequencyChart');
    if (ctxFreq) {
      if (freqChart) freqChart.destroy();
      freqChart = new Chart(ctxFreq, {
        type: 'bar',
        data: {
          labels: freqLabels,
          datasets: [{
            label: 'Detections',
            data: freqData,
            backgroundColor: [
              'rgba(99, 102, 241, 0.75)',
              'rgba(6, 182, 212, 0.75)',
              'rgba(16, 185, 129, 0.75)',
              'rgba(245, 158, 11, 0.75)',
              'rgba(244, 63, 94, 0.75)',
              'rgba(168, 85, 247, 0.75)'
            ],
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.06)' } },
            x: { grid: { display: false } }
          }
        }
      });
    }

    // Category distribution
    const catCounts = { "Regulatory": 7, "Prohibitory": 3, "Mandatory": 3, "Warning": 5, "Informational": 2 };
    const ctxCat = document.getElementById('categoryChart');
    if (ctxCat) {
      if (catChart) catChart.destroy();
      catChart = new Chart(ctxCat, {
        type: 'doughnut',
        data: {
          labels: Object.keys(catCounts),
          datasets: [{
            data: Object.values(catCounts),
            backgroundColor: [
              '#f43f5e', // Regulatory
              '#ef4444', // Prohibitory
              '#3b82f6', // Mandatory
              '#f59e0b', // Warning
              '#06b6d4'  // Informational
            ],
            borderWidth: 2,
            borderColor: '#111726'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'right', labels: { boxWidth: 12 } }
          }
        }
      });
    }
  }

  // 4. History Table Rendering
  function renderHistoryTable(history) {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;

    if (!history || history.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--text-dim); padding:24px;">No predictions logged yet. Run a prediction on the Recognition page!</td></tr>`;
      return;
    }

    tbody.innerHTML = '';
    // Show newest first
    const items = [...history].reverse().slice(0, 10);

    items.forEach(item => {
      const tr = document.createElement('tr');
      const timeStr = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : 'Just now';
      const catClass = (item.category || 'general').toLowerCase();
      const thumbSrc = item.image_url || '/sample_test_images/stop.png';

      tr.innerHTML = `
        <td style="color:var(--text-dim); font-size:0.85rem;">${timeStr}</td>
        <td><img src="${thumbSrc}" onerror="this.src='/sample_test_images/stop.png'" class="table-thumb" alt="Sign"></td>
        <td><strong>${item.sign_name}</strong></td>
        <td><span class="category-badge badge-${catClass}">${item.category}</span></td>
        <td style="color:var(--accent-emerald); font-weight:700;">${item.confidence}%</td>
        <td style="color:#a5b4fc; font-size:0.9rem;">${item.recommended_action}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // 5. 20-Class Library / Catalog
  let allCatalogItems = [];
  function renderCatalog(labels) {
    allCatalogItems = labels;
    const grid = document.getElementById('catalog-grid');
    if (!grid) return;

    displayFilteredCatalog('all');

    // Filter Buttons
    const filterBtns = document.querySelectorAll('#catalog-filters .filter-btn');
    filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const cat = btn.getAttribute('data-category');
        displayFilteredCatalog(cat);
      });
    });
  }

  function displayFilteredCatalog(category) {
    const grid = document.getElementById('catalog-grid');
    if (!grid) return;

    const filtered = category === 'all' 
      ? allCatalogItems 
      : allCatalogItems.filter(item => item.category.toLowerCase() === category.toLowerCase());

    grid.innerHTML = '';
    filtered.forEach(item => {
      const card = document.createElement('div');
      card.className = 'catalog-card';
      const key = item.sign_name.toLowerCase().replace(/ /g, '_');
      const emoji = SIGN_EMOJIS[key] || "🚦";

      card.innerHTML = `
        <div style="font-size:2.2rem; margin-bottom:8px;">${emoji}</div>
        <h4>${item.sign_name}</h4>
        <span class="category-badge badge-${item.category.toLowerCase()}" style="margin-bottom:8px;">${item.category}</span>
        <p style="font-size:0.8rem; color:var(--text-dim); margin-top:6px;">${item.shape} • ${item.color}</p>
        <p style="font-size:0.85rem; color:var(--text-muted); margin-top:4px;">${item.description}</p>
      `;
      grid.appendChild(card);
    });
  }

});
