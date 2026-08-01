/* Model Server Test Console Application Logic */

const API_BASE_URL = "http://localhost:8000/api/v1";

document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initPresetButtons();
  initFormSubmit();
  
  // Auto-run first prediction on page load
  runPrediction();
});

// Check Server Health
async function initHealthCheck() {
  const badge = document.getElementById("serverHealthBadge");
  const statusText = document.getElementById("serverStatusText");
  const ramText = document.getElementById("serverRamText");

  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      statusText.textContent = "Model Server Live";
      ramText.textContent = `RAM: ${data.ram_usage_mb} / ${data.ram_limit_mb} MB (LRU: ${data.lru_models_currently_in_ram.length}/${data.lru_max_models_cap})`;
    } else {
      statusText.textContent = "Server Error";
      badge.querySelector(".dot-green").style.background = "#f43f5e";
    }
  } catch (e) {
    statusText.textContent = "Server Offline (Check Port 8000)";
    badge.querySelector(".dot-green").style.background = "#f43f5e";
  }
}

let selectedRegionName = "Yangon";

// Preset Location Buttons
function initPresetButtons() {
  const buttons = document.querySelectorAll(".btn-preset");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      selectedRegionName = btn.dataset.region;
      document.getElementById("sysIndexInput").value = btn.dataset.index;
      document.getElementById("latInput").value = btn.dataset.lat;
      document.getElementById("lonInput").value = btn.dataset.lon;

      runPrediction();
    });
  });
}

// Form Submission
function initFormSubmit() {
  const form = document.getElementById("predictForm");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    runPrediction();
  });
}

// Main Prediction API Fetch Call
async function runPrediction() {
  const submitBtn = document.getElementById("submitBtn");
  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="spinner"></span> <span>Processing Model Inference...</span>`;

  const sysIndex = document.getElementById("sysIndexInput").value.trim();
  const lat = parseFloat(document.getElementById("latInput").value);
  const lon = parseFloat(document.getElementById("lonInput").value);
  const allTargets = document.getElementById("allTargetsCheck").checked;
  const fallbackToggle = document.getElementById("fallbackToggle").checked;

  const targetChecks = Array.from(document.querySelectorAll(".target-check:checked")).map(c => c.value);
  const compChecks = Array.from(document.querySelectorAll(".comp-check:checked")).map(c => c.value);

  const payload = {
    region_name: selectedRegionName || undefined,
    system_index: sysIndex || undefined,
    lat: isNaN(lat) ? undefined : lat,
    lon: isNaN(lon) ? undefined : lon,
    include_all_targets: allTargets,
    targets: allTargets ? undefined : targetChecks,
    composite_features: compChecks,
    use_fallback_models: fallbackToggle
  };

  try {
    const startT = performance.now();
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const elapsed = Math.round(performance.now() - startT);
    const data = await res.json();

    if (res.ok && data.status === "success") {
      renderResults(data, elapsed);
    } else {
      document.getElementById("jsonViewer").textContent = JSON.stringify(data, null, 2);
    }
  } catch (err) {
    document.getElementById("jsonViewer").textContent = `Error connecting to Model Server API:\n${err.message}\nMake sure uvicorn server is running on http://localhost:8000`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `<span>🚀 Run Model Prediction</span>`;
    initHealthCheck();
  }
}

// Render Results to UI
function renderResults(data, clientElapsed) {
  const meta = data.execution_metadata || {};
  const comp = data.composite_features || {};

  // Performance Banner
  document.getElementById("respTimeVal").textContent = `${meta.response_time_ms || clientElapsed} ms`;
  document.getElementById("cacheStatusVal").textContent = meta.cached ? "⚡ CACHED" : "🔥 LIVE ML";
  
  const modelsUsed = meta.models_used || {};
  const isProto = Object.values(modelsUsed).includes("prototype");
  document.getElementById("modelSourceVal").textContent = isProto ? "Fallback Prototype" : "Primary 40 Models";
  document.getElementById("ramUsedVal").textContent = `${meta.ram_used_mb || "--"} MB`;

  // 1. Render Crop Recommender
  const rankList = document.getElementById("cropRankList");
  const recommender = comp.crop_recommender || [];
  if (recommender.length > 0) {
    rankList.innerHTML = recommender.map(c => `
      <div class="crop-rank-item" style="border-left-color: ${c.color_code};">
        <span class="crop-name">${c.crop.replace(/_/g, " ")}</span>
        <span class="crop-pill" style="background: ${c.color_code};">${c.suitability.toUpperCase()} (${c.suitability_score}%)</span>
      </div>
    `).join("");
  } else {
    rankList.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem;">No recommender data selected.</div>`;
  }

  // 2. Render Risk Alert Banner
  const alertContent = document.getElementById("riskAlertContent");
  const risk = comp.multi_hazard_risk_alert;
  if (risk) {
    alertContent.innerHTML = `
      <div class="risk-banner" style="background: ${risk.alert_color_hex}25; color: ${risk.alert_color_hex};">
        <span>${risk.overall_alert_level}</span>
        <span>Score: ${risk.max_risk_score}</span>
      </div>
      <div class="risk-bars">
        <div class="risk-row"><span>Flood Risk</span><span>${risk.flood_risk}</span></div>
        <div class="risk-row"><span>Drought Risk Score</span><span>${risk.drought_risk_score}</span></div>
        <div class="risk-row"><span>Heat Stress Risk</span><span>${risk.heat_stress_risk}</span></div>
        <div class="risk-row"><span>Water Scarcity Risk</span><span>${risk.water_scarcity_risk}</span></div>
      </div>
    `;
  }

  // 3. Render Crop Health Layer
  const health = comp.crop_health_layer;
  if (health) {
    document.getElementById("healthScoreText").textContent = `${health.health_score_pct}%`;
    document.getElementById("healthStatusText").textContent = health.health_status;
    document.getElementById("ndviText").textContent = health.ndvi_median;
    document.getElementById("healthScoreText").style.color = health.map_color_hex;
  }

  // 4. Render Economic ROI Calculator
  const roi = comp.economic_roi_calculator;
  if (roi) {
    document.getElementById("yieldText").textContent = `${roi.projected_yield_t_ha} t/ha`;
    document.getElementById("roiRatingPill").textContent = roi.roi_rating;
    document.getElementById("roiDetailsText").textContent = `Market Integration: ${roi.market_integration_index} | GDP Growth Index: ${roi.gdp_growth_index}`;
  }

  // 5. Render Raw JSON
  document.getElementById("jsonViewer").textContent = JSON.stringify(data, null, 2);
}
