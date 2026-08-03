/* Model Server Test Console Application Logic */

const API_BASE_URL = "http://localhost:8000/api/v1";

// Map globals
let myanmarMap;
let mapPointsData = [];
let mapMarkersLayer;
let currentSelectedRank = 0; // 0 = 1st crop, 1 = 2nd crop, 2 = 3rd crop

// Gorgeous crop colors for map rendering (each crop has a unique color)
const CROP_COLOR_MAP = {
  "monsoon_rice": "#38bdf8",     // Sky Blue
  "dry_season_rice": "#0284c7",  // Ocean Blue
  "maize": "#fbbf24",           // Amber/Yellow
  "sugarcane": "#22c55e",       // Bright Green
  "cassava": "#84cc16",         // Lime Green
  "durian": "#eab308",          // Gold
  "mangosteen": "#d946ef",      // Magenta
  "longan": "#a855f7",          // Purple
  "mango": "#f43f5e",           // Rose Red
  "chili": "#ef4444",           // Chili Red
  "tomato": "#f97316",          // Orange
  "black_gram": "#6366f1",      // Indigo
  "green_gram": "#818cf8",      // Light Indigo
  "pigeon_pea": "#4f46e5",      // Dark Indigo
  "groundnut": "#fb923c",      // Light Orange
  "sesame": "#94a3b8",          // Slate Grey
  "rubber": "#0d9488"           // Teal
};

document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initPresetButtons();
  initFormSubmit();
  initGeePipeline();
  initMap();
  
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
      const liveDbInfo = data.live_prediction_db_loaded
        ? `🟢 Live DB: ${(data.live_prediction_db_rows || 0).toLocaleString()} rows`
        : `🟡 Live DB: not loaded (real-time ML fallback)`;
      ramText.textContent = `RAM: ${data.ram_usage_mb} / ${data.ram_limit_mb} MB | LRU: ${(data.models_currently_in_ram || []).length}/${data.lru_max_models_cap} | ${liveDbInfo}`;
    } else {
      statusText.textContent = "Server Error";
      badge.querySelector(".dot-green").style.background = "#f43f5e";
    }
  } catch (e) {
    statusText.textContent = "Server Offline (Check Port 8000)";
    badge.querySelector(".dot-green").style.background = "#f43f5e";
  }
}

// GEE Pipeline Upload
function initGeePipeline() {
  const csvInput = document.getElementById("geeCsvInput");
  const selectBtn = document.getElementById("geeSelectBtn");
  const uploadBtn = document.getElementById("geeUploadBtn");
  const statusDiv = document.getElementById("geeUploadStatus");
  if (!selectBtn || !uploadBtn || !csvInput) return;

  selectBtn.addEventListener("click", () => csvInput.click());

  csvInput.addEventListener("change", () => {
    const file = csvInput.files[0];
    if (file) {
      selectBtn.textContent = `📁 ${file.name}`;
      uploadBtn.disabled = false;
      statusDiv.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    }
  });

  uploadBtn.addEventListener("click", async () => {
    const file = csvInput.files[0];
    if (!file) return;

    uploadBtn.disabled = true;
    uploadBtn.textContent = "⏳ Uploading...";
    statusDiv.style.color = "#a78bfa";
    statusDiv.textContent = "Uploading GEE data... this may take a moment.";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE_URL}/pipeline/update`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        statusDiv.style.color = "#10b981";
        statusDiv.textContent = `✅ ${data.message || "Pipeline started in background. Map will update when done."}`;
        // Poll health every 10s to detect when live_prediction_db_loaded becomes true
        let polls = 0;
        const pollId = setInterval(async () => {
          polls++;
          await initHealthCheck();
          const healthRes = await fetch(`${API_BASE_URL}/health`);
          const healthData = await healthRes.json();
          if (healthData.live_prediction_db_loaded || polls >= 36) {
            clearInterval(pollId);
            if (healthData.live_prediction_db_loaded) {
              statusDiv.textContent = `✅ Live DB ready! ${healthData.live_prediction_db_rows.toLocaleString()} rows loaded. Reloading map...`;
              // Reload map with new data
              const newMapRes = await fetch(`${API_BASE_URL}/map-recommendations`);
              if (newMapRes.ok) {
                mapPointsData = await newMapRes.json();
                renderMapMarkers();
              }
            } else {
              statusDiv.textContent = "⚠️ Pipeline timed out. Check server logs.";
            }
          } else {
            statusDiv.textContent = `🔄 Processing... checking (${polls * 10}s elapsed).`;
          }
        }, 10000);
      } else {
        statusDiv.style.color = "#ef4444";
        statusDiv.textContent = `❌ Error: ${data.detail || JSON.stringify(data)}`;
      }
    } catch (e) {
      statusDiv.style.color = "#ef4444";
      statusDiv.textContent = `❌ Connection error: ${e.message}`;
    } finally {
      uploadBtn.disabled = false;
      uploadBtn.textContent = "⬆️ Upload & Run";
    }
  });
}

let selectedRegionName = "Yangon";

// Preset Location Buttons
function initPresetButtons() {
  const buttons = document.querySelectorAll(".btn-preset");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      // Don't trigger if click is inside the rank selector group
      if (btn.closest("#cropRankSelector")) return;

      buttons.forEach(b => {
        if (!b.closest("#cropRankSelector")) {
          b.classList.remove("active");
        }
      });
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

// Initialize Leaflet Map
async function initMap() {
  // Center map on central Myanmar
  myanmarMap = L.map("map", {
    center: [21.0, 96.0],
    zoom: 6,
    minZoom: 5,
    maxZoom: 11
  });

  // Dark-themed tiles style
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(myanmarMap);

  mapMarkersLayer = L.layerGroup().addTo(myanmarMap);

  // Setup Rank Selector buttons
  const rankButtons = document.querySelectorAll("#cropRankSelector .btn-preset");
  rankButtons.forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      rankButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentSelectedRank = parseInt(btn.dataset.rank);
      renderMapMarkers();
    });
  });

  // Fetch the pre-computed recommendations map data
  try {
    const response = await fetch(`${API_BASE_URL}/map-recommendations`);
    if (response.ok) {
      mapPointsData = await response.json();
      buildLegend();
      renderMapMarkers();
    } else {
      console.error("Failed to load map recommendations:", response.statusText);
    }
  } catch (error) {
    console.error("Error fetching map recommendations:", error);
  }
}

// Render circles for each grid cell
function renderMapMarkers() {
  if (!mapMarkersLayer) return;
  mapMarkersLayer.clearLayers();

  mapPointsData.forEach(row => {
    const recs = row.recommendations || [];
    if (recs.length <= currentSelectedRank) return;

    // Compact format: recs[i] = [cropName, score]
    const currentRec = recs[currentSelectedRank];
    const cropName = currentRec[0];
    const color = CROP_COLOR_MAP[cropName] || "#3b82f6";

    // Draw circle marker
    const marker = L.circleMarker([row.lat, row.lon], {
      radius: 5,
      fillColor: color,
      color: "#0f172a",
      weight: 1,
      opacity: 0.85,
      fillOpacity: 0.95
    });

    // Client-side score → suitability label conversion
    function scoreToSuit(score) {
      if (score >= 85) return { label: "EXCELLENT", color: "#10b981" };
      if (score >= 70) return { label: "GOOD",      color: "#3b82f6" };
      if (score >= 40) return { label: "MODERATE",  color: "#f59e0b" };
      return             { label: "POOR",       color: "#ef4444" };
    }

    // Detailed popup content — all 17 crops ranked by suitability
    const recsHtml = recs.map((r, idx) => {
      const cropKey = r[0];
      const score   = r[1];
      const suit    = scoreToSuit(score);
      const cropColor  = CROP_COLOR_MAP[cropKey] || "#3b82f6";
      const isSelected = idx === currentSelectedRank;
      return `
        <div style="margin-top:4px; padding:3px 4px; border-radius:4px; background:${isSelected ? 'rgba(16,185,129,0.1)' : 'transparent'}; border:1px solid ${isSelected ? 'rgba(16,185,129,0.4)' : 'transparent'};">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="display:flex; align-items:center; gap:5px; font-size:0.72rem; color:${isSelected ? '#10b981' : '#e2e8f0'};">
              <span style="width:8px;height:8px;border-radius:50%;background:${cropColor};flex-shrink:0;"></span>
              <strong>${idx + 1}.</strong> ${cropKey.replace(/_/g," ")}
            </span>
            <span style="font-size:0.65rem; background:${suit.color}22; color:${suit.color}; padding:1px 5px; border-radius:3px; font-weight:600; white-space:nowrap;">${suit.label} ${score}%</span>
          </div>
          <div style="margin-top:3px; height:3px; border-radius:2px; background:#1e293b;">
            <div style="height:100%;width:${Math.round(score)}%;border-radius:2px;background:${suit.color};transition:width 0.3s;"></div>
          </div>
        </div>
      `;
    }).join("");

    const popupContent = `
      <div style="max-height:340px; overflow-y:auto; min-width:240px;">
        <h4 style="margin:0 0 4px 0; font-size:0.85rem; color:#10b981; position:sticky; top:0; background:#1e293b; padding:4px 0;">📍 ${row.index}</h4>
        <div style="font-size:0.72rem; margin-bottom:6px; color:#94a3b8;">
          Region: <strong style="color:#e2e8f0;">${row.region.toUpperCase()}</strong> &nbsp;|&nbsp;
          ${row.lat.toFixed(4)}, ${row.lon.toFixed(4)}
        </div>
        <div style="border-top:1px solid rgba(255,255,255,0.1); padding-top:6px;">
          <div style="font-size:0.72rem; color:#94a3b8; margin-bottom:4px;">🌾 All 17 Crops — Ranked by Suitability:</div>
          ${recsHtml}
        </div>
        <div style="margin-top:8px; font-size:0.68rem; color:#3b82f6; font-style:italic; text-align:center; position:sticky; bottom:0; background:#1e293b; padding:3px 0;">
          Click to run live prediction for this point
        </div>
      </div>
    `;

    marker.bindPopup(popupContent, { maxWidth: 280 });

    // Click marker handler: Auto-fill inputs and run live predictions
    marker.on("click", () => {
      document.getElementById("sysIndexInput").value = row.index;
      document.getElementById("latInput").value = row.lat.toFixed(4);
      document.getElementById("lonInput").value = row.lon.toFixed(4);
      
      // Reset region preset active markers
      document.querySelectorAll(".btn-preset").forEach(b => {
        if (!b.closest("#cropRankSelector")) {
          b.classList.remove("active");
        }
      });

      selectedRegionName = row.region;
      runPrediction();
    });

    // Hover effect
    marker.on("mouseover", function (e) {
      this.openPopup();
    });

    mapMarkersLayer.addLayer(marker);
  });
}

// Build map legend dynamically
function buildLegend() {
  const legendContainer = document.getElementById("mapLegend");
  if (!legendContainer) return;

  legendContainer.innerHTML = Object.entries(CROP_COLOR_MAP).map(([crop, color]) => `
    <div class="legend-item">
      <span class="legend-color" style="background: ${color};"></span>
      <span class="legend-text">${crop.replace(/_/g, " ")}</span>
    </div>
  `).join("");
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
  const boostToggle = document.getElementById("boostModeToggle")?.checked || false;
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
    use_fallback_models: fallbackToggle,
    boost_mode: boostToggle
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
  document.getElementById("queueWaitVal").textContent = `${meta.queue_wait_ms || 0.0} ms`;
  document.getElementById("cacheStatusVal").textContent = meta.cached ? "⚡ CACHED" : "🔥 LIVE ML";
  
  const isBoost = meta.boost_mode_active;
  document.getElementById("modelSourceVal").textContent = isBoost ? "🚀 BOOST (Preloaded)" : "🐢 LRU Capped";
  
  // Render Data Ingestion prediction source
  const isLiveDb = meta.live_db_served;
  const predSourceEl = document.getElementById("predSourceVal");
  if (predSourceEl) {
    predSourceEl.textContent = isLiveDb ? "🟢 Live DB" : "🔥 ML model";
    predSourceEl.style.color = isLiveDb ? "#10b981" : "#f43f5e";
  }
  
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
