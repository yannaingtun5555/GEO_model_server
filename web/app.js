/* Model Server Test Console Application Logic */

const API_BASE_URL = "http://localhost:8001/api/v1";

document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initFormSubmit();
  initSampleButton();
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
      ramText.textContent = `RAM: ${data.ram_usage_mb || '--'} MB | Models: ${data.available_targets_count || 40}`;
    } else {
      statusText.textContent = "Server Degraded";
      if (badge) badge.querySelector(".dot-green").style.background = "#f59e0b";
    }
  } catch (e) {
    statusText.textContent = "Server Offline (Check Port 8001)";
    if (badge) badge.querySelector(".dot-green").style.background = "#f43f5e";
  }
}

// Sample CSV button
function initSampleButton() {
  const loadBtn = document.getElementById("loadSampleBtn");
  if (!loadBtn) return;
  loadBtn.addEventListener("click", () => {
    runPipelineWithSample();
  });
}

// Form Submission
function initFormSubmit() {
  const form = document.getElementById("pipelineForm");
  if (!form) return;
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("csvFileInput");
    if (fileInput.files && fileInput.files[0]) {
      runPipelineWithFile(fileInput.files[0]);
    } else {
      runPipelineWithSample();
    }
  });
}

// Run pipeline with selected file
async function runPipelineWithFile(file) {
  const submitBtn = document.getElementById("submitBtn");
  const jsonViewer = document.getElementById("jsonViewer");

  if (!file || file.size <= 10) {
    jsonViewer.textContent = `⚠️ Error: The selected file "${file ? file.name : ''}" is empty or too small (e.g., 1-byte placeholder file in data/test/).\n\nPlease select a populated CSV dataset (like one of the files in data/raw/ayeyawaddy, bago, magway, mandalay, sagaing, or yangon).`;
    return;
  }

  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="spinner"></span> <span>Slicing & Running Models...</span>`;

  try {
    // Read the file and slice first 5 lines (header + 4 rows) to prevent browser/network timeouts on 700MB files
    const text = await file.text();
    const lines = text.split(/\r?\n/);
    const validLines = lines.filter(line => line.trim().length > 0);
    
    if (validLines.length <= 1) {
      jsonViewer.textContent = `⚠️ Error: The file has no data rows to predict.`;
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span>🚀 Run Dataset Pipeline</span>`;
      return;
    }

    // Keep header + max 4 data rows
    const sliceCount = Math.min(validLines.length, 5);
    const slicedContent = validLines.slice(0, sliceCount).join("\n");
    const slicedBlob = new Blob([slicedContent], { type: "text/csv" });
    const slicedFile = new File([slicedBlob], file.name, { type: "text/csv" });

    const formData = new FormData();
    formData.append("file", slicedFile);

    const startT = performance.now();
    const res = await fetch(`${API_BASE_URL}/pipeline/run`, {
      method: "POST",
      body: formData
    });

    const elapsed = Math.round(performance.now() - startT);
    const data = await res.json();

    if (res.ok && data.status === "success") {
      renderResults(data, elapsed);
    } else {
      jsonViewer.textContent = JSON.stringify(data, null, 2);
    }
  } catch (err) {
    jsonViewer.textContent = `Error connecting to Model Server API:\n${err.message}\nMake sure uvicorn server is running on http://localhost:8001`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `<span>🚀 Run Dataset Pipeline</span>`;
    initHealthCheck();
  }
}

// Run pipeline with sample inline CSV
async function runPipelineWithSample() {
  const sampleCsvContent = (
    "system:index,latitude,longitude,ndvi_median_mean,soil_soc_g_kg_0_30cm,soil_ph_h2o_0_30cm,distance_to_surface_water_m\n" +
    "yangon_001,16.8661,96.1951,0.65,14.2,6.5,350.0\n" +
    "yangon_002,17.2016,95.7390,0.48,10.1,5.8,800.0\n" +
    "mandalay_001,21.9741,96.0831,0.58,12.5,7.0,120.0\n" +
    "sagaing_001,22.1151,95.1321,0.40,9.2,6.1,1500.0\n"
  );
  const blob = new Blob([sampleCsvContent], { type: "text/csv" });
  const file = new File([blob], "sample_dataset.csv", { type: "text/csv" });
  runPipelineWithFile(file);
}

// Render Results to UI
function renderResults(data, clientElapsed) {
  const meta = data.pipeline_metadata || {};
  const rows = data.rows || [];

  document.getElementById("respTimeVal").textContent = `${meta.execution_time_ms || clientElapsed} ms`;
  document.getElementById("totalRowsVal").textContent = data.total_rows || rows.length;
  document.getElementById("modelsUsedVal").textContent = meta.models_used_count || (rows.length * 40);

  if (rows.length === 0) return;

  const firstRow = rows[0];
  const comp = firstRow.composite_features || {};

  // 1. Crop Recommender
  const rankList = document.getElementById("cropRankList");
  const recommender = comp.crop_recommender || [];
  if (recommender.length > 0) {
    rankList.innerHTML = recommender.map(c => `
      <div class="crop-rank-item" style="border-left-color: ${c.color_code}; padding: 6px 10px; margin-bottom: 6px; background: rgba(255,255,255,0.03); border-radius: 4px; display: flex; justify-content: space-between;">
        <span class="crop-name" style="font-weight: 600;">${c.crop.replace(/_/g, " ")}</span>
        <span class="crop-pill" style="background: ${c.color_code}; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">${c.suitability.toUpperCase()} (${c.suitability_score}%)</span>
      </div>
    `).join("");
  }

  // 2. Risk Alerts
  const alertContent = document.getElementById("riskAlertContent");
  const risk = comp.risk_alerts;
  if (risk) {
    const scores = risk.risk_scores || {};
    alertContent.innerHTML = `
      <div style="font-size: 1.1rem; font-weight: 700; color: ${risk.overall_level === 'high' ? '#ef4444' : '#10b981'}; margin-bottom: 8px;">
        Overall Level: ${risk.overall_level.toUpperCase()}
      </div>
      <div style="font-size: 0.8rem; line-height: 1.6; color: var(--text-muted);">
        <div>Flood Risk: <strong>${scores.flood || 0.0}</strong></div>
        <div>Drought Risk: <strong>${scores.drought || 0.0}</strong></div>
        <div>Heat Stress: <strong>${scores.heat || 0.0}</strong></div>
        <div>Erosion Risk: <strong>${scores.erosion || 0.0}</strong></div>
        <div>Water Scarcity: <strong>${scores.water_scarcity || 0.0}</strong></div>
      </div>
    `;
  }

  // 3. Health Layer
  const health = comp.crop_health;
  if (health) {
    const healthContent = document.getElementById("healthLayerContent");
    healthContent.innerHTML = `
      <div style="font-size: 1.8rem; font-weight: 800; color: ${health.map_color_hex || '#10b981'};">${health.health_class}</div>
      <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;">Health Score: <strong>${health.health_score}</strong> | NDVI: <strong>${health.ndvi_median || 'N/A'}</strong></div>
    `;
  }

  // 4. Land Use Pattern
  const landUse = comp.land_use;
  if (landUse) {
    const landContent = document.getElementById("landUseContent");
    landContent.innerHTML = `
      <div style="font-size: 1.1rem; font-weight: 700; color: #3b82f6;">Risk Level: ${landUse.risk_level.toUpperCase()}</div>
      <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">
        Conversion Risk: <strong>${landUse.conversion_risk_score}</strong> | Encroachment Risk: <strong>${landUse.urban_encroachment_score}</strong>
      </div>
    `;
  }

  // 5. Raw JSON
  document.getElementById("jsonViewer").textContent = JSON.stringify(data, null, 2);
}
