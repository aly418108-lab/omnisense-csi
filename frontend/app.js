/**
 * OmniSense CSI v2.1 - Pure Hardware HUD Frontend Controller
 * Live ESP32 Ingestion, 2D Floorplan Localization & Interactive Target Vitals Lock-On
 * File: frontend/app.js
 */

(function () {
  "use strict";

  // Application State
  const state = {
    ws: null,
    connected: false,
    hardware: {
      connected: false,
      port: "/dev/ttyACM0",
      baudrate: 115200,
      packets_per_sec: 0.0,
      total_packets: 0
    },
    isTargetLockedForVitals: false, // Manual selection gate
    subcarrierData: [],
    disturbance: 0,
    spatial: {
      target_detected: false,
      x: 5.0,
      y: 5.0,
      velocity: 0.0,
      heading: 0.0,
      activity_state: "EMPTY_ROOM",
      trail: [],
      tx_pos: [5.0, 1.0],
      rx_pos: [5.0, 9.0]
    },
    vitals: {
      bpm: 0,
      rpm: 0,
      bpm_conf: 0,
      rpm_conf: 0,
      sqi: 0.0,
      status: "STANDBY_SELECT_TARGET",
      cardiac_wave: []
    },
    biometrics: {
      identity: "AMBIENT_SILENCE",
      confidence: 0,
      is_known: false,
      status_code: "IDLE",
      probabilities: {},
      color: "#ff0055"
    },
    enrolling: false,
    enrollBuffer: [],
    enrollDuration: 5.0,
    enrollStartTime: 0,
    profiles: []
  };

  // DOM Elements Cache
  const el = {
    // Header & Telemetry Badges
    rateCounter: document.getElementById("rate-counter"),
    telemetryRssi: document.getElementById("telemetry-rssi"),
    hwStatusBadge: document.getElementById("hw-status-badge"),
    hwStatusDot: document.getElementById("hw-status-dot"),
    hwStatusText: document.getElementById("hw-status-text"),
    connectionBadge: document.getElementById("connection-badge"),
    connectionText: document.getElementById("connection-text"),
    headerMac: document.getElementById("header-mac"),

    // Left Panel: Biometrics
    targetReticle: document.getElementById("target-reticle"),
    targetIcon: document.getElementById("target-icon"),
    targetIconWrap: document.getElementById("target-icon-wrap"),
    targetIdBox: document.getElementById("target-id-box"),
    targetStatusLabel: document.getElementById("target-status-label"),
    targetName: document.getElementById("target-name"),
    targetConfidence: document.getElementById("target-confidence"),
    enrollmentTriggerCard: document.getElementById("enrollment-trigger-card"),
    openEnrollModalBtn: document.getElementById("open-enroll-modal-btn"),
    quickEnrollBtn: document.getElementById("quick-enroll-btn"),
    profilesListContainer: document.getElementById("profiles-list-container"),
    refreshProfilesBtn: document.getElementById("refresh-profiles-btn"),

    // Center Panel: Floorplan Radar & Subcarrier Spectrogram
    tacticalMapCanvas: document.getElementById("tactical-map-canvas"),
    targetStatePill: document.getElementById("target-state-pill"),
    spatialX: document.getElementById("spatial-x"),
    spatialY: document.getElementById("spatial-y"),
    spatialVel: document.getElementById("spatial-vel"),
    spatialHdg: document.getElementById("spatial-hdg"),
    lockStatusIndicator: document.getElementById("lock-status-indicator"),
    btnCalibrateRadar: document.getElementById("btn-calibrate-radar"),
    btnUnlockVitalsTop: document.getElementById("btn-unlock-vitals-top"),

    disturbanceValue: document.getElementById("disturbance-value"),
    motionStatePill: document.getElementById("motion-state-pill"),
    disturbanceBar: document.getElementById("disturbance-bar"),
    subcarrierCanvas: document.getElementById("subcarrier-chart"),
    subcarrierSpectrumBars: document.getElementById("subcarrier-spectrum-bars"),

    // Right Panel: PCA Ballistocardiography Vitals
    vitalsLockBadge: document.getElementById("vitals-lock-badge"),
    vitalSqiVal: document.getElementById("vital-sqi-val"),
    vitalSqiBar: document.getElementById("vital-sqi-bar"),
    vitalBpm: document.getElementById("vital-bpm"),
    vitalBpmConf: document.getElementById("vital-bpm-conf"),
    vitalRpm: document.getElementById("vital-rpm"),
    vitalRpmConf: document.getElementById("vital-rpm-conf"),
    heartPulseIcon: document.getElementById("heart-pulse-icon"),
    respLungsIcon: document.getElementById("resp-lungs-icon"),
    vitalsGuidance: document.getElementById("vitals-guidance"),
    vitalsSubcarrierTag: document.getElementById("vitals-subcarrier-tag"),
    ecgCanvas: document.getElementById("ecg-canvas"),

    // Bottom Command Deck & Buttons
    deckHwPill: document.getElementById("deck-hw-pill"),
    deckHwText: document.getElementById("deck-hw-text"),
    btnReconnectHw: document.getElementById("btn-reconnect-hw"),
    btnCalibrateDeck: document.getElementById("btn-calibrate-deck"),
    btnUnlockVitalsDeck: document.getElementById("btn-unlock-vitals-deck"),
    btnEnrollDeck: document.getElementById("btn-enroll-deck"),
    statusToast: document.getElementById("status-toast"),
    statusToastMsg: document.getElementById("status-toast-msg"),

    // Live Enrollment Modal
    enrollModal: document.getElementById("enroll-modal"),
    closeEnrollModalBtn: document.getElementById("close-enroll-modal-btn"),
    enrollNameInput: document.getElementById("enroll-name-input"),
    enrollColorInput: document.getElementById("enroll-color-input"),
    enrollNotesInput: document.getElementById("enroll-notes-input"),
    startRecordingBtn: document.getElementById("start-recording-btn"),
    recordingProgressWrap: document.getElementById("recording-progress-wrap"),
    recordingCountdown: document.getElementById("recording-countdown"),
    recordingBar: document.getElementById("recording-bar"),

    // Background Canvas
    radarBgCanvas: document.getElementById("radar-bg-canvas")
  };

  // 1. Initialize Subcarrier Live Chart
  const selectedSCIndices = [8, 14, 20, 26, 32, 38];
  const maxChartPoints = 40;
  const chartLabels = Array(maxChartPoints).fill("");
  const chartColors = ["#00f3ff", "#38bdf8", "#818cf8", "#00ff88", "#fbbf24", "#f43f5e"];

  const datasets = selectedSCIndices.map((scIdx, i) => ({
    label: `SC #${scIdx}`,
    data: Array(maxChartPoints).fill(18),
    borderColor: chartColors[i % chartColors.length],
    borderWidth: 1.6,
    pointRadius: 0,
    tension: 0.35,
    fill: false
  }));

  const chartCtx = el.subcarrierCanvas.getContext("2d");
  const subcarrierChart = new Chart(chartCtx, {
    type: "line",
    data: { labels: chartLabels, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: { boxWidth: 7, boxHeight: 7, font: { size: 9, family: "JetBrains Mono" }, color: "#94a3b8" }
        },
        tooltip: { enabled: false }
      },
      scales: {
        x: { display: false },
        y: { min: 0, max: 42, grid: { color: "rgba(255, 255, 255, 0.04)" }, ticks: { color: "#64748b", font: { size: 8 } } }
      }
    }
  });

  // 2. Initialize 52 Spectrum Bars
  function initSpectrumBars() {
    el.subcarrierSpectrumBars.innerHTML = "";
    for (let i = 0; i < 52; i++) {
      const bar = document.createElement("div");
      bar.className = "flex-1 bg-cyan-500/40 rounded-t-sm transition-all duration-75 hover:bg-cyan-300";
      bar.style.height = "25%";
      bar.title = `Subcarrier #${i}`;
      el.subcarrierSpectrumBars.appendChild(bar);
    }
  }
  initSpectrumBars();

  // 3. Canvas Resizing Helper
  function resizeCanvases() {
    el.radarBgCanvas.width = window.innerWidth;
    el.radarBgCanvas.height = window.innerHeight;

    const dpr = window.devicePixelRatio || 1;
    if (el.tacticalMapCanvas) {
      const rect = el.tacticalMapCanvas.getBoundingClientRect();
      el.tacticalMapCanvas.width = rect.width * dpr;
      el.tacticalMapCanvas.height = rect.height * dpr;
    }
    if (el.ecgCanvas) {
      const rect = el.ecgCanvas.getBoundingClientRect();
      el.ecgCanvas.width = rect.width * dpr;
      el.ecgCanvas.height = rect.height * dpr;
    }
  }
  window.addEventListener("resize", resizeCanvases);
  setTimeout(resizeCanvases, 100);

  // 4. Centerpiece 2D Tactical Floorplan Radar Renderer
  const mapCtx = el.tacticalMapCanvas.getContext("2d");
  let radarPulseRadius = 0;
  let radarSweepAngle = 0;
  let sniperLockAngle = 0;

  function renderTacticalFloorplan() {
    const dpr = window.devicePixelRatio || 1;
    const w = el.tacticalMapCanvas.width;
    const h = el.tacticalMapCanvas.height;

    mapCtx.save();
    mapCtx.clearRect(0, 0, w, h);

    const pad = 24 * dpr;
    const mapW = w - pad * 2;
    const mapH = h - pad * 2;

    function toCanvasX(xm) { return pad + (xm / 10.0) * mapW; }
    function toCanvasY(ym) { return pad + (ym / 10.0) * mapH; }

    // Room background
    mapCtx.fillStyle = "#050814";
    mapCtx.fillRect(pad, pad, mapW, mapH);

    // Minor gridlines (1m)
    mapCtx.strokeStyle = "rgba(0, 243, 255, 0.05)";
    mapCtx.lineWidth = 1 * dpr;
    for (let i = 0; i <= 10; i++) {
      const x = toCanvasX(i);
      mapCtx.beginPath();
      mapCtx.moveTo(x, pad);
      mapCtx.lineTo(x, pad + mapH);
      mapCtx.stroke();

      const y = toCanvasY(i);
      mapCtx.beginPath();
      mapCtx.moveTo(pad, y);
      mapCtx.lineTo(pad + mapW, y);
      mapCtx.stroke();
    }

    // Major Quadrants (2.5m)
    mapCtx.strokeStyle = "rgba(0, 243, 255, 0.16)";
    mapCtx.lineWidth = 1.2 * dpr;
    for (let i = 0; i <= 4; i++) {
      const x = toCanvasX(i * 2.5);
      mapCtx.beginPath();
      mapCtx.moveTo(x, pad);
      mapCtx.lineTo(x, pad + mapH);
      mapCtx.stroke();

      const y = toCanvasY(i * 2.5);
      mapCtx.beginPath();
      mapCtx.moveTo(pad, y);
      mapCtx.lineTo(pad + mapW, y);
      mapCtx.stroke();
    }

    // Outer Room Border
    mapCtx.strokeStyle = "rgba(0, 243, 255, 0.35)";
    mapCtx.lineWidth = 1.5 * dpr;
    mapCtx.strokeRect(pad, pad, mapW, mapH);

    // TX and RX Node Coordinates
    const tx = state.spatial.tx_pos || [5.0, 1.0];
    const rx = state.spatial.rx_pos || [5.0, 9.0];
    const txX = toCanvasX(tx[0]);
    const txY = toCanvasY(tx[1]);
    const rxX = toCanvasX(rx[0]);
    const rxY = toCanvasY(rx[1]);

    // Dotted LoS Beam
    mapCtx.save();
    mapCtx.setLineDash([4 * dpr, 4 * dpr]);
    mapCtx.strokeStyle = "rgba(0, 243, 255, 0.25)";
    mapCtx.lineWidth = 1.2 * dpr;
    mapCtx.beginPath();
    mapCtx.moveTo(txX, txY);
    mapCtx.lineTo(rxX, rxY);
    mapCtx.stroke();
    mapCtx.restore();

    // Concentric RF Wave Pulses from Vortex TX
    radarPulseRadius += 0.8 * dpr;
    if (radarPulseRadius > 140 * dpr) radarPulseRadius = 10 * dpr;
    for (let rOffset = 0; rOffset < 140 * dpr; rOffset += 35 * dpr) {
      const curR = (radarPulseRadius + rOffset) % (140 * dpr);
      const alpha = Math.max(0, 0.35 * (1.0 - curR / (140 * dpr)));
      mapCtx.strokeStyle = `rgba(0, 255, 136, ${alpha})`;
      mapCtx.lineWidth = 1.2 * dpr;
      mapCtx.beginPath();
      mapCtx.arc(txX, txY, curR, 0, Math.PI * 2);
      mapCtx.stroke();
    }

    // Vortex TX Node
    mapCtx.fillStyle = "#00ff88";
    mapCtx.shadowColor = "#00ff88";
    mapCtx.shadowBlur = 12 * dpr;
    mapCtx.beginPath();
    mapCtx.arc(txX, txY, 5 * dpr, 0, Math.PI * 2);
    mapCtx.fill();
    mapCtx.shadowBlur = 0;
    mapCtx.fillStyle = "#a7f3d0";
    mapCtx.font = `bold ${8 * dpr}px 'JetBrains Mono', monospace`;
    mapCtx.fillText("VORTEX TX [5, 1]", txX + 8 * dpr, txY - 4 * dpr);

    // ESP32 RX Node
    mapCtx.fillStyle = "#00f3ff";
    mapCtx.shadowColor = "#00f3ff";
    mapCtx.shadowBlur = 12 * dpr;
    mapCtx.beginPath();
    mapCtx.arc(rxX, rxY, 5 * dpr, 0, Math.PI * 2);
    mapCtx.fill();
    mapCtx.shadowBlur = 0;
    mapCtx.fillStyle = "#7dd3fc";
    mapCtx.font = `bold ${8 * dpr}px 'JetBrains Mono', monospace`;
    mapCtx.fillText("ESP32 RX [5, 9]", rxX + 8 * dpr, rxY + 8 * dpr);

    // Dynamic RF Sweep Beam
    radarSweepAngle += 0.02;
    const sweepLen = Math.max(mapW, mapH) * 0.7;
    const sweepGrad = mapCtx.createRadialGradient(txX, txY, 5 * dpr, txX, txY, sweepLen);
    sweepGrad.addColorStop(0, "rgba(0, 243, 255, 0.12)");
    sweepGrad.addColorStop(1, "rgba(0, 243, 255, 0.0)");
    mapCtx.save();
    mapCtx.beginPath();
    mapCtx.moveTo(txX, txY);
    mapCtx.arc(txX, txY, sweepLen, radarSweepAngle - 0.2, radarSweepAngle);
    mapCtx.closePath();
    mapCtx.fillStyle = sweepGrad;
    mapCtx.fill();
    mapCtx.restore();

    // Render Historical Motion Ghost Trail
    const trail = state.spatial.trail || [];
    if (trail.length > 1) {
      mapCtx.save();
      for (let i = 1; i < trail.length; i++) {
        const pPrev = trail[i - 1];
        const pCurr = trail[i];
        const alpha = (i / trail.length) * 0.6;
        mapCtx.strokeStyle = `rgba(255, 0, 85, ${alpha})`;
        mapCtx.lineWidth = (1.5 + (i / trail.length) * 2) * dpr;
        mapCtx.beginPath();
        mapCtx.moveTo(toCanvasX(pPrev[0]), toCanvasY(pPrev[1]));
        mapCtx.lineTo(toCanvasX(pCurr[0]), toCanvasY(pCurr[1]));
        mapCtx.stroke();
      }
      mapCtx.restore();
    }

    // Render Tracked Human Avatar
    const targetDetected = state.spatial.target_detected;
    if (targetDetected) {
      const tgtX = toCanvasX(state.spatial.x);
      const tgtY = toCanvasY(state.spatial.y);
      const hdg = (state.spatial.heading * Math.PI) / 180.0;
      const speed = state.spatial.velocity || 0.0;

      // Color defaults to Glowing Tactical Red (#ff0055) for UNKNOWN TARGET
      let tgtColor = "#ff0055";
      if (state.biometrics.is_known && state.biometrics.color) {
        tgtColor = state.biometrics.color;
      }

      // Outer Pulsing Aura Ring
      const auraR = (14 + Math.sin(Date.now() / 150) * 2.5) * dpr;
      mapCtx.strokeStyle = tgtColor;
      mapCtx.shadowColor = tgtColor;
      mapCtx.shadowBlur = 18 * dpr;
      mapCtx.lineWidth = 2 * dpr;
      mapCtx.beginPath();
      mapCtx.arc(tgtX, tgtY, auraR, 0, Math.PI * 2);
      mapCtx.stroke();
      mapCtx.shadowBlur = 0;

      // Articulated Directional Avatar
      mapCtx.save();
      mapCtx.translate(tgtX, tgtY);
      mapCtx.rotate(hdg);

      // FOV Cone
      const coneGrad = mapCtx.createRadialGradient(0, 0, 4 * dpr, 0, 0, 36 * dpr);
      coneGrad.addColorStop(0, `${tgtColor}66`);
      coneGrad.addColorStop(1, `${tgtColor}00`);
      mapCtx.beginPath();
      mapCtx.moveTo(0, 0);
      mapCtx.arc(0, 0, 36 * dpr, -0.4, 0.4);
      mapCtx.closePath();
      mapCtx.fillStyle = coneGrad;
      mapCtx.fill();

      // Shoulders
      mapCtx.fillStyle = tgtColor;
      mapCtx.beginPath();
      mapCtx.ellipse(0, 0, 11 * dpr, 5 * dpr, 0, 0, Math.PI * 2);
      mapCtx.fill();

      // Head
      mapCtx.fillStyle = "#ffffff";
      mapCtx.beginPath();
      mapCtx.arc(0, 0, 4.5 * dpr, 0, Math.PI * 2);
      mapCtx.fill();

      // Direction pointer
      mapCtx.fillStyle = tgtColor;
      mapCtx.beginPath();
      mapCtx.moveTo(4 * dpr, -2 * dpr);
      mapCtx.lineTo(9 * dpr, 0);
      mapCtx.lineTo(4 * dpr, 2 * dpr);
      mapCtx.closePath();
      mapCtx.fill();
      mapCtx.restore();

      // INTERACTIVE TARGET LOCK-ON RETICLE
      if (state.isTargetLockedForVitals) {
        // Draw rotating double-diamond sniper lock-on reticle
        sniperLockAngle += 0.03;
        mapCtx.save();
        mapCtx.translate(tgtX, tgtY);
        mapCtx.rotate(sniperLockAngle);

        const lockSize = 24 * dpr;
        mapCtx.strokeStyle = "#00ff88";
        mapCtx.shadowColor = "#00ff88";
        mapCtx.shadowBlur = 14 * dpr;
        mapCtx.lineWidth = 1.8 * dpr;

        // Diamond 1
        mapCtx.strokeRect(-lockSize / 2, -lockSize / 2, lockSize, lockSize);
        // Diamond 2
        mapCtx.rotate(Math.PI / 4);
        mapCtx.strokeRect(-lockSize / 2, -lockSize / 2, lockSize, lockSize);
        mapCtx.restore();

        // Solid Selected Callout Tag
        const tagText = `🎯 TARGET LOCKED: ${state.biometrics.identity} | ${speed.toFixed(1)}m/s [${state.spatial.activity_state}]`;
        mapCtx.font = `bold ${9 * dpr}px 'JetBrains Mono', monospace`;
        const tagW = mapCtx.measureText(tagText).width + 14 * dpr;
        const tagH = 18 * dpr;
        const tagX = tgtX - tagW / 2;
        const tagY = tgtY - auraR - 26 * dpr;

        mapCtx.fillStyle = "rgba(10, 16, 34, 0.95)";
        mapCtx.strokeStyle = "#00ff88";
        mapCtx.lineWidth = 1.5 * dpr;
        mapCtx.fillRect(tagX, tagY, tagW, tagH);
        mapCtx.strokeRect(tagX, tagY, tagW, tagH);

        mapCtx.fillStyle = "#00ff88";
        mapCtx.fillText(tagText, tagX + 7 * dpr, tagY + 12 * dpr);

      } else {
        // Unselected Avatar - Render Floating Click Hint Tag
        const hintText = `[ 🎯 CLICK AVATAR TO LOCK VITALS ] ${state.biometrics.identity}`;
        mapCtx.font = `${9 * dpr}px 'JetBrains Mono', monospace`;
        const tagW = mapCtx.measureText(hintText).width + 12 * dpr;
        const tagH = 16 * dpr;
        const tagX = tgtX - tagW / 2;
        const tagY = tgtY - auraR - 22 * dpr;

        mapCtx.fillStyle = "rgba(10, 16, 34, 0.85)";
        mapCtx.strokeStyle = tgtColor;
        mapCtx.lineWidth = 1 * dpr;
        mapCtx.fillRect(tagX, tagY, tagW, tagH);
        mapCtx.strokeRect(tagX, tagY, tagW, tagH);

        mapCtx.fillStyle = tgtColor;
        mapCtx.fillText(hintText, tagX + 6 * dpr, tagY + 11 * dpr);
      }

    } else {
      // Empty Room
      mapCtx.fillStyle = "rgba(100, 116, 139, 0.7)";
      mapCtx.font = `${10 * dpr}px 'JetBrains Mono', monospace`;
      mapCtx.textAlign = "center";
      mapCtx.fillText("SWEEPING SECTOR... NO TARGET DETECTED", pad + mapW / 2, pad + mapH / 2);
      mapCtx.textAlign = "left";
    }

    mapCtx.restore();
    requestAnimationFrame(renderTacticalFloorplan);
  }
  requestAnimationFrame(renderTacticalFloorplan);

  // 5. Interactive Canvas Click Handler for Manual Target Selection
  el.tacticalMapCanvas.addEventListener("click", (e) => {
    const rect = el.tacticalMapCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const clickPixelX = (e.clientX - rect.left) * dpr;
    const clickPixelY = (e.clientY - rect.top) * dpr;

    const pad = 24 * dpr;
    const mapW = el.tacticalMapCanvas.width - pad * 2;
    const mapH = el.tacticalMapCanvas.height - pad * 2;

    // Convert pixel to 10m room coordinates
    const clickRoomX = ((clickPixelX - pad) / mapW) * 10.0;
    const clickRoomY = ((clickPixelY - pad) / mapH) * 10.0;

    if (!state.spatial.target_detected) {
      showToast("لا يوجد هدف مرصود حالياً في الغرفة.");
      return;
    }

    // Distance between click and detected avatar
    const avatarX = state.spatial.x;
    const avatarY = state.spatial.y;
    const distMeters = Math.hypot(clickRoomX - avatarX, clickRoomY - avatarY);

    if (distMeters <= 1.4) {
      // Target Selected! Lock on for vitals
      setVitalsLock(true);
      showToast("🎯 تم تحديد الهدف بنجاح! جاري قياس العلامات الحيوية (Target Locked for Vitals).");
    } else {
      // Clicked elsewhere on room map -> Deselect
      if (state.isTargetLockedForVitals) {
        setVitalsLock(false);
        showToast("🔓 تم إلغاء تحديد الهدف (Vitals Unlocked).");
      }
    }
  });

  function setVitalsLock(locked) {
    state.isTargetLockedForVitals = Boolean(locked);
    if (state.ws && state.connected) {
      state.ws.send(JSON.stringify({ action: "SET_VITALS_LOCK", locked: state.isTargetLockedForVitals }));
    }
    // Also post REST API
    fetch("/api/vitals/lock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ locked: state.isTargetLockedForVitals })
    }).catch(console.error);

    // Update UI Indicators
    if (state.isTargetLockedForVitals) {
      el.btnUnlockVitalsTop.classList.remove("hidden");
      el.btnUnlockVitalsDeck.classList.remove("hidden");
      el.lockStatusIndicator.innerText = "[ 🎯 VITALS LOCKED ON TARGET ]";
      el.lockStatusIndicator.className = "text-emerald-400 font-bold";
    } else {
      el.btnUnlockVitalsTop.classList.add("hidden");
      el.btnUnlockVitalsDeck.classList.add("hidden");
      el.lockStatusIndicator.innerText = "[ VITALS STANDBY ]";
      el.lockStatusIndicator.className = "text-slate-500 font-bold";
    }
  }

  el.btnUnlockVitalsTop.addEventListener("click", () => setVitalsLock(false));
  el.btnUnlockVitalsDeck.addEventListener("click", () => setVitalsLock(false));

  // 6. Medical-Grade PQRST BCG/ECG Waveform Canvas Trace
  const ecgCtx = el.ecgCanvas.getContext("2d");

  function drawECGWave(cardiacWave, isLocked, isAcquiring) {
    const dpr = window.devicePixelRatio || 1;
    const w = el.ecgCanvas.width;
    const h = el.ecgCanvas.height;
    ecgCtx.clearRect(0, 0, w, h);

    if (!state.isTargetLockedForVitals || !cardiacWave || cardiacWave.length < 5 || (!isLocked && !isAcquiring)) {
      // Flatline baseline
      ecgCtx.strokeStyle = "rgba(100, 116, 139, 0.4)";
      ecgCtx.lineWidth = 1.2 * dpr;
      ecgCtx.beginPath();
      ecgCtx.moveTo(0, h / 2);
      for (let x = 0; x < w; x += 10 * dpr) {
        ecgCtx.lineTo(x, h / 2 + (Math.random() - 0.5) * 1.5 * dpr);
      }
      ecgCtx.stroke();
      return;
    }

    const traceColor = isLocked ? "#00ff88" : "#f59e0b";

    ecgCtx.save();
    ecgCtx.strokeStyle = traceColor;
    ecgCtx.shadowColor = traceColor;
    ecgCtx.shadowBlur = 10 * dpr;
    ecgCtx.lineWidth = 2.2 * dpr;
    ecgCtx.beginPath();

    const step = w / (cardiacWave.length - 1);
    const midY = h / 2;
    const scaleY = h * 0.42;

    for (let i = 0; i < cardiacWave.length; i++) {
      const val = cardiacWave[i];
      const x = i * step;
      const y = midY - val * scaleY;
      if (i === 0) ecgCtx.moveTo(x, y);
      else ecgCtx.lineTo(x, y);
    }
    ecgCtx.stroke();

    // Phosphor leading dot
    const lastIdx = cardiacWave.length - 1;
    const leadX = lastIdx * step;
    const leadY = midY - cardiacWave[lastIdx] * scaleY;
    ecgCtx.fillStyle = "#ffffff";
    ecgCtx.shadowBlur = 14 * dpr;
    ecgCtx.beginPath();
    ecgCtx.arc(leadX, leadY, 3.5 * dpr, 0, Math.PI * 2);
    ecgCtx.fill();
    ecgCtx.restore();
  }

  // 7. Background Sweep Canvas
  const bgCtx = el.radarBgCanvas.getContext("2d");
  let radarBgAngle = 0;

  function renderRadarBg() {
    const w = el.radarBgCanvas.width;
    const h = el.radarBgCanvas.height;
    bgCtx.clearRect(0, 0, w, h);

    const cx = w * 0.5;
    const cy = h * 0.45;
    const maxR = Math.min(w, h) * 0.6;

    bgCtx.strokeStyle = "rgba(0, 243, 255, 0.05)";
    bgCtx.lineWidth = 1;
    for (let r = 80; r < maxR; r += 80) {
      bgCtx.beginPath();
      bgCtx.arc(cx, cy, r, 0, Math.PI * 2);
      bgCtx.stroke();
    }

    radarBgAngle += 0.015;
    const sweepGrad = bgCtx.createRadialGradient(cx, cy, 10, cx, cy, maxR);
    sweepGrad.addColorStop(0, "rgba(0, 243, 255, 0.12)");
    sweepGrad.addColorStop(1, "rgba(0, 243, 255, 0.0)");

    bgCtx.beginPath();
    bgCtx.moveTo(cx, cy);
    bgCtx.arc(cx, cy, maxR, radarBgAngle - 0.25, radarBgAngle);
    bgCtx.closePath();
    bgCtx.fillStyle = sweepGrad;
    bgCtx.fill();
    requestAnimationFrame(renderRadarBg);
  }
  requestAnimationFrame(renderRadarBg);

  // 8. WebSocket Telemetry Stream Client
  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/telemetry`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
      state.connected = true;
      el.connectionBadge.className = "flex items-center gap-2 px-3 py-1.5 rounded-md bg-emerald-950/50 border border-emerald-500/40 text-emerald-300";
      el.connectionText.innerText = "WS CONNECTED";
      showToast("WebSocket telemetry stream connected.");
    };

    state.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "TELEMETRY_FRAME") {
          handleTelemetryFrame(payload);
        }
      } catch (err) {
        console.error("Error parsing WS frame:", err);
      }
    };

    state.ws.onclose = () => {
      state.connected = false;
      el.connectionBadge.className = "flex items-center gap-2 px-3 py-1.5 rounded-md bg-rose-950/50 border border-rose-500/40 text-rose-300";
      el.connectionText.innerText = "WS RECONNECTING...";
      setTimeout(connectWebSocket, 1500);
    };

    state.ws.onerror = () => {
      state.ws.close();
    };
  }

  // 9. Real-Time Telemetry Dispatcher
  function handleTelemetryFrame(frame) {
    // Hardware Metrics
    if (frame.hardware) {
      state.hardware = frame.hardware;
      const pps = frame.hardware.packets_per_sec || 0;
      el.rateCounter.innerText = `${pps.toFixed(1)} PKT/S`;

      if (frame.hardware.connected) {
        el.hwStatusBadge.className = "flex items-center gap-2 px-3 py-1.5 rounded-md bg-emerald-950/60 border border-emerald-500/50 text-emerald-300";
        el.hwStatusDot.className = "w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping";
        el.hwStatusText.innerText = `ESP32 ONLINE (${frame.hardware.port})`;

        el.deckHwPill.className = "flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-950/70 border border-emerald-500/50 text-xs font-mono text-emerald-300";
        el.deckHwText.innerText = `🟢 ESP32 ONLINE (${frame.hardware.port}) | STREAMING: ${pps.toFixed(1)} PKTS/SEC | ${frame.hardware.baudrate} BAUD`;
      } else {
        el.hwStatusBadge.className = "flex items-center gap-2 px-3 py-1.5 rounded-md bg-rose-950/60 border border-rose-500/50 text-rose-300";
        el.hwStatusDot.className = "w-2.5 h-2.5 rounded-full bg-rose-500";
        el.hwStatusText.innerText = "ESP32 DISCONNECTED";

        el.deckHwPill.className = "flex items-center gap-2 px-3 py-1.5 rounded-lg bg-rose-950/70 border border-rose-500/50 text-xs font-mono text-rose-300";
        el.deckHwText.innerText = `🔴 ESP32 DISCONNECTED (${frame.hardware.port}) - CHECK USB`;
      }
    }

    if (frame.rssi !== undefined) el.telemetryRssi.innerText = `${frame.rssi} dBm`;
    if (frame.mac) el.headerMac.innerText = `MAC: ${frame.mac}`;

    // Handle Live Enrollment recording
    if (state.enrolling && frame.subcarriers) {
      state.enrollBuffer.push(frame.subcarriers);
      const elapsed = (Date.now() - state.enrollStartTime) / 1000.0;
      const progress = Math.min(100, (elapsed / state.enrollDuration) * 100);
      const remaining = Math.max(0, state.enrollDuration - elapsed).toFixed(1);

      el.recordingBar.style.width = `${progress}%`;
      el.recordingCountdown.innerText = `${remaining}s`;

      if (elapsed >= state.enrollDuration) {
        finishEnrollmentRecording();
      }
    }

    // Disturbance Index
    const dist = frame.disturbance_index || 0;
    state.disturbance = dist;
    el.disturbanceValue.innerText = `${dist.toFixed(1)}%`;
    el.disturbanceBar.style.width = `${Math.min(100, dist)}%`;

    // State Pill
    el.motionStatePill.innerText = frame.state_label || "AMBIENT";
    if (dist > 22) {
      el.motionStatePill.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-500/40";
    } else if (dist > 8) {
      el.motionStatePill.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-teal-950 text-teal-300 border border-teal-500/40";
    } else {
      el.motionStatePill.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-500/40";
    }

    // 2D Spatial Localization Updates
    if (frame.spatial) {
      state.spatial = frame.spatial;
      el.spatialX.innerText = `${frame.spatial.x.toFixed(1)}m`;
      el.spatialY.innerText = `${frame.spatial.y.toFixed(1)}m`;
      el.spatialVel.innerText = `${frame.spatial.velocity.toFixed(1)} m/s`;
      el.spatialHdg.innerText = `${frame.spatial.heading.toFixed(0)}°`;

      if (frame.spatial.target_detected) {
        el.targetStatePill.innerText = `TARGET ACQUIRED [${frame.spatial.activity_state}]`;
        el.targetStatePill.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-500/40 font-bold";
      } else {
        el.targetStatePill.innerText = "SWEEPING SECTOR (EMPTY)";
        el.targetStatePill.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-700 uppercase";
      }
    }

    // Subcarrier Chart Update
    const amps = frame.subcarriers || [];
    if (amps.length >= 52) {
      selectedSCIndices.forEach((scIdx, i) => {
        const val = amps[scIdx] !== undefined ? amps[scIdx] : 18;
        subcarrierChart.data.datasets[i].data.push(val);
        if (subcarrierChart.data.datasets[i].data.length > maxChartPoints) {
          subcarrierChart.data.datasets[i].data.shift();
        }
      });
      subcarrierChart.update("none");

      // 52-bar Spectrum
      const barChildren = el.subcarrierSpectrumBars.children;
      for (let i = 0; i < 52 && i < barChildren.length; i++) {
        const hPct = Math.min(100, Math.max(8, (amps[i] / 38.0) * 100));
        barChildren[i].style.height = `${hPct}%`;
      }
    }

    // Update Biometrics UI
    updateBiometricsUI(frame.biometrics);

    // Update Contactless Vitals UI
    updateVitalsUI(frame.vitals);
  }

  // 10. Update Biometrics UI
  function updateBiometricsUI(bio) {
    if (!bio) return;
    state.biometrics = bio;

    const idLabel = bio.identity || "AMBIENT_SILENCE";
    const conf = bio.confidence || 0;
    const statusCode = bio.status_code || "IDLE";

    el.targetReticle.classList.remove("glow-emerald", "glow-crimson", "glow-cyan");

    if (statusCode === "IDENTIFIED") {
      el.targetName.innerText = `[ ${idLabel.toUpperCase()} ]`;
      el.targetName.className = "text-lg font-black tracking-wide font-mono mt-0.5 text-emerald-400 truncate";
      el.targetStatusLabel.innerText = "AUTHORIZED BIOMETRIC MATCH";
      el.targetConfidence.innerText = `${conf.toFixed(1)}%`;
      el.targetConfidence.className = "font-bold text-emerald-400";
      el.targetIcon.className = "fa-solid fa-person-walking";
      el.targetIconWrap.className = "relative z-10 text-emerald-400 text-3xl";
      el.targetReticle.classList.add("glow-emerald");
      el.enrollmentTriggerCard.classList.add("hidden");

    } else if (statusCode === "UNKNOWN_ANOMALY" || idLabel === "UNKNOWN TARGET") {
      el.targetName.innerText = "[ UNKNOWN TARGET ]";
      el.targetName.className = "text-lg font-black tracking-wide font-mono mt-0.5 text-rose-500 animate-pulse truncate";
      el.targetStatusLabel.innerText = "شخص غير مسجل (UNENROLLED)";
      el.targetConfidence.innerText = conf > 0 ? `${conf.toFixed(1)}%` : "0.0%";
      el.targetConfidence.className = "font-bold text-rose-500";
      el.targetIcon.className = "fa-solid fa-user-secret";
      el.targetIconWrap.className = "relative z-10 text-rose-500 text-3xl";
      el.targetReticle.classList.add("glow-crimson");
      el.enrollmentTriggerCard.classList.remove("hidden");

    } else if (idLabel === "STATIONARY_TARGET") {
      el.targetName.innerText = "[ STATIONARY TARGET ]";
      el.targetName.className = "text-lg font-black tracking-wide font-mono mt-0.5 text-teal-300 truncate";
      el.targetStatusLabel.innerText = "هدف ثابت (STANDING STILL)";
      el.targetConfidence.innerText = "--";
      el.targetConfidence.className = "font-bold text-teal-400";
      el.targetIcon.className = "fa-solid fa-person";
      el.targetIconWrap.className = "relative z-10 text-teal-400 text-3xl";
      el.targetReticle.classList.add("glow-cyan");
      el.enrollmentTriggerCard.classList.add("hidden");

    } else {
      // Ambient Silence
      el.targetName.innerText = "[ AMBIENT SILENCE ]";
      el.targetName.className = "text-lg font-black tracking-wide font-mono mt-0.5 text-cyan-300 truncate";
      el.targetStatusLabel.innerText = "SCANNING WI-FI MULTIPATH...";
      el.targetConfidence.innerText = "0.0%";
      el.targetConfidence.className = "font-bold text-slate-400";
      el.targetIcon.className = "fa-solid fa-shield-halved";
      el.targetIconWrap.className = "relative z-10 text-cyan-400 text-3xl";
      el.enrollmentTriggerCard.classList.add("hidden");
    }
  }

  // 11. Update Contactless Vitals UI (Strict Manual Lock Gating)
  function updateVitalsUI(vitals) {
    if (!vitals) return;
    state.vitals = vitals;

    const bpm = vitals.bpm || 0;
    const rpm = vitals.rpm || 0;
    const bpmConf = (vitals.bpm_conf || 0) * 100;
    const rpmConf = (vitals.rpm_conf || 0) * 100;
    const sqi = vitals.sqi !== undefined ? vitals.sqi : 0.0;
    const status = vitals.status || "STANDBY_SELECT_TARGET";

    // Update SQI Meter
    el.vitalSqiVal.innerText = `${sqi.toFixed(0)}%`;
    el.vitalSqiBar.style.width = `${Math.min(100, sqi)}%`;

    if (!state.isTargetLockedForVitals || status === "STANDBY_SELECT_TARGET") {
      // GATE 1: STANDBY - NO TARGET SELECTED
      el.vitalBpm.innerText = "--";
      el.vitalBpmConf.innerText = "0% Conf";
      el.vitalRpm.innerText = "--";
      el.vitalRpmConf.innerText = "0% Conf";
      el.vitalSqiVal.innerText = "0%";
      el.vitalSqiBar.style.width = "0%";
      el.vitalsLockBadge.innerText = "STANDBY (NO TARGET SELECTED)";
      el.vitalsLockBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700";
      el.vitalsGuidance.innerText = "اضغط على مجسم الهدف في الخريطة لبدء قياس العلامات الحيوية (Click target avatar to lock vitals).";
      drawECGWave(null, false, false);
      return;
    }

    if (status === "TARGET_MOVING_HOLD_STILL") {
      // GATE 2: TARGET MOVING
      el.vitalBpm.innerText = "--";
      el.vitalBpmConf.innerText = "0% Conf";
      el.vitalRpm.innerText = "--";
      el.vitalRpmConf.innerText = "0% Conf";
      el.vitalsLockBadge.innerText = "TARGET MOVING (HOLD STILL)";
      el.vitalsLockBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-500/40 font-bold animate-pulse";
      el.vitalsGuidance.innerText = "الهدف يتحرك.. يرجى الوقوف بثبات لمدة 3 ثوان لعزل نبضات الصدر عبر الـ PCA.";
      drawECGWave(null, false, false);
      return;
    }

    if (status === "VITALS_LOCKED" && bpm > 0) {
      // LOCKED
      el.vitalBpm.innerText = bpm.toFixed(0);
      el.vitalBpmConf.innerText = `${bpmConf.toFixed(0)}% Conf`;
      el.vitalRpm.innerText = rpm.toFixed(1);
      el.vitalRpmConf.innerText = `${rpmConf.toFixed(0)}% Conf`;

      el.vitalsLockBadge.innerText = `VITALS LOCKED (${sqi.toFixed(0)}%)`;
      el.vitalsLockBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-500/40 font-bold";
      el.vitalsGuidance.innerText = "تم قفل الإشارة الحيوية بنجاح عبر خوارزمية PhaseBeat PCA.";

      // Pulse Heart Icon
      el.heartPulseIcon.style.transform = "scale(1.3)";
      setTimeout(() => { el.heartPulseIcon.style.transform = "scale(1.0)"; }, 150);

      drawECGWave(vitals.cardiac_wave, true, false);

    } else if (status === "ACQUIRING_LOCK") {
      // ACQUIRING
      el.vitalBpm.innerText = bpm > 0 ? bpm.toFixed(0) : "--";
      el.vitalBpmConf.innerText = `${bpmConf.toFixed(0)}% Conf`;
      el.vitalRpm.innerText = rpm > 0 ? rpm.toFixed(1) : "--";
      el.vitalRpmConf.innerText = `${rpmConf.toFixed(0)}% Conf`;

      el.vitalsLockBadge.innerText = `ACQUIRING LOCK (${sqi.toFixed(0)}%)`;
      el.vitalsLockBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-500/40 font-bold animate-pulse";
      el.vitalsGuidance.innerText = "جاري عزل نبضات الصدر عبر خوارزمية PhaseBeat PCA... حافظ على الثبات.";

      drawECGWave(vitals.cardiac_wave, false, true);

    } else {
      drawECGWave(null, false, false);
    }
  }

  // 12. Calibration & Reconnect Handlers
  function calibrateEmptyRoom() {
    if (state.ws && state.connected) {
      state.ws.send(JSON.stringify({ action: "CALIBRATE_BASELINE" }));
    }
    fetch("/api/calibrate", { method: "POST" })
      .then(res => res.json())
      .then(data => {
        showToast("🎯 تم تسجيل صمت الغرفة الحالية ومعايرة خط الأساس (Baseline Calibrated).");
      })
      .catch(console.error);
  }

  el.btnCalibrateRadar.addEventListener("click", calibrateEmptyRoom);
  el.btnCalibrateDeck.addEventListener("click", calibrateEmptyRoom);

  function reconnectESP32() {
    showToast("جاري إعادة الاتصال بالـ ESP32 على المنفذ /dev/ttyACM0...");
    if (state.ws && state.connected) {
      state.ws.send(JSON.stringify({ action: "RECONNECT_SERIAL", port: "/dev/ttyACM0", baudrate: 115200 }));
    }
    fetch("/api/reconnect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ port: "/dev/ttyACM0", baudrate: 115200 })
    })
      .then(res => res.json())
      .then(data => {
        if (data.connected) {
          showToast("🟢 تم الاتصال بالـ ESP32 بنجاح!");
        } else {
          showToast("🔴 تعذر الاتصال بالعتاد - تأكد من توصيل كيبل الـ USB.");
        }
      })
      .catch(console.error);
  }

  el.btnReconnectHw.addEventListener("click", reconnectESP32);

  // 13. Live Enrollment Modal
  function openEnrollModal() {
    el.enrollModal.classList.remove("hidden");
    el.enrollNameInput.focus();
  }

  el.openEnrollModalBtn.addEventListener("click", openEnrollModal);
  if (el.quickEnrollBtn) el.quickEnrollBtn.addEventListener("click", openEnrollModal);
  if (el.btnEnrollDeck) el.btnEnrollDeck.addEventListener("click", openEnrollModal);

  el.closeEnrollModalBtn.addEventListener("click", () => {
    el.enrollModal.classList.add("hidden");
    state.enrolling = false;
  });

  el.startRecordingBtn.addEventListener("click", () => {
    const name = el.enrollNameInput.value.trim();
    if (!name) {
      alert("يرجى إدخال اسم الشخص أولاً.");
      return;
    }

    state.enrolling = true;
    state.enrollBuffer = [];
    state.enrollStartTime = Date.now();
    el.recordingProgressWrap.classList.remove("hidden");
    el.startRecordingBtn.disabled = true;
    el.startRecordingBtn.classList.add("opacity-50");

    showToast(`جاري تسجيل 5 ثوانٍ من حركة المشي الحية للشخص: ${name}...`);
  });

  function finishEnrollmentRecording() {
    state.enrolling = false;
    const name = el.enrollNameInput.value.trim();
    const notes = el.enrollNotesInput.value.trim();
    const color = el.enrollColorInput ? el.enrollColorInput.value : "#00ff88";

    el.recordingProgressWrap.classList.add("hidden");
    el.startRecordingBtn.disabled = false;
    el.startRecordingBtn.classList.remove("opacity-50");
    el.enrollModal.classList.add("hidden");

    showToast(`جاري تدريب النموذج على ${state.enrollBuffer.length} إطار من الـ ESP32...`);

    fetch("/api/enroll", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: name,
        description: notes,
        color: color,
        frames: state.enrollBuffer
      })
    })
      .then(res => res.json())
      .then(data => {
        if (data.status === "ok") {
          showToast(`تم تسجيل وتدريب "${name}" في ${data.train_duration_ms}ms!`);
          fetchProfiles();
        } else {
          showToast(`فشل التدريب: ${data.detail || data.message}`);
        }
      })
      .catch(err => {
        showToast(`خطأ في التدريب: ${err}`);
      });
  }

  // 14. Biometric Profiles Registry Loader & Deletion
  function fetchProfiles() {
    fetch("/api/profiles")
      .then(res => res.json())
      .then(data => {
        state.profiles = data.profiles || [];
        renderProfilesList();
      })
      .catch(console.error);
  }

  function renderProfilesList() {
    el.profilesListContainer.innerHTML = "";
    if (state.profiles.length === 0) {
      el.profilesListContainer.innerHTML = `
        <div class="text-[11px] text-slate-500 font-mono text-center p-3">
          لا توجد بصمات مسجلة حالياً.<br>
          <span class="text-cyan-400">اضغط "تدريب وتسجيل" لتسجيل أول شخص.</span>
        </div>`;
      return;
    }

    state.profiles.forEach(p => {
      const item = document.createElement("div");
      item.className = "flex items-center justify-between p-2 rounded bg-slate-900/70 border border-slate-800 text-xs font-mono transition-all hover:border-slate-700";
      
      item.innerHTML = `
        <div class="flex items-center gap-2 truncate">
          <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" style="background-color: ${p.color || '#00f3ff'}"></span>
          <span class="text-slate-200 font-semibold truncate">${p.name}</span>
        </div>
        <div class="flex items-center gap-2 flex-shrink-0">
          <span class="text-[10px] text-slate-400">${p.sample_count || 0} عينة</span>
          <button class="profile-del-btn text-slate-400 hover:text-rose-400" data-del-id="${p.id}" title="حذف وإعادة التدريب">
            <i class="fa-solid fa-trash text-[11px]"></i>
          </button>
        </div>
      `;

      const delBtn = item.querySelector(".profile-del-btn");
      if (delBtn) {
        delBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          deleteProfile(p.id, p.name);
        });
      }

      el.profilesListContainer.appendChild(item);
    });
  }

  function deleteProfile(profileId, profileName) {
    if (!confirm(`هل أنت متأكد من حذف بصمة "${profileName}" وإعادة تدريب النموذج؟`)) return;

    fetch(`/api/profiles/${profileId}`, { method: "DELETE" })
      .then(res => res.json())
      .then(data => {
        if (data.status === "ok") {
          showToast(`تم حذف "${profileName}" وتحديث النموذج.`);
          fetchProfiles();
        } else {
          showToast(`فشل الحذف: ${data.detail || data.message}`);
        }
      })
      .catch(err => {
        showToast(`خطأ في الحذف: ${err}`);
      });
  }

  el.refreshProfilesBtn.addEventListener("click", fetchProfiles);

  // 15. Notification Toast Helper
  let toastTimeout;
  function showToast(message) {
    if (toastTimeout) clearTimeout(toastTimeout);
    el.statusToastMsg.innerText = message;
    el.statusToast.classList.remove("opacity-0");
    el.statusToast.classList.add("opacity-100");
    toastTimeout = setTimeout(() => {
      el.statusToast.classList.remove("opacity-100");
      el.statusToast.classList.add("opacity-0");
    }, 4000);
  }

  // Initialization
  connectWebSocket();
  fetchProfiles();

})();
