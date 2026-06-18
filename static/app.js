/* ── BridgeSign app.js v2 ────────────────────────────────
   Key fixes:
   - Emergency TTS uses Web Speech API (plays in browser, not server)
   - Custom phrase add/delete
   - Session timer
   - Stat strip updates
   - Lucide icons re-init on tab switch
─────────────────────────────────────────────────────── */

const PERF_CONFIG = window.BRIDGESIGN_PERF || {};
const LIVE_PERF = PERF_CONFIG.live || {};
const SPEECH_PERF = PERF_CONFIG.speech || PERF_CONFIG.voice || {};
const IMAGE_PERF = PERF_CONFIG.image || {};

let _speechVoices = [];

function refreshSpeechVoices() {
  if (!window.speechSynthesis) return [];
  _speechVoices = window.speechSynthesis.getVoices() || [];
  return _speechVoices;
}

function preferredSpeechVoice() {
  const voices = _speechVoices.length ? _speechVoices : refreshSpeechVoices();
  return voices.find(v => v.lang.startsWith("en") && v.name.includes("Female"))
      || voices.find(v => v.lang.startsWith("en"))
      || voices[0]
      || null;
}

// ── Web Speech API speak helper ────────────────────────────
function speak(text) {
  if (!text || !window.speechSynthesis) return;
  // If the string is all-caps (assembled from signed letters), convert to
  // Title Case so the TTS engine reads it as a WORD not an abbreviation.
  // e.g. "HELLO WORLD" → "Hello World"  (not "H-E-L-L-O  W-O-R-L-D")
  const normalized = text === text.toUpperCase()
    ? text.toLowerCase().replace(/\b\w/g, c => c.toUpperCase())
    : text;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(normalized);
  const rateSetting = numberOption(SPEECH_PERF.rate, 1.15);
  const pitchSetting = numberOption(SPEECH_PERF.pitch, 1.0);
  utt.rate   = rateSetting;
  utt.pitch  = pitchSetting;
  utt.volume = 1;
  const pref = preferredSpeechVoice();
  if (pref) utt.voice = pref;
  window.speechSynthesis.speak(utt);
}
// Voices load async in some browsers
if (window.speechSynthesis) {
  refreshSpeechVoices();
  window.speechSynthesis.onvoiceschanged = refreshSpeechVoices;
}

const rateSetting = numberOption(SPEECH_PERF.rate, 1.15);
const thresholdSetting = Math.max(2, numberOption(LIVE_PERF.consecutiveThreshold, 2));
console.log(`[Perf] Voice rate: ${rateSetting}, Consecutive: ${thresholdSetting}`);

// ── Tab Switching ──────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const panel = document.getElementById("tab-" + btn.dataset.tab);
    panel.classList.add("active");
    panel.querySelectorAll(".pour-in").forEach(el => {
      el.style.animation = "none";
      void el.offsetWidth;
      el.style.animation = "";
    });
    if (typeof lucide !== "undefined") lucide.createIcons();
    if (btn.dataset.tab === "emergency") loadEmergency();
    if (btn.dataset.tab === "stats")     loadStats();
    if (btn.dataset.tab === "speech")    initializeAvatar();
    if (btn.dataset.tab === "motion")    initMotionRecorder();
  });
});

// ── State ──────────────────────────────────────────────────
let cameraRunning  = false;
let lastLabel      = "";
let lastWord       = "";
let currentMode    = "letter";
let sessionStart   = null;
let timerInterval  = null;
let signCount      = 0;
// Inference loop (replaces MJPEG + status polling)
let _videoStream   = null;
let _inferInterval = null;
let _inferInFlight = false;
let _inferSeq      = 0;
let _lastAppliedInferSeq = 0;
let _permissionRetry = null;
let _inferFrameCounter = 0;
let _lastInferResponseAt = 0;
const INFER_INTERVAL_MS = Number(LIVE_PERF.inferIntervalMs || 70);
const INFER_FRAME_SKIP = Math.max(1, Number(LIVE_PERF.frameSkip || 1));
const INFER_SLOW_WARN_MS = Number(LIVE_PERF.slowWarnMs || 80);
const INFER_HARD_TIMEOUT_MS = Number(LIVE_PERF.hardTimeoutMs || 550);
const LOCAL_HANDS_TIMEOUT_MS = Number(LIVE_PERF.localHandsTimeoutMs || 700);
const LOCAL_NO_HAND_FALLBACK_FRAMES = Number(LIVE_PERF.noHandFallbackFrames || 30);
const LIVE_MEDIAPIPE_MAX_DIM = Number(LIVE_PERF.mediaPipeMaxDim || 320);
const SERVER_FRAME_MAX_DIM = Number(LIVE_PERF.serverFrameMaxDim || 320);
const SERVER_JPEG_QUALITY = Number(LIVE_PERF.jpegQuality || 0.55);
const STALE_CLEAR_MS = Number(LIVE_PERF.staleClearMs || 350);
const PERF_LOG_EVERY = Number(LIVE_PERF.logEvery || 60);
const DEBUG_INFER_LOGS = false;
const USE_CLIENT_HANDS = LIVE_PERF.useClientHands !== false;

function numberOption(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function liveConsecutiveThreshold() {
  const base = numberOption(LIVE_PERF.consecutiveThreshold, 2);
  const wordBase = numberOption(LIVE_PERF.wordConsecutiveThreshold, base);
  return Math.max(2, currentMode === "word" ? wordBase : base);
}

function scaledSize(width, height, maxDim) {
  const cap = numberOption(maxDim, 0);
  if (!cap || width <= cap && height <= cap) return { width, height };
  const scale = cap / Math.max(width, height);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale))
  };
}

function drawVideoFrame(video, canvas, maxDim) {
  const size = scaledSize(video.videoWidth, video.videoHeight, maxDim);
  canvas.width = size.width;
  canvas.height = size.height;
  canvas.getContext("2d").drawImage(video, 0, 0, size.width, size.height);
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 0) {
  const timeout = numberOption(timeoutMs, 0);
  if (!timeout) return fetch(url, options);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

let facingMode     = "user"; // "user" or "environment"
let _localHands    = null;
let useClientInference = false;
let normalizeFrontCameraLandmarks = true;
let _mediapipeFramesProcessed = 0;  // Track if MediaPipe actually works
let _mediapipeWatchdog = null;      // Timer to detect hung MediaPipe
let _localNoHandFrames = 0;
let _localHandFrames = 0;
const _inferPerf = {
  count: 0,
  clientTotal: 0,
  serverTotal: 0,
  lastMode: ""
};

function clearMediaPipeWatchdog() {
  if (_mediapipeWatchdog) {
    clearTimeout(_mediapipeWatchdog);
    _mediapipeWatchdog = null;
  }
}

function resetLocalHandsHealth() {
  _mediapipeFramesProcessed = 0;
  _localNoHandFrames = 0;
  _localHandFrames = 0;
}

function switchToServerInference(reason, toastText = "Hand tracking switched to server mode") {
  if (!useClientInference) return;
  console.warn(`[BridgeSign] ${reason} Falling back to server-side hand detection.`);
  useClientInference = false;
  clearMediaPipeWatchdog();
  _localNoHandFrames = 0;
  if (cameraRunning && toastText) showToast(toastText);
}

// ── Initialize MediaPipe Hands locally if available ──
function initLocalHands() {
  if (!USE_CLIENT_HANDS) {
    useClientInference = false;
    console.log("[BridgeSign] Client-side hand tracking disabled; using stable server detection.");
    return;
  }
  if (typeof Hands !== "undefined") {
    try {
      _localHands = new Hands({
        locateFile: (file) => `/static/lib/mediapipe/${file}`
      });
      const mediaPipeOptions = LIVE_PERF.mediaPipe || {};
      _localHands.setOptions({
        maxNumHands: numberOption(mediaPipeOptions.maxNumHands, 1),
        modelComplexity: numberOption(mediaPipeOptions.modelComplexity, 0),
        minDetectionConfidence: numberOption(mediaPipeOptions.minDetectionConfidence, 0.5),
        minTrackingConfidence: numberOption(mediaPipeOptions.minTrackingConfidence, 0.5)
      });
      _localHands.onResults(handleLocalHandsResults);
      useClientInference = true;
      console.log("[BridgeSign] Client-side landmark inference active! 🚀");
      
      // Show flip button since we have local tracking capability
      const flipBtn = document.getElementById("camFlipBtn");
      if (flipBtn) flipBtn.style.display = "inline-flex";
    } catch (err) {
      console.warn("[BridgeSign] Failed to initialize local Hands, falling back to server-side:", err);
      useClientInference = false;
    }
  } else {
    console.log("[BridgeSign] MediaPipe Hands not found in window — using server-side frame inference.");
  }
}

// Handle results from client-side MediaPipe Hands
async function handleLocalHandsResults(results) {
  if (!cameraRunning || !useClientInference) {
    _inferInFlight = false;
    return;
  }

  const video = document.getElementById("videoFeed");
  if (!video || !video.videoWidth) {
    _inferInFlight = false;
    return;
  }

  // MediaPipe is alive — count processed frames & cancel watchdog
  _mediapipeFramesProcessed++;
  clearMediaPipeWatchdog();

  let landmarksList = null;
  const hasLocalHand = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
  if (hasLocalHand) {
    _localHandFrames++;
    _localNoHandFrames = 0;
    const hand = results.multiHandLandmarks[0];
    const w = video.videoWidth;
    const h = video.videoHeight;
    
    landmarksList = hand.map((lm, idx) => [
      idx,
      Math.round(normalizeLandmarkX(lm.x) * w),
      Math.round(lm.y * h)
    ]);
  } else {
    _localNoHandFrames++;
    if (_localNoHandFrames >= LOCAL_NO_HAND_FALLBACK_FRAMES) {
      switchToServerInference(
        `Local MediaPipe returned ${_localNoHandFrames} frames with no landmarks.`,
        "Switching to server hand detection..."
      );
      _inferInFlight = false;
      return;
    }
  }

  try {
    await sendLocalLandmarks(landmarksList);
  } finally {
    _inferInFlight = false;
  }
}

// Send pre-extracted landmarks to the server
async function sendLocalLandmarks(landmarks) {
  const seq = ++_inferSeq;
  const startedAt = performance.now();
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), INFER_HARD_TIMEOUT_MS);
  try {
    const res = await fetch("/api/infer_landmarks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        landmarks,
        camera_facing: facingMode,
        mirrored: shouldMirrorFrontCameraLandmarks(),
        consecutive_threshold: liveConsecutiveThreshold()
      }),
      signal: controller.signal
    });
    const elapsed = performance.now() - startedAt;
    if (elapsed > INFER_SLOW_WARN_MS) {
      console.debug(`[Infer] slow landmark request ${Math.round(elapsed)}ms`);
    }
    const data = await res.json().catch(() => ({}));
    data._clientSeq = seq;
    if (res.ok) {
      handleInferResponse(data);
    } else {
      console.error(`[Infer] Landmarks server error:`, data.error || res.status, data.inference_error || "");
      if (data.inference_error || data.error === "Inference not ready") {
        showToast("Sign model not loaded on server — check Render logs / redeploy.");
      }
    }
  } catch (err) {
    if (err?.name === "AbortError") {
      console.warn(`[Infer] Landmarks request timed out after ${INFER_HARD_TIMEOUT_MS}ms; dropping frame.`);
    } else {
      console.error("[Infer] Landmarks network error:", err);
    }
  } finally {
    clearTimeout(timeoutId);
  }
}

function shouldMirrorFrontCameraLandmarks() {
  return normalizeFrontCameraLandmarks && facingMode === "user";
}

function normalizeLandmarkX(x) {
  return shouldMirrorFrontCameraLandmarks() ? 1 - x : x;
}

// ── Flip Camera (Dynamic facingMode switch) ──
async function flipCamera() {
  const btn = document.getElementById("camFlipBtn");
  if (btn) btn.disabled = true;

  if (cameraRunning) {
    // Toggling while running: stop camera, switch facingMode, restart camera
    if (_videoStream) {
      _videoStream.getTracks().forEach(t => t.stop());
      _videoStream = null;
    }
    stopInferLoop();
    cameraRunning = false;
    
    // Toggle facingMode
    facingMode = facingMode === "user" ? "environment" : "user";
    
    // Restart camera
    await toggleCamera();
  } else {
    // If not running, just toggle state
    facingMode = facingMode === "user" ? "environment" : "user";
  }

  if (btn) {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="refresh-cw" width="16" height="16"></i> ${facingMode === "user" ? "Back Cam" : "Front Cam"}`;
    if (typeof lucide !== "undefined") lucide.createIcons();
  }
}

function isLocalMediaHost() {
  return ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
}

function hasSafeMediaOrigin() {
  return window.isSecureContext || isLocalMediaHost();
}

function getPermissionErrorCode(err) {
  return err?.name || err?.error || "unknown";
}

function getPermissionHelpContent(kind, err) {
  const code = getPermissionErrorCode(err);
  const isCamera = kind === "camera";
  const deviceName = isCamera ? "camera" : "microphone";
  const actionLabel = isCamera ? "Start Camera" : "Start Listening";

  const blockedSteps = [
    "Look at the address bar at the top of Chrome.",
    `Click the lock icon or the small ${deviceName} icon next to the website address.`,
    `Set ${isCamera ? "Camera" : "Microphone"} to Allow.`,
    `Reload this page, then press ${actionLabel} again.`,
    `If it is still blocked, open Windows Settings > Privacy & security > ${isCamera ? "Camera" : "Microphone"} and turn access on.`,
  ];

  if (code === "InsecureContextError") {
    return {
      kicker: "Wrong page address",
      title: `${isCamera ? "Camera" : "Microphone"} cannot run from this URL`,
      intro: `Browsers only allow ${deviceName} access on HTTPS pages or on local safe addresses like localhost.`,
      detail: `If you opened the app using an IP like http://10.x.x.x:5000 or http://0.0.0.0:5000, the browser may refuse media access even when Windows and Chrome both say Allow.`,
      steps: [
        "Close this tab.",
        "Open the app again using exactly one of these addresses:",
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "If you need to use a network IP from another device, you must serve the app over HTTPS.",
      ],
    };
  }

  if (["NotFoundError", "DevicesNotFoundError", "FoundNoDevicesError", "audio-capture"].includes(code)) {
    return {
      kicker: `${isCamera ? "Camera" : "Microphone"} setup`,
      title: `No ${deviceName} was available`,
      intro: `SMART SIGN could not find a usable ${deviceName}. This usually means the device is unplugged, disabled in Windows, or not selected by the browser.`,
      detail: `Quick check: connect the device, confirm Windows can see it, then try again here.`,
      steps: [
        `Make sure your ${deviceName} is connected and not disabled.`,
        `Open Windows Settings > Privacy & security > ${isCamera ? "Camera" : "Microphone"} and make sure access is turned on.`,
        "Close any browser tab or app that may be holding the device in a broken state.",
        `Reload this page, then press ${actionLabel} again.`,
      ],
    };
  }

  if (["NotReadableError", "TrackStartError", "AbortError"].includes(code)) {
    return {
      kicker: `${isCamera ? "Camera" : "Microphone"} busy`,
      title: `Your ${deviceName} may already be in use`,
      intro: `SMART SIGN asked for the ${deviceName}, but Chrome could not start it. Another app may already be using it.`,
      detail: `Common causes: Zoom, Teams, the Windows Camera app, voice recorder apps, or another browser tab.`,
      steps: [
        `Close other apps that might be using the ${deviceName}.`,
        "Wait a few seconds after closing them.",
        "Return to this page and try again.",
        `If it still fails, reload the page and check Windows Privacy settings for ${isCamera ? "Camera" : "Microphone"}.`,
      ],
    };
  }

  return {
    kicker: "Permission help",
    title: `${isCamera ? "Camera" : "Microphone"} access is blocked`,
    intro: `SMART SIGN needs your ${deviceName} to use this feature, but Chrome did not get permission.`,
    detail: `If you clicked Block earlier, Chrome remembers that choice until you change it from the address bar or site settings.`,
    steps: blockedSteps,
  };
}

function showPermissionHelp(kind, err, retryFn = null) {
  const panel  = document.getElementById("permissionHelp");
  const kicker = document.getElementById("permissionHelpKicker");
  const title  = document.getElementById("permissionHelpTitle");
  const intro  = document.getElementById("permissionHelpIntro");
  const steps  = document.getElementById("permissionHelpSteps");
  const detail = document.getElementById("permissionHelpDetail");
  const retry  = document.getElementById("permissionHelpRetry");
  if (!panel || !kicker || !title || !intro || !steps || !detail || !retry) return;

  const content = getPermissionHelpContent(kind, err);
  kicker.textContent = content.kicker;
  title.textContent = content.title;
  intro.textContent = content.intro;
  detail.textContent = content.detail;
  steps.innerHTML = "";
  content.steps.forEach(step => {
    const li = document.createElement("li");
    li.textContent = step;
    steps.appendChild(li);
  });

  retry.textContent = kind === "camera" ? "Retry Camera" : "Retry Microphone";
  _permissionRetry = () => {
    closePermissionHelp();
    if (typeof retryFn === "function") retryFn();
    else if (kind === "camera") toggleCamera();
    else toggleSTT();
  };
  retry.onclick = () => {
    const fn = _permissionRetry;
    if (typeof fn === "function") fn();
  };

  panel.hidden = false;
}

function closePermissionHelp() {
  const panel = document.getElementById("permissionHelp");
  if (panel) panel.hidden = true;
  _permissionRetry = null;
}

// ── Camera Toggle (browser getUserMedia → canvas → /api/infer_frame) ──
// ── Camera Toggle (browser getUserMedia → canvas → /api/infer_frame) ──
async function toggleCamera() {
  const btn  = document.getElementById("camToggleBtn");
  const pill = document.getElementById("statusPill");
  const idle = document.getElementById("cameraIdle");
  const feed = document.getElementById("videoFeed");

  if (!cameraRunning) {
    if (!hasSafeMediaOrigin()) {
      showPermissionHelp("camera", { name: "InsecureContextError" });
      return;
    }
    // Ask browser for camera access
    try {
      _videoStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode }
      });
    } catch (err) {
      showPermissionHelp("camera", err);
      return;
    }
    // Notify server to reset session state
    await fetch("/api/camera/start", { method: "POST" });

    feed.srcObject = _videoStream;
    await feed.play();
    feed.style.transform = facingMode === "user" ? "scaleX(-1)" : "none";

    cameraRunning = true;
    btn.innerHTML = '<i data-lucide="square" width="16" height="16"></i> Stop Camera';
    btn.classList.add("stop");
    if (typeof lucide !== "undefined") lucide.createIcons();
    pill.textContent = "● Live";
    pill.classList.add("live");
    idle.style.display = "none";
    feed.style.display = "block";
    sessionStart = Date.now();
    signCount    = 0;
    updateLiveTracking({ frame_count: 0, landmark_count: 0, tracking_parts: [] });

    // Watchdog check: if client MediaPipe never completes a frame, fall back.
    resetLocalHandsHealth();
    if (useClientInference) {
      _mediapipeWatchdog = setTimeout(() => {
        if (cameraRunning && _mediapipeFramesProcessed === 0 && useClientInference) {
          switchToServerInference(
            "MediaPipe watchdog triggered: no frames processed.",
            "Switching to server hand detection..."
          );
        }
      }, Math.max(4000, LOCAL_HANDS_TIMEOUT_MS * 10));
    }

    startInferLoop();
    startTimer();
    triggerRipple();
  } else {
    stopInferLoop();
    clearMediaPipeWatchdog();
    resetLocalHandsHealth();
    if (_videoStream) { _videoStream.getTracks().forEach(t => t.stop()); _videoStream = null; }
    await fetch("/api/camera/stop", { method: "POST" });
    cameraRunning = false;
    btn.innerHTML = '<i data-lucide="video" width="16" height="16"></i> Start Camera';
    btn.classList.remove("stop");
    if (typeof lucide !== "undefined") lucide.createIcons();
    pill.textContent = "Ready";
    pill.classList.remove("live");
    idle.style.display = "flex";
    feed.style.display = "none";
    feed.srcObject = null;
    stopTimer();
    updateLiveTracking({ frame_count: 0, landmark_count: 0, tracking_parts: [] });
    resetSign();
  }
}

// ── Inference loop (browser canvas → /api/infer_frame) ─────
function startInferLoop() {
  _inferInterval = setInterval(captureAndInfer, INFER_INTERVAL_MS);
}
function stopInferLoop() {
  clearInterval(_inferInterval);
  _inferInterval = null;
  _inferInFlight = false;
  _inferSeq = 0;
  _lastAppliedInferSeq = 0;
}

async function captureAndInfer() {
  if (_inferInFlight || !cameraRunning || document.hidden) return;
  const video  = document.getElementById("videoFeed");
  if (!video || !video.videoWidth) return;
  _inferFrameCounter = (_inferFrameCounter + 1) % INFER_FRAME_SKIP;
  if (_inferFrameCounter !== 0) return;

  if (useClientInference && _localHands) {
    _inferInFlight = true;
    // Safety timeout: if MediaPipe hangs (WASM not loaded / blocked), fall back
    const handsTimeout = setTimeout(() => {
      if (_inferInFlight) {
        switchToServerInference(
          `MediaPipe timed out after ${LOCAL_HANDS_TIMEOUT_MS}ms.`,
          "Hand tracking switched to server mode"
        );
        _inferInFlight = false;
      }
    }, LOCAL_HANDS_TIMEOUT_MS);
    try {
      await _localHands.send({ image: video });
      clearTimeout(handsTimeout);
    } catch (err) {
      clearTimeout(handsTimeout);
      switchToServerInference(
        `Local MediaPipe send error: ${err?.message || err}`,
        "Hand tracking switched to server mode"
      );
      _inferInFlight = false;
    }
  } else {
    const canvas = document.getElementById("inferCanvas");
    if (!canvas) return;
    // Draw the frame as captured; server runs dual-orientation static prediction.
    drawVideoFrame(video, canvas, SERVER_FRAME_MAX_DIM);

    _inferInFlight = true;
    const startedAt = performance.now();
    canvas.toBlob(async (blob) => {
      if (!blob) { _inferInFlight = false; return; }
      const seq = ++_inferSeq;
      const fd = new FormData();
      fd.append("frame", blob, "frame.jpg");
      fd.append("consecutive_threshold", liveConsecutiveThreshold());
      try {
        const res = await fetchWithTimeout(
          "/api/infer_frame",
          { method: "POST", body: fd },
          INFER_HARD_TIMEOUT_MS
        );
        const elapsed = performance.now() - startedAt;
        if (elapsed > INFER_SLOW_WARN_MS) {
          console.debug(`[Infer] slow frame request ${Math.round(elapsed)}ms`);
        }
        const data = await res.json().catch(() => ({}));
        data._clientSeq = seq;
        if (res.ok) {
          handleInferResponse(data);
        } else {
          console.error(`[Infer] Server error ${res.status}:`, data.error || res.statusText, data.inference_error || "");
          if (res.status === 401) {
            showToast("Session expired — please log in again.");
            stopInferLoop();
          } else if (data.inference_error || data.error === "Inference not ready") {
            showToast("Sign model not loaded on server — check Render logs / redeploy.");
          }
        }
      } catch (err) {
        if (err?.name === "AbortError") {
          console.warn(`[Infer] frame request timed out after ${INFER_HARD_TIMEOUT_MS}ms; dropping frame.`);
        } else {
          console.error("[Infer] Network/fetch error:", err);
        }
      }
      _inferInFlight = false;
    }, "image/jpeg", SERVER_JPEG_QUALITY);
  }
}

function handleInferResponse(d) {
  const seq = Number(d._clientSeq || 0);
  if (seq && seq < _lastAppliedInferSeq) {
    console.debug(`[Infer] stale response ignored seq=${seq}, latest=${_lastAppliedInferSeq}`);
    return;
  }
  if (seq) _lastAppliedInferSeq = seq;
  _lastInferResponseAt = performance.now();

  const handState = d.hand_state || "no_hand";
  updateLiveTracking(d);

  if (DEBUG_INFER_LOGS) logInferDecision(d);

  if (handState === "no_hand") {
    lastLabel = "";
    clearActiveSignDisplay();
  } else if (handState === "pending") {
    if (d.pending_label && d.pending_label !== lastLabel) {
      lastLabel = "";
      showPendingSign(d.pending_label, d.pending_confidence || 0);
    }
  } else if (handState === "low_confidence" || handState === "detecting") {
    lastLabel = "";
    clearActiveSignDisplay();
  }
  updateHandState(handState, d);

  if (d.mode === "word") {
    applyAssemblerState(d);
  }

  if (d.label && d.label !== lastLabel && handState === "recognised") {
    updateSign(
      d.label,
      d.confidence,
      d.mode || "letter",
      d.result_source || "",
      d.result_unit || ""
    );
    lastLabel = d.label;
    addRecentChip(d.label);
    if (d.mode !== "word") {
      signCount++;
      updateStripCount(signCount);
      softChime();
    }
  }

  if (d.confidence > 0 && handState === "recognised") {
    setConfidence(d.confidence, "confidenceFluid", "confidenceLabel");
  }
}

function logInferDecision(d) {
  const handState = d.hand_state || "no_hand";
  const pred = d.pending_label || d.static_label || "";
  const conf = d.pending_confidence || d.static_confidence || 0;
  const consecutive = d.consecutive ?? 0;
  const threshold = d.consecutive_threshold ?? 2;
  const decision = d.debug_decision || (handState === "recognised" ? `CONFIRMED ${d.label}` : "waiting");
  const top = Array.isArray(d.static_top)
    ? d.static_top.map(x => `${x.label}:${Math.round((x.confidence || 0) * 100)}%`).join(", ")
    : "";
  const changed = d.label && lastLabel && d.label !== lastLabel
    ? `, different from confirmed(${lastLabel}) -> reset counter`
    : "";
  console.debug(
    `[Infer] frame: pred=${pred || "-"} (conf=${conf.toFixed(2)}), ` +
    `consecutive_same=${consecutive}, threshold=${threshold} -> ${decision}${changed}` +
    (handState === "no_hand" ? " | no hand -> cleared" : "") +
    (top ? ` | top=[${top}]` : "")
  );
}

function showPendingSign(label, conf) {
  const el = document.getElementById("signDisplay");
  const msg = document.getElementById("signStateMsg");
  if (el) {
    el.textContent = label;
    el.dataset.state = "low_confidence";
  }
  setConfidence(conf, "confidenceFluid", "confidenceLabel");
  if (msg) {
    msg.textContent = "Confirming...";
    msg.className = "sign-state-msg unclear";
  }
}

function updateLiveTracking(d = {}) {
  const frameEl = document.getElementById("liveFrameCount");
  const partEl = document.getElementById("livePartStatus");
  const landmarkEl = document.getElementById("liveLandmarkStatus");
  const tracked = d.tracked_count ?? 0;
  const total = d.frame_count ?? 0;
  if (frameEl) frameEl.textContent = `${tracked}/${total}`;
  if (partEl) {
    const parts = Array.isArray(d.tracking_parts) ? d.tracking_parts : [];
    partEl.textContent = parts.length ? parts.join(", ") : "No hand";
  }
  if (landmarkEl) landmarkEl.textContent = String(d.landmark_count ?? 0);
}

function clearActiveSignDisplay() {
  const el = document.getElementById("signDisplay");
  if (el) el.textContent = "–";
  setConfidence(0, "confidenceFluid", "confidenceLabel");
  const fb = document.getElementById("signFeedback");
  if (fb) fb.textContent = "";
}

function resetSign() {
  const el = document.getElementById("signDisplay");
  clearActiveSignDisplay();
  el.dataset.state = "no_hand";
  document.getElementById("signStateMsg").textContent = "";
  document.getElementById("signStateMsg").className = "sign-state-msg";
  const buf = document.getElementById("wordBufferText");
  if (buf) buf.textContent = "–";
  const lw = document.getElementById("lastWordVal");
  if (lw) lw.textContent = "—";
  const sd = document.getElementById("sentenceDisplay");
  if (sd) sd.textContent = "—";
  lastLabel = "";
  lastWord = "";
}

// ── Sign Update ────────────────────────────────────────────
function updateSign(label, conf, mode, source = "", unit = "") {
  const el = document.getElementById("signDisplay");
  el.textContent = label;
  el.dataset.state = "recognised";
  el.classList.remove("bloom", "pulse");
  void el.offsetWidth;
  el.classList.add("bloom");
  setTimeout(() => { el.classList.remove("bloom"); el.classList.add("pulse"); }, 520);
  setTimeout(() => el.classList.remove("pulse"), 3500);

  // Clear state message when a real sign lands
  const msg = document.getElementById("signStateMsg");
  if (source === "gesture") {
    msg.textContent = unit === "letter" ? "Motion letter recognised" : "Gesture word recognised";
  } else if (source === "handsign") {
    msg.textContent = unit === "letter" ? "Alphabet hand sign recognised" : "Static sign recognised";
  } else {
    msg.textContent = "";
  }
  msg.className = "sign-state-msg";

  // Feedback text (only in letter mode; word mode has its own feedback)
  const fb = document.getElementById("signFeedback");
  if (mode !== "word") {
    if (source === "gesture") {
      fb.textContent = unit === "letter" ? "Motion path active" : "Gesture path active";
    } else if (conf >= 0.90) {
      fb.textContent = "Perfect 🔥";
    } else if (conf >= 0.82) {
      fb.textContent = "Great form!";
    } else {
      fb.textContent = "";
    }
  } else {
    fb.textContent = "";
  }

  // First sign celebration
  if (lastLabel === "") {
    setTimeout(() => showToast(`That's '${label}' 👋 Nice!`), 200);
  }

  const el2 = document.getElementById("stripTop");
  if (el2 && mode !== "word") el2.textContent = label;
}

// ── Hand State (honest per-frame feedback) ────────────────────
function updateHandState(handState, d) {
  const signEl = document.getElementById("signDisplay");
  const msg    = document.getElementById("signStateMsg");
  if (!signEl || !msg) return;

  if (handState === "low_confidence") {
    signEl.dataset.state = "low_confidence";
    // Show the raw static prediction even if below threshold — helps user orient hand
    const rawLabel = (d && d.static_label) ? d.static_label : "";
    const rawConf  = (d && d.static_confidence > 0) ? Math.round(d.static_confidence * 100) : 0;
    if (rawLabel && rawConf > 0) {
      msg.textContent = `🟡 Best guess: ${rawLabel} (${rawConf}%) — hold still`;
    } else {
      msg.textContent = "🟡 Sign unclear — hold still and try again";
    }
    msg.className = "sign-state-msg unclear";
  } else if (handState === "detecting") {
    signEl.dataset.state = "low_confidence";
    signEl.textContent = "–";
    msg.textContent = "Detecting...";
    msg.className = "sign-state-msg unclear";
  } else if (handState === "pending") {
    signEl.dataset.state = "low_confidence";
    const pending = d?.pending_label || d?.static_label || "";
    const conf = Math.round(((d?.pending_confidence || d?.static_confidence || 0) * 100));
    msg.textContent = pending ? `Confirming ${pending} (${conf}%)...` : "Confirming...";
    msg.className = "sign-state-msg unclear";
  } else if (handState === "no_hand") {
    if (cameraRunning) {
      signEl.dataset.state = "no_hand";
      signEl.textContent = "–";
      msg.textContent = "Show your hand to the camera";
      msg.className = "sign-state-msg no-hand";
    }
  }
  // Recognised: message cleared by updateSign()
}

// ── Word Buffer ─────────────────────────────────────────
function updateWordBuffer(buffer, lastWordVal) {
  const bufEl = document.getElementById("wordBufferText");
  const lwEl  = document.getElementById("lastWordVal");
  if (bufEl) bufEl.textContent = buffer || "–";
  if (lwEl && lastWordVal) lwEl.textContent = lastWordVal;
  if (DEBUG_INFER_LOGS) console.debug(`[WordModule] Buffered: "${buffer || ""}" (pause: false)`);
}

function updateSentence(sentence) {
  const el = document.getElementById("sentenceDisplay");
  if (el) el.textContent = sentence || "—";
}

// ── Mode Toggle ─────────────────────────────────────────
async function switchMode(mode) {
  currentMode = mode;
  // Toggle buttons
  document.getElementById("modeLetter").classList.toggle("active", mode === "letter");
  document.getElementById("modeWord").classList.toggle("active", mode === "word");
  // Show / hide word UI
  const wbc = document.getElementById("wordBufferCard");
  const ss  = document.getElementById("sentenceStrip");
  if (wbc) wbc.style.display = mode === "word" ? "block" : "none";
  if (ss)  ss.style.display  = mode === "word" ? "flex"  : "none";
  // Reset letter tracking
  lastLabel = "";
  lastWord  = "";
  resetSign();
  // Notify backend
  await fetch("/api/mode", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ mode }),
  });
  showToast(mode === "word" ? "📝 Word mode on — sign letters to build words!" : "🔤 Letter mode on");
}

// ── Confidence ─────────────────────────────────────────────
function setConfidence(val, fluidId, labelId) {
  const bar = document.getElementById(fluidId);
  const lbl = document.getElementById(labelId);
  if (bar) bar.style.setProperty("--conf", Math.round(val * 100) + "%");
  if (lbl) lbl.textContent = val > 0 ? Math.round(val * 100) + "%" : "–";
}

// ── Recent chips ───────────────────────────────────────────
function addRecentChip(label) {
  const row   = document.getElementById("recentRow");
  const empty = row.querySelector(".recent-empty");
  if (empty) empty.remove();
  const chip = document.createElement("span");
  chip.className   = "recent-chip";
  chip.textContent = label;
  row.prepend(chip);
  if (row.children.length > 8) row.removeChild(row.lastChild);
}

function clearLog() {
  const row = document.getElementById("recentRow");
  row.innerHTML = '<span class="recent-empty">None yet…</span>';
  resetSign();
  lastLabel = "";
  lastWord  = "";
}

// ── Session timer ──────────────────────────────────────────
function startTimer() {
  timerInterval = setInterval(() => {
    const secs = Math.floor((Date.now() - sessionStart) / 1000);
    const m    = String(Math.floor(secs / 60)).padStart(2,"0");
    const s    = String(secs % 60).padStart(2,"0");
    const el   = document.getElementById("stripTime");
    if (el) el.textContent = `${m}:${s}`;
  }, 1000);
}
function stopTimer() { clearInterval(timerInterval); timerInterval = null; }
function updateStripCount(n) {
  const el = document.getElementById("stripTotal");
  if (el) el.textContent = n;
}

// ── Ripple ─────────────────────────────────────────────────
function triggerRipple() {
  const wrap = document.getElementById("rippleContainer");
  const cont = document.querySelector(".camera-container");
  if (!wrap || !cont) return;
  const rect = cont.getBoundingClientRect();
  for (let i = 0; i < 3; i++) {
    setTimeout(() => {
      const r   = document.createElement("div");
      r.className = "ripple";
      r.style.left = (rect.width / 2) + "px";
      r.style.top  = (rect.height / 2) + "px";
      wrap.appendChild(r);
      setTimeout(() => r.remove(), 1300);
    }, i * 280);
  }
}

// ── Speak (live) ───────────────────────────────────────────
function speakLast() {
  const label = document.getElementById("signDisplay").textContent;
  if (label && label !== "–") speak(label);
}

function speakLastWord() {
  const lw = document.getElementById("lastWordVal");
  if (lw && lw.textContent && lw.textContent !== "—") speak(lw.textContent);
}

function speakSentence() {
  const el = document.getElementById("sentenceDisplay");
  if (el && el.textContent && el.textContent !== "—") speak(el.textContent);
}

async function flushWord() {
  try {
    const res = await fetch("/api/assembler/flush", {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) return;
    const d = await res.json();
    if (d.ok) applyAssemblerState(d);
  } catch (err) {
    console.error("Error flushing word:", err);
  }
}

async function clearCurrentWord() {
  try {
    const res = await fetch("/api/assembler/clear-word", {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) return;
    const d = await res.json();
    if (d.ok) {
      applyAssemblerState(d);
      showToast("Current word cleared");
    }
  } catch (err) {
    console.error("Error clearing word:", err);
  }
}

async function undoLastWord() {
  try {
    const res = await fetch("/api/assembler/undo", {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) return;
    const d = await res.json();
    if (d.ok) {
      applyAssemblerState(d);
      showToast(d.removed_word ? `Removed "${d.removed_word}"` : "No completed word to undo");
    }
  } catch (err) {
    console.error("Error undoing word:", err);
  }
}

async function demoIntroPhrase() {
  try {
    const res = await fetch("/api/assembler/demo-intro", {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) return;
    const d = await res.json();
    if (d.ok) {
      applyAssemblerState(d);
      const topEl = document.getElementById("stripTop");
      if (topEl) topEl.textContent = d.sentence || "Hi, my name is Benson.";
      showToast("Intro phrase ready");
    }
  } catch (err) {
    console.error("Error loading intro phrase:", err);
  }
}

function applyAssemblerState(d) {
  updateWordBuffer(d.word_buffer || "", d.last_word || "");
  updateSentence(d.sentence || "");

  if (d.completed_word) {
    console.debug(`[WordModule] Pause detected -> confirmed word: "${d.completed_word}"`);
    console.debug(`[WordModule] Final phrase: "${d.sentence || d.completed_word}"`);
    lastWord = d.completed_word;
    speak(d.completed_word);
    addRecentChip(d.completed_word);
    signCount++;
    updateStripCount(signCount);
    softChime();
    const el = document.getElementById("stripTop");
    if (el) el.textContent = d.completed_word;
  } else if (Object.prototype.hasOwnProperty.call(d, "last_word")) {
    lastWord = d.last_word || "";
  }

  if (d.completed_sentence) {
    const wordCount = d.completed_sentence.trim().split(/\s+/).length;
    if (wordCount > 1) speak(d.completed_sentence);
    showToast(`📢 "${d.completed_sentence}"`);
    addRecentChip(`💬 ${d.completed_sentence}`);
    signCount++;
    updateStripCount(signCount);
    updateSentence("");
    const lwEl = document.getElementById("lastWordVal");
    if (lwEl) lwEl.textContent = "—";
    lastWord = "";
    const topEl = document.getElementById("stripTop");
    if (topEl) topEl.textContent = d.completed_sentence;
    const strip = document.getElementById("sentenceStrip");
    if (strip) {
      strip.style.transition = "box-shadow .15s ease";
      strip.style.boxShadow  = "0 0 0 4px rgba(74,140,140,.5)";
      setTimeout(() => { strip.style.boxShadow = ""; }, 1400);
    }
  }
}

// ── Soft chime (Web Audio) ─────────────────────────────────
function softChime() {
  try {
    const ctx  = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.frequency.value = 660;
    osc.type            = "sine";
    gain.gain.setValueAtTime(0, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.08, ctx.currentTime + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.5);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.5);
  } catch (_) {}
}

// ── Toast ──────────────────────────────────────────────────
let _toastTimer = null;
function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.style.display = "block";
  t.style.opacity = "1";
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { t.style.opacity = "0"; setTimeout(() => t.style.display = "none", 400); }, 3000);
}

// ══ IMAGE TAB ══════════════════════════════════════════════
async function prepareImageUpload(file) {
  const maxDim = numberOption(IMAGE_PERF.maxDim, 0);
  if (!maxDim || !file?.type?.startsWith("image/") || typeof createImageBitmap === "undefined") {
    return { blob: file, name: file?.name || "upload.jpg" };
  }

  try {
    const bitmap = await createImageBitmap(file);
    const size = scaledSize(bitmap.width, bitmap.height, maxDim);
    if (size.width === bitmap.width && size.height === bitmap.height) {
      if (bitmap.close) bitmap.close();
      return { blob: file, name: file.name || "upload.jpg" };
    }

    const canvas = document.createElement("canvas");
    canvas.width = size.width;
    canvas.height = size.height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0, size.width, size.height);
    if (bitmap.close) bitmap.close();

    const quality = numberOption(IMAGE_PERF.jpegQuality, 0.72);
    const blob = await canvasToBlob(canvas, "image/jpeg", quality);
    return {
      blob: blob || file,
      name: (file.name || "upload").replace(/\.[^.]+$/, "") + ".jpg"
    };
  } catch (err) {
    console.warn("[Image] could not resize upload, sending original:", err);
    return { blob: file, name: file?.name || "upload.jpg" };
  }
}

async function uploadImage(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById("previewImg");
    img.src = e.target.result;
    img.style.display = "block";
    document.getElementById("dropHint").style.display = "none";
  };
  reader.readAsDataURL(file);

  const el = document.getElementById("imgSignDisplay");
  el.textContent = "…";

  // Clear any previous no-hand warning
  let warnEl = document.getElementById("imgNoHandWarning");
  if (!warnEl) {
    warnEl = document.createElement("p");
    warnEl.id = "imgNoHandWarning";
    warnEl.style.cssText = "color:#ef4444;font-weight:600;font-size:.95rem;margin-top:10px;text-align:center;display:none;";
    el.parentNode && el.parentNode.insertBefore(warnEl, el.nextSibling);
  }
  warnEl.style.display = "none";
  warnEl.textContent = "";

  const fd = new FormData();
  const upload = await prepareImageUpload(file);
  fd.append("image", upload.blob, upload.name);
  let d;
  try {
    const res = await fetchWithTimeout(
      "/api/translate_image",
      { method: "POST", body: fd },
      numberOption(IMAGE_PERF.timeoutMs, 1200)
    );
    d = await res.json();
  } catch (err) {
    el.textContent = "–";
    warnEl.textContent = err?.name === "AbortError"
      ? "Upload timed out. Try a smaller or clearer photo."
      : "⚠️ Upload failed. Please try again.";
    warnEl.style.display = "block";
    return;
  }

  // No hand detected — show warning, refuse to display a result
  if (d.no_hand || d.error) {
    el.textContent = "–";
    setConfidence(0, "imgConfFluid", "imgConfLabel");
    warnEl.textContent = "🚫 " + (d.error || "No hand detected. Upload a clear photo of a hand sign.");
    warnEl.style.display = "block";
    return;
  }

  el.textContent = d.label || "–";
  el.classList.remove("bloom"); void el.offsetWidth; el.classList.add("bloom");
  const pct = parseInt((d.confidence || "0%").replace("%", "")) / 100;
  setConfidence(pct, "imgConfFluid", "imgConfLabel");
  softChime();
}

function speakImgResult() {
  const label = document.getElementById("imgSignDisplay").textContent;
  if (label && label !== "–") speak(label);
}

// Drag-and-drop
const dropZone = document.getElementById("imgDropZone");
if (dropZone) {
  dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.style.opacity = ".75"; });
  dropZone.addEventListener("dragleave", () => dropZone.style.opacity = "1");
  dropZone.addEventListener("drop", e => {
    e.preventDefault(); dropZone.style.opacity = "1";
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      uploadImage({ target: { files: [file] } });
    }
  });
}

// ══ LEARN TAB ══════════════════════════════════════════════
let currentSign = "Hello";

function selectSign(btn, sign) {
  document.querySelectorAll("#learnPills .pill").forEach(p => p.classList.remove("active"));
  btn.classList.add("active");
  currentSign = sign;
}

async function fetchLesson() {
  const res  = await fetch(`/api/learn/lesson?sign=${encodeURIComponent(currentSign)}`);
  const data = await res.json();
  const tip  = document.getElementById("lessonTip");
  tip.textContent = `💡 ${data.tip}`;
  tip.style.animation = "none"; void tip.offsetWidth;
  tip.style.animation = "pourIn .5s ease";
  speak(`Here's how to sign ${currentSign}. ${data.tip}`);
}

// ══ EMERGENCY TAB ══════════════════════════════════════════
async function loadEmergency() {
  const res     = await fetch("/api/emergency/all");
  const phrases = await res.json();
  const grid    = document.getElementById("emergencyGrid");
  grid.innerHTML = "";

  let i = 0;
  for (const [id, phrase] of Object.entries(phrases)) {
    const isCustom = id.startsWith("c_");
    const card = document.createElement("div");
    card.className = "emergency-card pour-in";
    card.style.animationDelay = `${i * 0.06}s`;
    card.innerHTML = `
      <div class="emergency-phrase">${phrase}</div>
      <div class="emergency-actions">
        <button class="emergency-speak-btn" onclick="emergencySpeak(event,'${phrase}')">
          <i data-lucide="volume-2" width="13" height="13"></i> Speak
        </button>
        ${isCustom ? `<button class="emergency-delete-btn" onclick="deleteCustomPhrase(event,'${id}')">
          <i data-lucide="trash-2" width="14" height="14"></i>
        </button>` : ""}
      </div>`;
    grid.appendChild(card);
    i++;
  }
  if (typeof lucide !== "undefined") lucide.createIcons();
}

function emergencySpeak(e, phrase) {
  e.stopPropagation();
  speak(phrase);
  softChime();
}

async function addCustomPhrase() {
  const input  = document.getElementById("customPhraseInput");
  const phrase = input.value.trim();
  if (!phrase) { showToast("Please type a phrase first."); return; }
  const res = await fetch("/api/emergency/custom", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phrase }),
  });
  if (res.ok) {
    input.value = "";
    showToast(`"${phrase}" added!`);
    loadEmergency();
  }
}

async function deleteCustomPhrase(e, id) {
  e.stopPropagation();
  await fetch(`/api/emergency/custom/${id}`, { method: "DELETE" });
  loadEmergency();
}

// Allow Enter in custom phrase input
const cpi = document.getElementById("customPhraseInput");
if (cpi) cpi.addEventListener("keydown", e => { if (e.key === "Enter") addCustomPhrase(); });

// ══ CALL ROOM ENTRY ═══════════════════════════════════════
function extractRoomCode(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";

  try {
    const url = new URL(raw);
    const match = url.pathname.match(/\/call\/([^/]+)/);
    if (match) return decodeURIComponent(match[1]);
  } catch (_) {}

  return raw.replace(/^\/?call\//i, "").trim();
}

function joinCallRoom() {
  const input = document.getElementById("joinRoomInput");
  const roomCode = extractRoomCode(input ? input.value : "");

  if (!/^[A-Za-z0-9_-]{3,40}$/.test(roomCode)) {
    showToast("Enter a valid room code or SMART SIGN call link.");
    if (input) input.focus();
    return;
  }

  window.location.href = `/call/${encodeURIComponent(roomCode)}`;
}

const joinInput = document.getElementById("joinRoomInput");
if (joinInput) joinInput.addEventListener("keydown", e => { if (e.key === "Enter") joinCallRoom(); });

// ══ STATS TAB ══════════════════════════════════════════════
async function loadStats() {
  const res   = await fetch("/api/stats");
  const stats = await res.json();

  document.getElementById("statsCards").innerHTML = `
    <div class="stat-card">
      <div class="stat-number">${stats.total || 0}</div>
      <div class="stat-label">Total Translations</div>
    </div>
    <div class="stat-card">
      <div class="stat-number" style="font-size:1.8rem;color:var(--teal)">${stats.most_common || "–"}</div>
      <div class="stat-label">Most Common Sign</div>
    </div>`;

  const breakdown = document.getElementById("breakdownWrap");
  breakdown.innerHTML = "";
  const counts = stats.counts || {};
  const max    = Math.max(...Object.values(counts), 1);
  for (const [sign, count] of Object.entries(counts)) {
    const row = document.createElement("div");
    row.className = "breakdown-row";
    row.innerHTML = `
      <span class="breakdown-sign">${sign}</span>
      <div class="breakdown-bar-wrap">
        <div class="breakdown-bar" style="width:${Math.round(count / max * 100)}%"></div>
      </div>
      <span class="breakdown-count">${count}</span>`;
    breakdown.appendChild(row);
  }
  if (!Object.keys(counts).length) {
    breakdown.innerHTML = `<p style="color:var(--earth-mid);margin-top:16px;font-size:.9rem">No data yet — start signing to see your stats here.</p>`;
  }
}

// ══ AVATAR MOTION RECORDER (MediaPipe Holistic in browser) ══
let _motion = {
  holistic: null,
  stream: null,
  running: false,
  recording: false,
  processing: false,
  trackerReady: false,
  trackerLoading: false,
  trackerError: "",
  trackerMode: "server_hand",
  trackerWarmupStartedAt: 0,
  lastTrackerResultAt: 0,
  lastSentAt: 0,
  processedFrames: 0,
  trackedFrames: 0,
  lastLandmarkCount: 0,
  lastParts: [],
  captureCanvas: null,
  frames: [],
  startedAt: 0,
  lastClip: null,
  lastLabel: "",
  availableAnimations: [],
  cwasaLabels: [],
  vrmLabels: [],
  libraryLoaded: false,
};
const MOTION_HOLISTIC_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629";
const MOTION_TRACK_INTERVAL_MS = 90;
const MOTION_POSE_LINES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];
const MOTION_HAND_LINES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]];
const MOTION_FINGERS = {
  Thumb: [1, 2, 3, 4],
  Index: [5, 6, 7, 8],
  Middle: [9, 10, 11, 12],
  Ring: [13, 14, 15, 16],
  Pinky: [17, 18, 19, 20],
};
const MOTION_FINGER_BONES = {
  "Thumb:0": "rightThumbMetacarpal",
  "Thumb:1": "rightThumbProximal",
  "Thumb:2": "rightThumbDistal",
  "Index:0": "rightIndexProximal",
  "Index:1": "rightIndexIntermediate",
  "Index:2": "rightIndexDistal",
  "Middle:0": "rightMiddleProximal",
  "Middle:1": "rightMiddleIntermediate",
  "Middle:2": "rightMiddleDistal",
  "Ring:0": "rightRingProximal",
  "Ring:1": "rightRingIntermediate",
  "Ring:2": "rightRingDistal",
  "Pinky:0": "rightLittleProximal",
  "Pinky:1": "rightLittleIntermediate",
  "Pinky:2": "rightLittleDistal",
};

function initMotionRecorder() {
  initMotionAvatar();
  setMotionStatus(_motion.running
    ? (_motion.trackerReady ? "Camera ready - server tracker active" : "Camera ready - warming up tracker")
    : "Saved signs ready");
  loadMotionLibrary();
  setMotionDiagnostics();
  updateMotionButtons();
}

function initMotionAvatar() {
  if (!window.BridgeSignAvatar) return;
  window.BridgeSignAvatar.init("motionAvatarContainer", {
    globalName: "motionAvatar",
    loaderId: "motionAvatarLoading",
    fallbackLabel: "Sign Preview",
    readyText: "Avatar ready",
    activeNote: "Playing saved sign clips. Unknown words are fingerspelled."
  });
}

function setMotionStatus(text) {
  const el = document.getElementById("motionStatus");
  if (el) el.textContent = text;
}

function setMotionFrameCount(count) {
  const el = document.getElementById("motionFrameCount");
  if (el) el.textContent = `${count} frame${count === 1 ? "" : "s"}`;
}

function motionPartsText(parts) {
  return parts && parts.length ? parts.join(", ") : "none";
}

function setMotionDiagnostics() {
  const processedEl = document.getElementById("motionProcessedCount");
  const partsEl = document.getElementById("motionPartsStatus");
  const landmarkEl = document.getElementById("motionLandmarkStatus");
  if (processedEl) processedEl.textContent = `${_motion.trackedFrames || 0}/${_motion.processedFrames || 0} tracked`;
  if (partsEl) partsEl.textContent = `Parts: ${motionPartsText(_motion.lastParts)}`;
  if (landmarkEl) landmarkEl.textContent = `${_motion.lastLandmarkCount || 0} landmarks`;
}

function resetMotionDiagnostics() {
  _motion.processedFrames = 0;
  _motion.trackedFrames = 0;
  _motion.lastLandmarkCount = 0;
  _motion.lastParts = [];
  setMotionDiagnostics();
}

function updateMotionButtons() {
  const cameraBtn = document.getElementById("motionCameraBtn");
  const recordBtn = document.getElementById("motionRecordBtn");
  const stopBtn = document.getElementById("motionStopBtn");
  const previewBtn = document.getElementById("motionPreviewBtn");
  const saveBtn = document.getElementById("motionSaveBtn");
  const downloadBtn = document.getElementById("motionDownloadBtn");

  if (cameraBtn) {
    cameraBtn.innerHTML = _motion.running
      ? '<i data-lucide="video-off" width="16" height="16"></i> Stop Camera'
      : '<i data-lucide="video" width="16" height="16"></i> Camera';
    cameraBtn.classList.toggle("stop", _motion.running);
  }
  if (recordBtn) {
    recordBtn.disabled = !_motion.running || !_motion.trackerReady || _motion.recording || !!_motion.trackerError;
    if (!_motion.running) recordBtn.title = "Start the motion camera first.";
    else if (_motion.trackerError) recordBtn.title = _motion.trackerError;
    else if (!_motion.trackerReady) recordBtn.title = "Waiting for the hand tracker to load.";
    else recordBtn.title = "";
  }
  if (stopBtn) stopBtn.disabled = !_motion.recording;
  if (previewBtn) previewBtn.disabled = !_motion.lastClip;
  if (saveBtn) saveBtn.disabled = !_motion.lastClip;
  if (downloadBtn) downloadBtn.disabled = !_motion.lastClip;
  if (typeof lucide !== "undefined") lucide.createIcons();
}

function safeMotionLabel() {
  const input = document.getElementById("motionLabelInput");
  const raw = (input?.value || "SIGN").trim().toUpperCase().replace(/[^A-Z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  const label = /^[A-Z]/.test(raw) ? raw.slice(0, 40) : `SIGN_${raw}`.slice(0, 40);
  if (input) input.value = label;
  return label || "SIGN";
}

async function loadMotionLibrary() {
  if (_motion.libraryLoaded) {
    renderMotionLibrary();
    return;
  }
  const countEl = document.getElementById("motionLibraryCount");
  if (countEl) countEl.textContent = "Loading";
  try {
    const res = await fetch("/api/avatar/animations");
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Library request failed: ${res.status}`);
    _motion.availableAnimations = Array.isArray(data.animations) ? data.animations : [];
    _motion.cwasaLabels = Array.isArray(data.cwasa) ? data.cwasa : [];
    _motion.vrmLabels = Array.isArray(data.vrm) ? data.vrm : [];
    _motion.libraryLoaded = true;
    renderMotionLibrary();
  } catch (err) {
    console.warn("Avatar sign library failed:", err);
    if (countEl) countEl.textContent = "Unavailable";
    const grid = document.getElementById("motionLibraryGrid");
    if (grid) grid.innerHTML = `<span class="stt-guidance-empty">Saved signs could not be loaded.</span>`;
  }
}

function renderMotionLibrary() {
  const grid = document.getElementById("motionLibraryGrid");
  const countEl = document.getElementById("motionLibraryCount");
  if (!grid) return;
  const labels = _motion.availableAnimations || [];
  if (countEl) countEl.textContent = `${labels.length} signs`;
  grid.innerHTML = "";
  if (!labels.length) {
    grid.innerHTML = `<span class="stt-guidance-empty">No saved signs found.</span>`;
    return;
  }
  labels.forEach(label => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "motion-sign-chip";
    btn.textContent = label.replace(/_/g, " ");
    btn.onclick = () => playSavedSign(label);
    grid.appendChild(btn);
  });
}

async function waitForMotionAvatar() {
  initMotionAvatar();
  const avatar = window.motionAvatar;
  if (!avatar) {
    showToast("Avatar is still loading.");
    return null;
  }
  if (avatar._readyPromise) await avatar._readyPromise;
  return avatar;
}

function normalizeMotionSignLabel(value) {
  return String(value || "").trim().toUpperCase().replace(/[^A-Z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function motionCWASALabelSet() {
  const labels = _motion.cwasaLabels || [];
  return new Set(labels.map(normalizeMotionSignLabel));
}

function motionCanPlayCWASA(label, cwasaSet = motionCWASALabelSet()) {
  const safeLabel = normalizeMotionSignLabel(label);
  return !!safeLabel && cwasaSet.has(safeLabel);
}

async function queueMotionLetters(avatar, text) {
  const letters = String(text || "").replace(/[^A-Za-z]/g, "").toUpperCase();
  if (!letters) return;
  if (typeof avatar.queueLetters === "function") {
    await avatar.queueLetters(letters);
    return;
  }
  for (const letter of letters) await avatar.queueSign(letter);
}

async function queueMotionCWASATextFallback(avatar, text) {
  const cwasaSet = motionCWASALabelSet();
  const tokens = String(text || "").match(/[A-Za-z_]+/g) || [];
  for (const token of tokens) {
    const label = normalizeMotionSignLabel(token);
    if (motionCanPlayCWASA(label, cwasaSet)) await avatar.queueSign(label);
    else await queueMotionLetters(avatar, label || token);
  }
}

async function queueMotionGuidedText(avatar, guidance) {
  const cwasaSet = motionCWASALabelSet();
  for (const item of guidance || []) {
    const label = normalizeMotionSignLabel(item.sign_label);
    if (label && motionCanPlayCWASA(label, cwasaSet)) {
      await avatar.queueSign(label);
      continue;
    }

    if (item.type === "letter" && label) {
      await avatar.queueSign(label);
      continue;
    }

    const fallbackText = item.word || label;
    await queueMotionLetters(avatar, fallbackText);
  }
}

async function playSavedSign(label) {
  const safeLabel = normalizeMotionSignLabel(label);
  if (!safeLabel) return;
  const avatar = await waitForMotionAvatar();
  if (!avatar) return;
  const status = document.getElementById("motionClipStatus");
  if (status) status.textContent = `Playing ${safeLabel.replace(/_/g, " ")}`;
  setMotionStatus("Playing saved sign");
  await avatar.queueSign(safeLabel);
}

async function playMotionText() {
  const input = document.getElementById("motionTextInput");
  const text = (input?.value || "").trim();
  if (!text) {
    showToast("Type a word or phrase for the avatar.");
    return;
  }
  const avatar = await waitForMotionAvatar();
  if (!avatar) return;
  const status = document.getElementById("motionClipStatus");
  if (status) status.textContent = "Preparing phrase";
  setMotionStatus("Preparing CWASA text");

  await loadMotionLibrary();

  try {
    const res = await fetch("/api/stt/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`Text guidance failed: ${res.status}`);
    const data = await res.json();
    const guidance = Array.isArray(data.guidance) ? data.guidance : [];
    if (guidance.length) {
      if (status) status.textContent = "Playing CWASA-guided phrase";
      setMotionStatus("Playing CWASA-guided phrase");
      await queueMotionGuidedText(avatar, guidance);
      return;
    }
  } catch (err) {
    console.warn("Motion text guidance failed, using direct CWASA spelling:", err);
  }

  if (status) status.textContent = "Playing CWASA phrase";
  setMotionStatus("Playing CWASA phrase");
  await queueMotionCWASATextFallback(avatar, text);
}

function loadMotionScript(src) {
  return new Promise((resolve, reject) => {
    if ([...document.scripts].some(script => script.src === src)) {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.crossOrigin = "anonymous";
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Could not load ${src}`));
    document.head.appendChild(script);
  });
}

async function ensureMotionHolistic() {
  if (_motion.holistic) {
    _motion.trackerError = "";
    return _motion.holistic;
  }
  _motion.trackerLoading = true;
  _motion.trackerError = "";
  setMotionStatus("Loading tracker");
  try {
    await loadMotionScript(`${MOTION_HOLISTIC_CDN}/holistic.js`);
    if (typeof Holistic === "undefined") throw new Error("MediaPipe Holistic did not register.");
    _motion.holistic = new Holistic({ locateFile: file => `${MOTION_HOLISTIC_CDN}/${file}` });
    _motion.holistic.setOptions({
      modelComplexity: 0,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.55,
      minTrackingConfidence: 0.55,
    });
    _motion.holistic.onResults(handleMotionResults);
    return _motion.holistic;
  } catch (err) {
    _motion.trackerReady = false;
    _motion.trackerError = "Motion tracker could not load. Check internet access and reload the page.";
    throw err;
  } finally {
    _motion.trackerLoading = false;
    updateMotionButtons();
  }
}

async function toggleMotionCamera() {
  if (_motion.running) {
    stopMotionCamera();
    return;
  }
  await startMotionCamera();
}

async function startMotionCamera() {
  if (!hasSafeMediaOrigin()) {
    showPermissionHelp("camera", { name: "InsecureContextError" }, toggleMotionCamera);
    return;
  }
  try {
    _motion.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 960, height: 720, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    console.warn("Motion camera/tracker failed:", err);
    if (String(err?.message || "").includes("MediaPipe")) showToast("Motion tracker could not load. Check internet access.");
    else showPermissionHelp("camera", err, toggleMotionCamera);
    setMotionStatus("Camera failed");
    return;
  }

  const video = document.getElementById("motionVideo");
  const empty = document.getElementById("motionVideoEmpty");
  if (!video) return;
  video.srcObject = _motion.stream;
  await video.play();
  if (empty) empty.style.display = "none";
  video.style.display = "block";
  _motion.running = true;
  _motion.processing = false;
  _motion.trackerReady = true;
  _motion.trackerError = "";
  _motion.trackerMode = "server_hand";
  _motion.trackerWarmupStartedAt = performance.now();
  _motion.lastTrackerResultAt = 0;
  _motion.lastSentAt = 0;
  resetMotionDiagnostics();
  setMotionStatus("Camera ready - server hand tracker active");
  updateMotionButtons();
  requestAnimationFrame(processMotionFrame);
}

function stopMotionCamera() {
  if (_motion.recording) stopMotionRecording();
  if (_motion.stream) {
    _motion.stream.getTracks().forEach(track => track.stop());
    _motion.stream = null;
  }
  _motion.running = false;
  _motion.processing = false;
  _motion.trackerReady = false;
  _motion.trackerError = "";
  _motion.trackerWarmupStartedAt = 0;
  _motion.lastTrackerResultAt = 0;
  _motion.lastSentAt = 0;
  const video = document.getElementById("motionVideo");
  const overlay = document.getElementById("motionOverlay");
  const empty = document.getElementById("motionVideoEmpty");
  if (video) {
    video.pause();
    video.srcObject = null;
    video.style.display = "none";
  }
  if (overlay) overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  if (empty) empty.style.display = "flex";
  resetMotionDiagnostics();
  setMotionStatus("Idle");
  updateMotionButtons();
}

async function processMotionFrame() {
  if (!_motion.running) return;
  const video = document.getElementById("motionVideo");
  const now = performance.now();
  if (video && video.videoWidth && !_motion.processing && now - _motion.lastSentAt >= MOTION_TRACK_INTERVAL_MS) {
    _motion.processing = true;
    _motion.lastSentAt = now;
    try {
      await sendMotionFrameToServer(video);
    } catch (err) {
      console.warn("Motion tracker frame failed:", err);
      _motion.trackerError = "Motion tracker frame failed. Check the Flask server and try again.";
      setMotionStatus("Tracker failed");
      updateMotionButtons();
    } finally {
      _motion.processing = false;
    }
  }
  requestAnimationFrame(processMotionFrame);
}

function canvasToBlob(canvas, type = "image/jpeg", quality = 0.72) {
  return new Promise(resolve => canvas.toBlob(resolve, type, quality));
}

async function sendMotionFrameToServer(video) {
  if (!_motion.captureCanvas) _motion.captureCanvas = document.createElement("canvas");
  const canvas = _motion.captureCanvas;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await canvasToBlob(canvas);
  if (!blob) return;
  const fd = new FormData();
  fd.append("frame", blob, "motion.jpg");
  const res = await fetch("/api/track_frame", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Tracker request failed: ${res.status}`);
  handleMotionTrackResponse(data);
}

function startMotionRecording() {
  if (!_motion.running) {
    showToast("Start the motion camera first.");
    return;
  }
  if (!_motion.trackerReady) {
    showToast(_motion.trackerError || "Motion tracker is still loading.");
    return;
  }
  _motion.frames = [];
  _motion.startedAt = performance.now();
  _motion.recording = true;
  _motion.lastClip = null;
  resetMotionDiagnostics();
  setMotionFrameCount(0);
  setMotionStatus("Recording - waiting for tracked hand landmarks");
  const out = document.getElementById("motionClipOutput");
  if (out) out.value = "";
  const status = document.getElementById("motionClipStatus");
  if (status) status.textContent = "Recording";
  updateMotionButtons();
}

function stopMotionRecording() {
  if (!_motion.recording) return;
  _motion.recording = false;
  try {
    const label = safeMotionLabel();
    const clip = buildMotionClip(label, _motion.frames);
    _motion.lastClip = clip;
    _motion.lastLabel = label;
    const out = document.getElementById("motionClipOutput");
    if (out) out.value = JSON.stringify(clip, null, 2);
    const status = document.getElementById("motionClipStatus");
    if (status) status.textContent = `${label} ready`;
    setMotionStatus("Clip ready");
  } catch (err) {
    console.warn("Motion clip build failed:", err);
    setMotionStatus("Clip failed");
    const status = document.getElementById("motionClipStatus");
    if (status) status.textContent = "No clip";
    showToast(err.message || "Motion clip could not be built.");
  }
  updateMotionButtons();
}

function handleMotionTrackResponse(data) {
  if (!_motion.running) return;
  _motion.processedFrames += 1;
  _motion.trackerError = "";
  handleMotionResults(motionResultsFromServer(data || {}));
}

function motionResultsFromServer(data) {
  const hands = Array.isArray(data.hands) ? data.hands : [];
  let rightHand = null;
  let leftHand = null;

  for (const hand of hands) {
    const label = String(hand.label || "").toLowerCase();
    const landmarks = Array.isArray(hand.landmarks) ? hand.landmarks : null;
    if (!landmarks) continue;
    if (label.includes("right") && !rightHand) rightHand = landmarks;
    else if (label.includes("left") && !leftHand) leftHand = landmarks;
    else if (!rightHand) rightHand = landmarks;
    else if (!leftHand) leftHand = landmarks;
  }

  // Map pose landmarks from server (33 MediaPipe pose landmarks)
  const rawPose = Array.isArray(data.pose) ? data.pose : [];
  const poseLandmarks = rawPose.length >= 25 ? rawPose : null;

  return {
    poseLandmarks: poseLandmarks,
    rightHandLandmarks: rightHand,
    leftHandLandmarks: leftHand,
    faceLandmarks: null,
    _parts: Array.isArray(data.parts) ? data.parts : [],
    _landmarkCount: Number(data.landmark_count || 0),
  };
}

function motionPartsFromResults(results) {
  if (Array.isArray(results?._parts) && results._parts.length) return results._parts;
  const parts = [];
  if (results?.poseLandmarks) parts.push("body");
  if (results?.rightHandLandmarks) parts.push("right hand");
  if (results?.leftHandLandmarks) parts.push("left hand");
  if (results?.faceLandmarks) parts.push("face");
  return parts;
}

function motionLandmarkCount(results) {
  if (Number.isFinite(results?._landmarkCount)) return results._landmarkCount;
  return (results?.poseLandmarks?.length || 0)
    + (results?.rightHandLandmarks?.length || 0)
    + (results?.leftHandLandmarks?.length || 0)
    + (results?.faceLandmarks?.length || 0);
}

function handleMotionResults(results) {
  _motion.lastTrackerResultAt = performance.now();
  if (!_motion.trackerReady) {
    _motion.trackerReady = true;
    _motion.trackerError = "";
    setMotionStatus("Camera ready - tracker active");
    updateMotionButtons();
  }
  const parts = motionPartsFromResults(results);
  const landmarkCount = motionLandmarkCount(results);
  _motion.lastParts = parts;
  _motion.lastLandmarkCount = landmarkCount;
  setMotionDiagnostics();
  drawMotionOverlay(results);
  if (!_motion.recording) return;
  const pose = cloneLandmarks(results.poseLandmarks);
  const rightHand = cloneLandmarks(results.rightHandLandmarks);
  const leftHand = cloneLandmarks(results.leftHandLandmarks);
  const face = cloneLandmarks(results.faceLandmarks);
  if (!pose && !rightHand && !leftHand) {
    setMotionFrameCount(_motion.frames.length);
    setMotionStatus(`Recording — ${_motion.processedFrames} sent, no landmarks yet. Show your hands and upper body.`);
    setMotionDiagnostics();
    return;
  }
  _motion.trackedFrames += 1;
  _motion.frames.push({
    t: (performance.now() - _motion.startedAt) / 1000,
    pose,
    rightHand,
    leftHand,
    face,
  });
  setMotionFrameCount(_motion.frames.length);
  const savedParts = [];
  if (pose) savedParts.push("body");
  if (rightHand) savedParts.push("R hand");
  if (leftHand) savedParts.push("L hand");
  if (face) savedParts.push("face");
  setMotionStatus(`Recording — ${savedParts.join(", ")} — ${_motion.frames.length} saved / ${_motion.processedFrames} sent`);
}

function cloneLandmarks(landmarks) {
  if (!landmarks || !landmarks.length) return null;
  return landmarks.map(p => ({
    x: Number(p.x),
    y: Number(p.y),
    z: Number(p.z || 0),
    visibility: Number(p.visibility ?? 1),
  }));
}

function drawMotionOverlay(results) {
  const video = document.getElementById("motionVideo");
  const canvas = document.getElementById("motionOverlay");
  if (!video || !canvas || !video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(74,140,140,.9)";
  ctx.fillStyle = "rgba(192,83,58,.95)";
  drawLandmarkLines(ctx, results.poseLandmarks, MOTION_POSE_LINES, canvas.width, canvas.height);
  drawLandmarkLines(ctx, results.rightHandLandmarks, MOTION_HAND_LINES, canvas.width, canvas.height);
  drawLandmarkLines(ctx, results.leftHandLandmarks, MOTION_HAND_LINES, canvas.width, canvas.height);
  drawLandmarkDots(ctx, results.poseLandmarks, canvas.width, canvas.height, 3);
  drawLandmarkDots(ctx, results.rightHandLandmarks, canvas.width, canvas.height, 2.5);
  drawLandmarkDots(ctx, results.leftHandLandmarks, canvas.width, canvas.height, 2.5);
}

function drawLandmarkLines(ctx, landmarks, lines, width, height) {
  if (!landmarks) return;
  for (const [a, b] of lines) {
    const p1 = landmarks[a], p2 = landmarks[b];
    if (!p1 || !p2) continue;
    ctx.beginPath();
    ctx.moveTo(p1.x * width, p1.y * height);
    ctx.lineTo(p2.x * width, p2.y * height);
    ctx.stroke();
  }
}

function drawLandmarkDots(ctx, landmarks, width, height, radius) {
  if (!landmarks) return;
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * width, p.y * height, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function buildMotionClip(label, frames) {
  if (!frames || frames.length < 8) throw new Error("Record at least 8 tracked frames.");
  const sampled = sampleMotionFrames(frames, 180);
  const firstTime = sampled[0].t;
  const times = sampled.map(frame => Math.max(0, frame.t - firstTime));
  const duration = Math.max(times[times.length - 1], 0.25);
  const poses = [];
  let previous = null;
  for (const frame of sampled) {
    const pose = motionFrameToPose(frame, previous);
    previous = pose;
    poses.push(pose);
  }
  const trackNames = [...new Set(poses.flatMap(pose => Object.keys(pose)))].sort();
  const tracks = trackNames.map(name => {
    const values = [];
    for (const pose of poses) values.push(...(pose[name] || [0, 0, 0, 1]));
    return { name: `${name}.quaternion`, type: "quaternion", times, values };
  });
  return { name: label, duration, source: "motion_recorder", tracks };
}

function sampleMotionFrames(frames, maxFrames) {
  if (frames.length <= maxFrames) return frames;
  const sampled = [];
  for (let i = 0; i < maxFrames; i++) sampled.push(frames[Math.round(i * (frames.length - 1) / (maxFrames - 1))]);
  return sampled;
}

function motionFrameToPose(frame, previous = null) {
  const pose = { ...(previous || {}), ...bodyPoseFromLandmarks(frame.pose) };
  if (frame.rightHand) Object.assign(pose, handPoseFromLandmarks(frame.rightHand, "right"));
  if (frame.leftHand) Object.assign(pose, handPoseFromLandmarks(frame.leftHand, "left"));
  return pose;
}

function clampMotion(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function motionQuat(roll, pitch, yaw) {
  const cr = Math.cos(roll / 2), sr = Math.sin(roll / 2);
  const cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
  const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2);
  return [
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
    cr * cp * cy + sr * sp * sy,
  ];
}

function bodyPoseFromLandmarks(pose) {
  if (!pose) return {};
  const ls = pose[11], rs = pose[12], le = pose[13], re = pose[14], lw = pose[15], rw = pose[16];
  const lh = pose[23], rh = pose[24], nose = pose[0];
  if (!ls || !rs || !lh || !rh) return {};

  const shoulderCenter = avgPoint(ls, rs);
  const hipCenter = avgPoint(lh, rh);

  // Normalize all spatial tracking offsets by the user's shoulder-to-shoulder width.
  // This makes the avatar translation completely distance-invariant (scale-invariant).
  const shoulderDist = Math.hypot(rs.x - ls.x, rs.y - ls.y) || 0.1;

  const torsoLean = clampMotion((shoulderCenter.x - hipCenter.x) / shoulderDist, -1.2, 1.2);
  const shoulderTilt = clampMotion((rs.y - ls.y) / shoulderDist, -0.6, 0.6);
  const forward = clampMotion(((hipCenter.y - shoulderCenter.y) / shoulderDist) - 1.8, -0.5, 0.5);

  const headYaw = nose ? clampMotion((nose.x - shoulderCenter.x) / shoulderDist, -1.2, 1.2) : 0;
  const headPitch = nose ? clampMotion(((nose.y - shoulderCenter.y) / shoulderDist) - 0.6, -0.8, 0.8) : -0.03;

  const result = {
    hips: motionQuat(0, torsoLean * 0.35, -shoulderTilt * 0.7),
    spine: motionQuat(0.04 + forward * 0.5, torsoLean * 0.5, shoulderTilt * 0.6),
    chest: motionQuat(0.08 + forward * 0.8, torsoLean * 0.7, shoulderTilt * 0.8),
    upperChest: motionQuat(0.05 + forward * 0.4, torsoLean * 0.5, shoulderTilt * 0.6),
    neck: motionQuat(-0.02, headYaw * 0.4, -shoulderTilt * 0.3),
    head: motionQuat(headPitch * 1.2, headYaw * 1.4, -shoulderTilt * 0.5),
  };

  if (rs && re && rw) Object.assign(result, armPoseFromLandmarks("right", rs, re, rw));
  if (ls && le && lw) Object.assign(result, armPoseFromLandmarks("left", ls, le, lw));
  return result;
}

function avgPoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: ((a.z || 0) + (b.z || 0)) / 2 };
}

function armPoseFromLandmarks(side, shoulder, elbow, wrist) {
  const upper = { x: elbow.x - shoulder.x, y: elbow.y - shoulder.y, z: (elbow.z || 0) - (shoulder.z || 0) };
  const lower = { x: wrist.x - elbow.x, y: wrist.y - elbow.y, z: (wrist.z || 0) - (elbow.z || 0) };

  const upperLen = Math.hypot(upper.x, upper.y) || 0.001;
  const lowerLen = Math.hypot(lower.x, lower.y) || 0.001;
  const ux = upper.x / upperLen;
  const uy = upper.y / upperLen;
  const lx = lower.x / lowerLen;
  const ly = lower.y / lowerLen;

  const prefix = side === "right" ? "right" : "left";
  const zSign = side === "right" ? -1 : 1;
  const sideSign = side === "right" ? -1 : 1;

  const lift = clampMotion(-uy, -1, 1);
  const reach = clampMotion(sideSign * ux, -1, 1);
  const forearmLift = clampMotion(-ly, -1, 1);
  const forearmReach = clampMotion(sideSign * lx, -1, 1);
  const depth = clampMotion(((upper.z || 0) + (lower.z || 0)) * 2.5, -0.8, 0.8);

  const bendAngle = angleBetween2D({ x: ux, y: uy }, { x: lx, y: ly });
  const elbowBend = clampMotion((Math.PI - bendAngle) / Math.PI, 0, 1);

  const shoulderPitch = clampMotion(-0.18 - lift * 1.0 + depth * 0.45, -1.35, 0.55);
  const shoulderYaw = clampMotion(reach * 1.05 + depth * 0.35, -1.15, 1.15);
  const shoulderRoll = clampMotion(zSign * (0.2 + reach * 0.45 + lift * 0.55), -1.25, 1.25);

  const elbowPitch = clampMotion(-0.18 - elbowBend * 1.85, -2.05, -0.12);
  const elbowYaw = clampMotion(forearmReach * 1.2 + depth * 0.25, -1.25, 1.25);
  const elbowRoll = clampMotion(zSign * (forearmLift * 0.55 + forearmReach * 0.35), -1.1, 1.1);

  const wristPitch = clampMotion((forearmLift - lift) * 0.45, -0.75, 0.75);
  const wristYaw = clampMotion(forearmReach * 0.75, -0.8, 0.8);
  const wristRoll = clampMotion(zSign * (forearmReach * 0.45 + forearmLift * 0.25), -0.8, 0.8);

  return {
    [`${prefix}Shoulder`]: motionQuat(0.02 + lift * 0.05, reach * 0.22, zSign * (0.04 + reach * 0.18)),
    [`${prefix}UpperArm`]: motionQuat(shoulderPitch, shoulderYaw, shoulderRoll),
    [`${prefix}LowerArm`]: motionQuat(elbowPitch, elbowYaw, elbowRoll),
    [`${prefix}Hand`]: motionQuat(wristPitch, wristYaw, wristRoll),
  };
}

function getMotionFingerBone(side, fingerName, index) {
  const rightBone = MOTION_FINGER_BONES[`${fingerName}:${index}`];
  if (!rightBone) return "";
  return side === "left" ? rightBone.replace(/^right/, "left") : rightBone;
}

function handPoseFromLandmarks(hand, side = "right") {
  const tracks = {};
  for (const [fingerName, landmarks] of Object.entries(MOTION_FINGERS)) {
    const curls = computeMotionFingerCurl(hand, landmarks);
    const spread = computeMotionFingerSpread(hand, fingerName);
    for (let i = 0; i < 3; i++) {
      const bone = getMotionFingerBone(side, fingerName, i);
      if (!bone) continue;
      const raw = curls[i];
      if (fingerName === "Thumb") {
        const curl = clampMotion(raw * (i === 0 ? 0.8 : 0.9), 0, i === 0 ? 1.2 : 1.4);
        tracks[bone] = i === 0 ? motionQuat(curl, spread * 0.4, spread * 0.8) : motionQuat(curl, 0, 0);
      } else {
        const curl = clampMotion(raw, 0, 1.6);
        tracks[bone] = motionQuat(curl, 0, i === 0 ? spread * 0.4 : 0);
      }
    }
  }
  const wrist = hand[0], mid = hand[9], idx = hand[5];
  if (wrist && mid && idx) {
    const palm = { x: mid.x - wrist.x, y: mid.y - wrist.y };
    const lateral = { x: idx.x - mid.x, y: idx.y - mid.y };
    // Boost wrist rotation response so waving is highly visible on the avatar
    tracks[`${side}Hand`] = motionQuat(0, Math.atan2(palm.x, -palm.y) * 1.6, Math.atan2(lateral.y, lateral.x) * 1.2);
  }
  return tracks;
}

function computeMotionFingerCurl(points, landmarks) {
  const [mcpIdx, pipIdx, dipIdx, tipIdx] = landmarks;
  const wrist = points[0], mcp = points[mcpIdx], pip = points[pipIdx], dip = points[dipIdx], tip = points[tipIdx];
  return [
    angleBetween2D(vec2(wrist, mcp), vec2(mcp, pip)),
    angleBetween2D(vec2(mcp, pip), vec2(pip, dip)),
    angleBetween2D(vec2(pip, dip), vec2(dip, tip)),
  ];
}

function computeMotionFingerSpread(points, fingerName) {
  const landmarks = MOTION_FINGERS[fingerName];
  const wrist = points[0], mcp = points[landmarks[0]], tip = points[landmarks[3]];
  const base = vec2(wrist, mcp);
  const finger = vec2(mcp, tip);
  return clampMotion((base.x * finger.y - base.y * finger.x) * 2.0, -0.5, 0.5);
}

function vec2(a, b) {
  return { x: b.x - a.x, y: b.y - a.y };
}

function angleBetween2D(a, b) {
  const n1 = Math.hypot(a.x, a.y);
  const n2 = Math.hypot(b.x, b.y);
  if (n1 < 1e-8 || n2 < 1e-8) return 0;
  return Math.acos(clampMotion((a.x * b.x + a.y * b.y) / (n1 * n2), -1, 1));
}

async function previewMotionClip() {
  if (!_motion.lastClip) return;
  initMotionAvatar();
  const avatar = window.motionAvatar;
  if (!avatar) {
    showToast("Avatar preview is still loading.");
    return;
  }
  if (avatar._readyPromise) await avatar._readyPromise;
  avatar.animations[_motion.lastLabel] = _motion.lastClip;
  if (avatar.missingAnimations) avatar.missingAnimations.delete(_motion.lastLabel);
  avatar.queueSign(_motion.lastLabel);
}

async function saveMotionClip() {
  if (!_motion.lastClip) return;
  try {
    const res = await fetch("/api/avatar/animation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: _motion.lastLabel, clip: _motion.lastClip }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Save failed.");
    showToast(`Saved ${data.label}`);
    setMotionStatus("Saved");
  } catch (err) {
    console.warn("Motion save failed:", err);
    showToast(err.message || "Motion clip could not be saved.");
    setMotionStatus("Save failed");
  }
}

function downloadMotionClip() {
  if (!_motion.lastClip) return;
  const blob = new Blob([JSON.stringify(_motion.lastClip, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${_motion.lastLabel}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// ══ SPEECH-TO-TEXT TAB (Web Speech API — Chrome built-in) ══
let sttActive    = false;
let _recognition = null;

function setSTTStatus(text, active = sttActive) {
  const dot = document.getElementById("sttStatusIndicator");
  const dotText = document.getElementById("sttStatusText");
  if (dot) dot.classList.toggle("active", !!active);
  if (dotText) dotText.textContent = text;
}

function setTranscriptDisplay(text, state = "placeholder") {
  const el = document.getElementById("sttTranscriptDisplay");
  if (!el) return;
  el.textContent = "";
  const span = document.createElement("span");
  span.className = state === "placeholder" ? "stt-transcript-placeholder" : "stt-transcript-text";
  if (state === "interim") span.classList.add("is-interim");
  span.textContent = state === "placeholder" ? text : `"${text}"`;
  el.appendChild(span);
  if (state === "final") {
    el.classList.remove("stt-flash");
    void el.offsetWidth;
    el.classList.add("stt-flash");
  }
}

function setGlossDisplay(gloss) {
  const glossEl = document.getElementById("sttGlossDisplay");
  if (!glossEl) return;
  glossEl.textContent = gloss || "No interpreted gloss yet.";
  glossEl.classList.toggle("has-gloss", !!gloss);
}

async function processSpeechText(rawText, options = {}) {
  const finalTranscript = String(rawText || "").trim();
  if (!finalTranscript) return;

  setTranscriptDisplay(finalTranscript, "final");
  setSTTStatus("Interpreting speech...", sttActive);
  if (options.chime) softChime();

  try {
    const res = await fetch("/api/stt/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: finalTranscript }),
    });
    if (res.status === 401) {
      setSTTStatus("Session expired", false);
      showToast("Please log in again.");
      return;
    }
    if (!res.ok) throw new Error(`STT request failed: ${res.status}`);
    const d = await res.json();
    setGlossDisplay(d.gloss_order || "");
    if (d.guidance && d.guidance.length) renderGuidance(d.guidance);
    if (d.history  && d.history.length)  renderSTTHistory(d.history);
    setSTTStatus(d.guidance && d.guidance.length ? "Signing phrase" : "Listening...", sttActive);
  } catch (err) {
    console.warn("Speech interpretation failed:", err);
    setSTTStatus("Could not interpret speech", sttActive);
    showToast("Speech could not be interpreted. Try again.");
  }
}

function submitSpeechText() {
  const input = document.getElementById("sttManualInput");
  const text = input ? input.value.trim() : "";
  if (!text) {
    if (input) input.focus();
    return;
  }
  if (input) input.value = "";
  processSpeechText(text, { chime: false });
}

function toggleSTT() {
  const btn     = document.getElementById("sttMicBtn");
  const label   = document.getElementById("sttMicLabel");

  if (!sttActive) {
    if (!hasSafeMediaOrigin()) {
      showPermissionHelp("microphone", { name: "InsecureContextError" });
      return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      setSTTStatus("Speech recognition unavailable", false);
      showToast("Speech recognition requires Chrome or Edge. Type a phrase instead.");
      return;
    }
    _recognition = new SR();
    _recognition.continuous      = true;
    _recognition.interimResults  = true;
    _recognition.lang            = "en-US";

    _recognition.onresult = async (event) => {
      let interimTranscript = '';
      let finalTranscript = '';
      
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      if (finalTranscript) {
        processSpeechText(finalTranscript, { chime: true });
      } else if (interimTranscript) {
        setTranscriptDisplay(interimTranscript.trim(), "interim");
        setSTTStatus("Hearing speech...", true);
      }
    };

    _recognition.onerror = (e) => {
      if (["not-allowed", "service-not-allowed", "audio-capture"].includes(e.error)) {
        showPermissionHelp("microphone", e);
        _stopSTTUI();
        return;
      }
      if (e.error === "network") {
        _stopSTTUI("Chrome speech service unavailable");
        showToast("Chrome speech service is unavailable here. Type a phrase instead.");
        return;
      }
      if (e.error === "no-speech") {
        setSTTStatus("Listening...", true);
        return;
      }
      console.warn("Speech recognition error:", e.error);
      setSTTStatus(`Speech error: ${e.error || "unknown"}`, true);
    };

    // Chrome stops recognition after ~60 s silence; auto-restart while active
    _recognition.onend = () => {
      if (!sttActive || !_recognition) return;
      try { _recognition.start(); } catch (_) {}
    };

    try {
      _recognition.start();
      sttActive = true;
      if (btn) btn.classList.add("listening");
      if (label) label.textContent = "Stop Listening";
      setTranscriptDisplay("Listening...", "placeholder");
      setSTTStatus("Listening...", true);
    } catch (err) {
      console.warn("Speech recognition could not start:", err);
      _recognition = null;
      sttActive = false;
      setSTTStatus("Microphone could not start", false);
      showToast("Microphone could not start. Type a phrase instead.");
    }
  } else {
    _stopSTTUI();
  }
  if (typeof lucide !== "undefined") lucide.createIcons();
}

function _stopSTTUI(statusText = "Microphone off") {
  sttActive = false;
  if (_recognition) { _recognition.onend = null; _recognition.stop(); _recognition = null; }
  const btn     = document.getElementById("sttMicBtn");
  const label   = document.getElementById("sttMicLabel");
  if (btn)     btn.classList.remove("listening");
  if (label)   label.textContent   = "Start Listening";
  setSTTStatus(statusText, false);
}

function initializeAvatar() {
  if (!window.BridgeSignAvatar) return;
  window.BridgeSignAvatar.init("avatarContainer", {
    globalName: "appAvatar",
    loaderId: "avatarLoading",
    statusId: "sttStatusText",
    fallbackLabel: "Sign Preview",
    readyText: sttActive ? "Listening..." : "Avatar ready",
    activeNote: "Speech guidance is active. This panel is following the current sign sequence."
  });
}

function queueGuidanceOnAvatar(guidance, attempt = 0) {
  if (!window.appAvatar) initializeAvatar();

  const avatar = window.appAvatar;
  if (!avatar) {
    if (attempt < 50) {
      setTimeout(() => queueGuidanceOnAvatar(guidance, attempt + 1), 200);
    }
    return;
  }

  guidance.forEach(g => {
    if (g.sign_label) {
      avatar.queueSign(g.sign_label);
    } else if (g.type === "fingerspell") {
      for (const letter of String(g.word || "").toUpperCase()) {
        if (letter >= "A" && letter <= "Z") avatar.queueSign(letter);
      }
    }
  });
}

function renderGuidance(guidance) {
  const grid = document.getElementById("sttGuidanceGrid");
  grid.innerHTML = "";

  guidance.forEach((g, i) => {
    const card = document.createElement("div");
    card.className = `stt-guidance-item ${g.type}`;
    card.style.animationDelay = `${i * 0.06}s`;

    const icon = g.type === "gesture" ? "hand-metal"
               : g.type === "letter"  ? "type"
               : "keyboard";

    const badge = g.type === "gesture" ? "Gesture"
                : g.type === "letter"  ? "Letter"
                : "Fingerspell";

    const badgeClass = g.type === "gesture" ? "badge-gesture"
                     : g.type === "letter"  ? "badge-letter"
                     : "badge-fingerspell";

    const header = document.createElement("div");
    header.className = "stt-guidance-header";

    const iconEl = document.createElement("i");
    iconEl.setAttribute("data-lucide", icon);
    iconEl.setAttribute("width", "16");
    iconEl.setAttribute("height", "16");
    header.appendChild(iconEl);

    const word = document.createElement("span");
    word.className = "stt-guidance-word";
    word.textContent = g.word || "";
    header.appendChild(word);

    const badgeEl = document.createElement("span");
    badgeEl.className = `stt-guidance-badge ${badgeClass}`;
    badgeEl.textContent = badge;
    header.appendChild(badgeEl);

    card.appendChild(header);

    if (g.sign_label) {
      const imgName = "sign_" + g.sign_label.toLowerCase().replace(/_/g, "") + ".png";
      const imgSrc = "/static/signs/" + imgName;
      const imageWrap = document.createElement("div");
      imageWrap.className = "stt-guidance-image";
      const img = document.createElement("img");
      img.src = imgSrc;
      img.alt = g.sign_label;
      img.onerror = () => { imageWrap.style.display = "none"; };
      imageWrap.appendChild(img);
      card.appendChild(imageWrap);
    }

    const tip = document.createElement("div");
    tip.className = "stt-guidance-tip";
    tip.textContent = g.tip || "";
    card.appendChild(tip);

    grid.appendChild(card);
  });

  if (typeof lucide !== "undefined") lucide.createIcons();

  queueGuidanceOnAvatar(guidance);
}

function renderSTTHistory(history) {
  const list = document.getElementById("sttHistoryList");
  list.innerHTML = "";

  history.forEach((item, i) => {
    const row = document.createElement("div");
    row.className = "stt-history-item";
    const gestures = item.guidance ? item.guidance.filter(g => g.type === "gesture").length : 0;
    const total    = item.guidance ? item.guidance.length : 0;
    const time = document.createElement("span");
    time.className = "stt-history-time";
    time.textContent = item.ts || "";
    const text = document.createElement("span");
    text.className = "stt-history-text";
    text.textContent = `"${item.text || ""}"`;
    const stats = document.createElement("span");
    stats.className = "stt-history-stats";
    stats.textContent = `${gestures}/${total} signs`;
    row.append(time, text, stats);
    // Click to re-show guidance
    row.addEventListener("click", () => {
      if (item.guidance) renderGuidance(item.guidance);
      // Update transcript display too
      const el = document.getElementById("sttTranscriptDisplay");
      if (el) setTranscriptDisplay(item.text || "", "final");
      setGlossDisplay(item.gloss_order || "");
    });
    list.appendChild(row);
  });
}

// ── Init ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  if (typeof lucide !== "undefined") lucide.createIcons();
  
  // Try initializing client-side MediaPipe tracking
  initLocalHands();
  
  const activeTab = document.querySelector(".tab-btn.active")?.dataset.tab;
  if (activeTab === "speech") initializeAvatar();
  if (activeTab === "motion") initMotionRecorder();
  const manualInput = document.getElementById("sttManualInput");
  if (manualInput) {
    manualInput.addEventListener("keydown", e => {
      if (e.key === "Enter") submitSpeechText();
    });
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closePermissionHelp();
});

// ── Avatar Event Listeners ─────────────────────────────────
window.addEventListener("avatar-signing", (e) => {
  const { label, queueLength } = e.detail;
  const progressEl = document.getElementById("sttSigningProgress");
  if (progressEl) {
    progressEl.textContent = `Now signing: ${label} (${queueLength} remaining)`;
    progressEl.style.display = "block";
  }
  setSTTStatus(`Signing: ${String(label || "").replace(/_/g, " ")}`, true);
});

window.addEventListener("avatar-idle", () => {
  const progressEl = document.getElementById("sttSigningProgress");
  if (progressEl) {
    progressEl.style.display = "none";
  }
  setSTTStatus(sttActive ? "Listening..." : "Avatar ready", sttActive);
});

window.addEventListener("avatar-error", (e) => {
  const message = e.detail?.message || "Avatar could not load";
  setSTTStatus(message, false);
  showToast(message);
});
