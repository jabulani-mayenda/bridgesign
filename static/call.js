/* ── BridgeSign call.js ────────────────────────────────────────────
   Fixes in this version:
   1. Flip camera  – toggles front / back camera (phone-friendly)
   2. Sign-to-speech – deaf partner's signs are spoken aloud on hearing side
   3. Speech-to-sign – hearing partner's speech drives the avatar on deaf side
   4. Role-aware UI labels and avatar status messages
─────────────────────────────────────────────────────────────────── */

const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
const wsUrl = `${wsScheme}://${window.location.host}/ws/call/${window.ROOM_ID}`;
const PERF_CONFIG = window.BRIDGESIGN_PERF || {};
const CALL_PERF = PERF_CONFIG.call || {};
const SPEECH_PERF = PERF_CONFIG.speech || PERF_CONFIG.voice || {};

let ws;
let pc;
let dataChannel;
let localStream;
let role = "deaf"; // deaf | hearing
let isMuted = false;
let cameraOff = false;
let pendingDataMessages = [];

// ── Camera facing mode (default: environment = back camera for phones) ──
let facingMode = "environment";

let _recognition = null;
let _inferInterval = null;
let _inferInFlight = false;
let _inferSeq = 0;
let _lastAppliedInferSeq = 0;
let lastSign = "";
let lastSignTime = 0;
let lastInterimSent = "";
let lastInterimSentAt = 0;
let lastAvatarSpeechText = "";

let _localHands = null;
let useClientInference = false;
let normalizeFrontCameraLandmarks = true;

const signCache = new Map();
const avatarQueue = [];
let avatarQueueRunning = false;
const speechQueue = [];
let speechQueueRunning = false;
let callWordModule = null;
let _inferFrameCounter = 0;

const INFER_INTERVAL_MS = Number(CALL_PERF.inferIntervalMs || 140);
const INFER_FRAME_SKIP = Math.max(1, Number(CALL_PERF.frameSkip || 1));
const INFER_SLOW_WARN_MS = Number(CALL_PERF.slowWarnMs || 100);
const INFER_HARD_TIMEOUT_MS = Number(CALL_PERF.hardTimeoutMs || 900);
const SERVER_FRAME_MAX_DIM = Number(CALL_PERF.serverFrameMaxDim || 320);
const SERVER_JPEG_QUALITY = Number(CALL_PERF.jpegQuality || 0.72);
const SIGN_REPEAT_MS = 900;
const SPEECH_PARTIAL_SEND_MS = Number(CALL_PERF.speechPartialSendMs || SPEECH_PERF.interimDebounceMs || 500);
const wordModuleConfig = {
  WORD_PAUSE_MS: 400,
  MIN_WORD_LENGTH: 1,
  AUTO_SPACE: true,
  ENABLE_WORD_MAPPING: true,
  CASE_SENSITIVE: false,
  DUPLICATE_SIGN_MS: 850,
  IDLE_CLEAR_MS: 5000
};
const SIGN_WORD_MAP = {
  I: "I",
  A: "A",
  LOVE: "love",
  YOU: "you",
  ILOVEYOU: "I love you",
  THANKYOU: "thank you",
  THANK_YOU: "thank you",
  HELLO: "hello",
  HELP: "help",
  YES: "yes",
  NO: "no",
  PLEASE: "please",
  SORRY: "sorry",
  STOP: "stop"
};
const ICE_SERVERS = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "stun:stun1.l.google.com:19302" },
    { urls: "stun:stun2.l.google.com:19302" },
    { urls: "stun:stun3.l.google.com:19302" },
    { urls: "stun:stun4.l.google.com:19302" }
  ]
};

function numberOption(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function callConsecutiveThreshold() {
  return Math.max(2, numberOption(CALL_PERF.consecutiveThreshold, 2));
}

function isFirstConfirmedFrame(data) {
  const threshold = Number(data?.consecutive_threshold || 0);
  const consecutive = Number(data?.consecutive || 0);
  return !threshold || !consecutive || consecutive === threshold;
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

// ── Initialize MediaPipe Hands locally if available ──
function initLocalHands() {
  if (typeof Hands !== "undefined") {
    try {
      _localHands = new Hands({
        locateFile: (file) => `/static/lib/mediapipe/${file}`
      });
      const mediaPipeOptions = CALL_PERF.mediaPipe || {};
      _localHands.setOptions({
        maxNumHands: numberOption(mediaPipeOptions.maxNumHands, 1),
        modelComplexity: numberOption(mediaPipeOptions.modelComplexity, 0),
        minDetectionConfidence: numberOption(mediaPipeOptions.minDetectionConfidence, 0.55),
        minTrackingConfidence: numberOption(mediaPipeOptions.minTrackingConfidence, 0.55)
      });
      _localHands.onResults(handleLocalHandsResults);
      useClientInference = true;
      console.log("[Call] Client-side landmark inference active for call rooms! 🚀");
    } catch (err) {
      console.warn("[Call] Failed to initialize local Hands, using server fallback:", err);
      useClientInference = false;
    }
  }
}

// Handle results from local MediaPipe Hands
async function handleLocalHandsResults(results) {
  if (!localStream || cameraOff || role !== "deaf") {
    _inferInFlight = false;
    return;
  }

  const video = els.localVideo;
  if (!video || !video.videoWidth) {
    _inferInFlight = false;
    return;
  }

  let landmarksList = null;
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const hand = results.multiHandLandmarks[0];
    const w = video.videoWidth;
    const h = video.videoHeight;
    
    landmarksList = hand.map((lm, idx) => [
      idx,
      Math.round(normalizeLandmarkX(lm.x) * w),
      Math.round(lm.y * h)
    ]);
  }

  try {
    await sendLocalLandmarks(landmarksList);
  } finally {
    _inferInFlight = false;
  }
}

// Send local landmarks to the landmarks infer endpoint
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
        consecutive_threshold: callConsecutiveThreshold()
      }),
      signal: controller.signal
    });
    const elapsed = performance.now() - startedAt;
    if (elapsed > INFER_SLOW_WARN_MS) {
      console.debug(`[Call] slow landmark request ${Math.round(elapsed)}ms`);
    }
    const data = await res.json().catch(() => ({}));
    data._clientSeq = seq;
    if (res.ok) {
      handleInferResponse(data);
    }
  } catch (err) {
    if (err?.name === "AbortError") {
      console.warn(`[Call] Landmarks request timed out after ${INFER_HARD_TIMEOUT_MS}ms; dropping frame.`);
    } else {
      console.error("[Call] Landmarks infer error:", err);
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

const els = {};

function $(id) {
  return document.getElementById(id);
}

function cacheElements() {
  [
    "roleSelect", "localVideo", "remoteVideo", "callStatus", "captionBar",
    "connectionState", "muteBtn", "cameraBtn", "flipCameraBtn", "endBtn",
    "copyLinkBtn", "typedMessageInput", "sendTextBtn", "callTranscript",
    "callAvatarStatus"
  ].forEach(id => { els[id] = $(id); });
}

function setStatus(text, visible = true) {
  if (!els.callStatus) return;
  els.callStatus.textContent = text;
  els.callStatus.style.display = visible ? "block" : "none";
}

function setConnection(text) {
  if (els.connectionState) els.connectionState.textContent = text;
}

function setCaption(text, color = "#fff") {
  if (!els.captionBar) return;
  els.captionBar.textContent = text || "Waiting...";
  els.captionBar.style.color = color;
  els.captionBar.style.transform = "translateX(-50%) scale(1.04)";
  setTimeout(() => {
    if (els.captionBar) els.captionBar.style.transform = "translateX(-50%) scale(1)";
  }, 180);
}

function setTranscript(text) {
  if (els.callTranscript) els.callTranscript.textContent = text || "Waiting for captions...";
}

function setAvatarStatus(text) {
  if (els.callAvatarStatus) els.callAvatarStatus.textContent = text || "Avatar ready";
}

function normalizeSpokenSign(text) {
  const clean = String(text || "").replace(/_/g, " ").trim();
  if (!clean) return "";
  if (/^[A-Z]$/.test(clean)) return clean;
  return clean;
}

// ── Web Speech API speak helper (browser TTS) ──────────────────────
function speakNow(text) {
  if (!text || !window.speechSynthesis) return;
  const normalized = text === text.toUpperCase()
    ? text.toLowerCase().replace(/\b\w/g, c => c.toUpperCase())
    : text;
  const utt = new SpeechSynthesisUtterance(normalized);
  const baseRate = numberOption(SPEECH_PERF.rate, 1.15);
  utt.rate = baseRate;
  utt.pitch = 1;
  utt.volume = 1;
  const voices = window.speechSynthesis.getVoices();
  const pref = voices.find(v => v.lang.startsWith("en") && v.name.includes("Female"))
            || voices.find(v => v.lang.startsWith("en"))
            || voices[0];
  if (pref) utt.voice = pref;
  window.speechSynthesis.speak(utt);
}
if (window.speechSynthesis) window.speechSynthesis.onvoiceschanged = () => {};

function speak(text) {
  speakSign(text);
}

function speakSign(label) {
  const spoken = normalizeSpokenSign(label);
  if (!spoken || !window.speechSynthesis) return;
  console.log(`[Call] Received sign: "${label}" -> speaking "${spoken}" via TTS`);
  speechQueue.push(spoken);
  runSpeechQueue();
}

function runSpeechQueue() {
  if (speechQueueRunning || !window.speechSynthesis) return;
  const next = speechQueue.shift();
  if (!next) return;

  speechQueueRunning = true;
  const utt = new SpeechSynthesisUtterance(next);
  const baseRate = numberOption(SPEECH_PERF.rate, 1.15);
  utt.rate = /^[A-Z]$/.test(next) ? baseRate * 0.85 : baseRate;
  utt.pitch = 1;
  utt.volume = 1;
  const voices = window.speechSynthesis.getVoices();
  const pref = voices.find(v => v.lang.startsWith("en") && v.name.includes("Female"))
            || voices.find(v => v.lang.startsWith("en"))
            || voices[0];
  if (pref) utt.voice = pref;
  utt.onend = () => {
    speechQueueRunning = false;
    runSpeechQueue();
  };
  utt.onerror = () => {
    speechQueueRunning = false;
    runSpeechQueue();
  };
  window.speechSynthesis.speak(utt);
}

// ── Avatar helpers ─────────────────────────────────────────────────
function initCallAvatar() {
  if (!window.BridgeSignAvatar) return;
  window.BridgeSignAvatar.init("callAvatarContainer", {
    globalName: "callAvatar",
    loaderId: "callAvatarLoading",
    statusId: "callAvatarStatus",
    fallbackLabel: "Call Avatar",
    readyText: "Avatar ready",
    activeNote: "Following the latest call message."
  });
}

function queueGuidanceOnAvatar(guidance) {
  const avatar = window.callAvatar;
  if (!avatar || !Array.isArray(guidance)) return;

  guidance.forEach(item => {
    if (item.sign_label) {
      avatar.queueSign(item.sign_label);
      return;
    }
    if (item.type === "fingerspell" && item.word) {
      if (typeof avatar.queueLetters === "function") avatar.queueLetters(item.word);
      else avatar.queueText(item.word);
    }
  });
}

async function playTextOnAvatar(text, options = {}) {
  const cleanText = String(text || "").trim();
  if (!cleanText) return;

  let phrase = cleanText;
  if (options.partial) {
    phrase = incrementalSpeechChunk(cleanText);
    if (!phrase) return;
  } else {
    if (lastAvatarSpeechText) {
      phrase = incrementalSpeechChunk(cleanText) || "";
    }
    lastAvatarSpeechText = "";
    if (!phrase) return;
  }

  console.log(`[Call] Received speech: "${cleanText}" -> queueing signs for avatar`);
  avatarQueue.push(phrase);
  runAvatarQueue();
}

function incrementalSpeechChunk(text) {
  const clean = normalizeCacheKey(text);
  const previous = normalizeCacheKey(lastAvatarSpeechText);
  lastAvatarSpeechText = text;
  if (!previous) return text;
  if (clean === previous) return "";
  if (clean.startsWith(previous + " ")) {
    return text.slice(previous.length).trim();
  }
  return text;
}

async function runAvatarQueue() {
  if (avatarQueueRunning) return;
  const phrase = avatarQueue.shift();
  if (!phrase) return;

  avatarQueueRunning = true;
  initCallAvatar();
  const avatar = window.callAvatar;
  console.log(`[Call] Avatar animation started for phrase: "${phrase}"`);

  if (!avatar) {
    avatarQueueRunning = false;
    setTimeout(runAvatarQueue, 150);
    return;
  }

  try {
    const guidance = await getGuidanceForText(phrase);
    if (guidance && guidance.length) {
      queueGuidanceOnAvatar(guidance);
    } else if (typeof avatar.queueText === "function") {
      avatar.queueText(phrase);
    }
  } catch (err) {
    console.warn("[Call] Avatar guidance failed:", err);
    if (typeof avatar.queueText === "function") avatar.queueText(phrase);
  }

  const waitMs = Math.min(6000, Math.max(900, phrase.split(/\s+/).length * 850));
  setTimeout(() => {
    console.log("[Call] Avatar animation complete");
    avatarQueueRunning = false;
    runAvatarQueue();
  }, waitMs);
}

async function getGuidanceForText(text) {
  const key = normalizeCacheKey(text);
  if (signCache.has(key)) return signCache.get(key);

  const startedAt = performance.now();
  const res = await fetch("/api/stt/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  const elapsed = performance.now() - startedAt;
  if (elapsed > 100) {
    console.debug(`[Call] /api/stt/text slow ${Math.round(elapsed)}ms for "${text}"`);
  }
  if (!res.ok) return null;

  const data = await res.json();
  const guidance = Array.isArray(data.guidance) ? data.guidance : [];
  signCache.set(key, guidance);
  return guidance;
}

function normalizeCacheKey(text) {
  return String(text || "").trim().toLowerCase().replace(/\s+/g, " ");
}

class WordModule {
  constructor(config = wordModuleConfig) {
    this.config = config;
    this.currentWordBuffer = "";
    this.confirmedWords = [];
    this.lastSign = "";
    this.lastSignAt = 0;
    this.lastInputAt = 0;
  }

  pushSign(sign, now = Date.now()) {
    const clean = String(sign || "").replace(/\s+/g, "_").trim().toUpperCase();
    if (!clean) return this.state();

    if (
      clean === this.lastSign
      && now - this.lastSignAt < this.config.DUPLICATE_SIGN_MS
    ) {
      return this.state();
    }

    if (this.lastInputAt && now - this.lastInputAt >= this.config.WORD_PAUSE_MS) {
      this.commitCurrentWord(now - this.lastInputAt);
    }

    if (clean.length > 1) {
      this.commitCurrentWord(0);
      this.confirmedWords.push(this.mapWord(clean));
      this.lastSign = clean;
      this.lastSignAt = now;
      this.lastInputAt = now;
      console.log(`[WordModule] Pause detected -> confirmed word: "${this.confirmedWords.at(-1)}"`);
      console.log(`[WordModule] Final phrase: "${this.finalPhrase()}"`);
      return this.state({ completedWord: this.confirmedWords.at(-1) });
    }

    this.currentWordBuffer += clean;
    this.lastSign = clean;
    this.lastSignAt = now;
    this.lastInputAt = now;
    console.log(`[WordModule] Buffered: "${this.currentWordBuffer}" (pause: false)`);
    return this.state();
  }

  tick(handPresent, now = Date.now()) {
    if (
      !handPresent
      && this.currentWordBuffer
      && this.lastInputAt
      && now - this.lastInputAt >= this.config.WORD_PAUSE_MS
    ) {
      const completedWord = this.commitCurrentWord(now - this.lastInputAt);
      return this.state({ completedWord });
    }
    if (
      !handPresent
      && !this.currentWordBuffer
      && this.lastInputAt
      && now - this.lastInputAt >= this.config.IDLE_CLEAR_MS
    ) {
      this.lastSign = "";
    }
    return this.state();
  }

  commitCurrentWord(pauseMs = 0) {
    if (this.currentWordBuffer.length < this.config.MIN_WORD_LENGTH) return "";
    const word = this.mapWord(this.currentWordBuffer);
    this.currentWordBuffer = "";
    if (word) this.confirmedWords.push(word);
    console.log(`[WordModule] Pause detected (${Math.round(pauseMs)}ms) -> confirmed word: "${word}"`);
    console.log(`[WordModule] Final phrase: "${this.finalPhrase()}"`);
    return word;
  }

  mapWord(raw) {
    const clean = String(raw || "").replace(/\s+/g, "").toUpperCase();
    const mapped = this.config.ENABLE_WORD_MAPPING ? SIGN_WORD_MAP[clean] : "";
    if (mapped) return mapped;
    return this.config.CASE_SENSITIVE ? raw : clean;
  }

  undoWord() {
    return this.confirmedWords.pop() || "";
  }

  clearWord() {
    this.currentWordBuffer = "";
  }

  finalPhrase() {
    return this.confirmedWords.join(this.config.AUTO_SPACE ? " " : "");
  }

  state(extra = {}) {
    return {
      currentWordBuffer: this.currentWordBuffer,
      confirmedWords: [...this.confirmedWords],
      finalPhrase: this.finalPhrase(),
      ...extra
    };
  }
}

// ── Main startup ───────────────────────────────────────────────────
async function startCall() {
  const rateSetting = numberOption(SPEECH_PERF.rate, 1.15);
  const thresholdSetting = callConsecutiveThreshold();
  console.log(`[Perf] Voice rate: ${rateSetting}, Consecutive: ${thresholdSetting}`);

  cacheElements();
  bindControls();
  initCallAvatar();
  callWordModule = new WordModule();
  
  // Try initializing client-side MediaPipe tracking
  initLocalHands();
  
  setStatus("Starting camera...");

  await acquireCamera();
  updateMediaButtons();
  connectWebSocket();
  updateRoleLogic();
}

// ── Camera acquisition (respects facingMode) ───────────────────────
async function acquireCamera() {
  const constraints = {
    video: { width: 960, height: 540, facingMode },
    audio: true
  };

  try {
    localStream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    try {
      localStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode },
        audio: false
      });
      isMuted = true;
      setStatus("Microphone unavailable. Video call started without audio.");
    } catch (fallbackErr) {
      console.error("Media access error:", fallbackErr);
      setStatus("Camera access failed. Check browser permissions.");
      return;
    }
  }

  if (els.localVideo) {
    // Mirror the preview only when using front camera
    els.localVideo.style.transform = facingMode === "user" ? "scaleX(-1)" : "none";
    els.localVideo.srcObject = localStream;
    await els.localVideo.play().catch(() => {});
  }
}

// ── Flip camera (toggle front ↔ back) ─────────────────────────────
async function flipCamera() {
  const btn = els.flipCameraBtn;
  if (btn) btn.disabled = true;

  // Stop current tracks
  if (localStream) localStream.getVideoTracks().forEach(t => t.stop());

  // Switch facing mode
  facingMode = facingMode === "environment" ? "user" : "environment";

  try {
    const newStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode },
      audio: false
    });

    // Replace audio tracks into the new stream
    if (localStream) {
      localStream.getAudioTracks().forEach(t => newStream.addTrack(t));
    }
    localStream = newStream;

    // Update local preview
    if (els.localVideo) {
      els.localVideo.style.transform = facingMode === "user" ? "scaleX(-1)" : "none";
      els.localVideo.srcObject = localStream;
      await els.localVideo.play().catch(() => {});
    }

    // Replace video sender on the peer connection
    if (pc) {
      const newVideoTrack = localStream.getVideoTracks()[0];
      if (newVideoTrack) {
        const sender = pc.getSenders().find(s => s.track && s.track.kind === "video");
        if (sender) await sender.replaceTrack(newVideoTrack);
      }
    }

    // Update canvas flip state and inference
    cameraOff = false;
    updateMediaButtons();
  } catch (err) {
    console.error("Flip camera failed:", err);
    // Revert facing mode
    facingMode = facingMode === "environment" ? "user" : "environment";
    setStatus("Could not switch camera.");
  }

  if (btn) btn.disabled = false;
  updateFlipBtn();
}

function updateFlipBtn() {
  const btn = els.flipCameraBtn;
  if (!btn) return;
  const label = facingMode === "environment" ? "Front Cam" : "Back Cam";
  btn.innerHTML = `<i data-lucide="refresh-cw" width="16" height="16"></i> ${label}`;
  if (typeof lucide !== "undefined") lucide.createIcons();
}

// ── Controls wiring ────────────────────────────────────────────────
function bindControls() {
  if (els.roleSelect) {
    role = els.roleSelect.value;
    els.roleSelect.addEventListener("change", (e) => {
      role = e.target.value;
      updateRoleLogic();
    });
  }

  if (els.muteBtn)        els.muteBtn.addEventListener("click", toggleMute);
  if (els.cameraBtn)      els.cameraBtn.addEventListener("click", toggleCamera);
  if (els.flipCameraBtn)  els.flipCameraBtn.addEventListener("click", flipCamera);
  if (els.endBtn)         els.endBtn.addEventListener("click", endCall);
  if (els.copyLinkBtn)    els.copyLinkBtn.addEventListener("click", copyLink);
  if (els.sendTextBtn)    els.sendTextBtn.addEventListener("click", sendTypedMessage);
  if (els.typedMessageInput) {
    els.typedMessageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendTypedMessage();
    });
  }

  updateFlipBtn();
}

// ── WebSocket / WebRTC ─────────────────────────────────────────────
function connectWebSocket() {
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    setConnection(`Room ${window.ROOM_ID} - waiting for partner`);
  };

  ws.onmessage = async (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === "ready") {
      const shouldOffer = Boolean(msg.should_offer);
      setStatus(shouldOffer ? "Partner joined. Starting call..." : "Partner joined. Waiting for connection...");
      if (!pc) createPeerConnection(shouldOffer);
      return;
    }

    if (msg.type === "peer_left") {
      handlePeerLeft();
      return;
    }

    if (msg.type === "error") {
      setStatus(msg.message || "Call room error.");
      return;
    }

    if (msg.type === "offer") {
      if (!pc) createPeerConnection(false);
      await pc.setRemoteDescription(new RTCSessionDescription(msg.offer));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      ws.send(JSON.stringify({ type: "answer", answer }));
      return;
    }

    if (msg.type === "answer") {
      if (pc) await pc.setRemoteDescription(new RTCSessionDescription(msg.answer));
      return;
    }

    if (msg.type === "candidate" && pc) {
      await pc.addIceCandidate(new RTCIceCandidate(msg.candidate));
    }
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    setStatus("Signaling error. Retry the room link.");
  };

  ws.onclose = () => {
    if (!pc || pc.connectionState !== "connected") {
      setConnection("Disconnected from room");
      setStatus("Disconnected from the call room.");
    }
  };
}

function createPeerConnection(isOfferer) {
  if (pc) return pc;

  pc = new RTCPeerConnection(ICE_SERVERS);

  if (localStream) {
    localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
  }

  pc.ontrack = (event) => {
    if (els.remoteVideo) els.remoteVideo.srcObject = event.streams[0];
    setStatus("", false);
    setConnection("Connected");
  };

  pc.onconnectionstatechange = () => {
    if (!pc) return;
    if (["connected", "completed"].includes(pc.connectionState)) {
      setStatus("", false);
      setConnection("Connected");
    } else if (["failed", "disconnected"].includes(pc.connectionState)) {
      setStatus("Connection lost. Waiting for recovery...");
      setConnection("Connection unstable");
    } else if (pc.connectionState === "closed") {
      setConnection("Call ended");
    }
  };

  pc.onicecandidate = (event) => {
    if (event.candidate && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "candidate", candidate: event.candidate }));
    }
  };

  if (isOfferer) {
    dataChannel = pc.createDataChannel("bridgesign-captions");
    setupDataChannel();
    createOffer();
  } else {
    pc.ondatachannel = (event) => {
      dataChannel = event.channel;
      setupDataChannel();
    };
  }

  return pc;
}

async function createOffer() {
  try {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    ws.send(JSON.stringify({ type: "offer", offer: pc.localDescription }));
  } catch (err) {
    console.error("Offer creation failed:", err);
    setStatus("Could not start the call offer.");
  }
}

function setupDataChannel() {
  dataChannel.onopen = () => {
    pendingDataMessages.forEach(payload => dataChannel.send(JSON.stringify(payload)));
    pendingDataMessages = [];
  };

  dataChannel.onmessage = (event) => {
    try {
      handleDataMessage(JSON.parse(event.data));
    } catch (err) {
      console.warn("Bad data channel message:", err);
    }
  };
}

// ── Data channel message handler ───────────────────────────────────
// FIX #2: speak received signs aloud (sign-to-speech)
// FIX #3: play avatar for received speech (speech-to-sign)
function handleDataMessage(data) {
  if (data.type === "sign") {
    const label = data.label || "";
    setCaption(label, "var(--terracotta)");
    setTranscript(`Partner signed: ${label}`);

    if (role === "hearing") {
      speakSign(label);
    }

    if (role === "deaf") {
      const avatar = window.callAvatar;
      if (avatar && label) {
        try { avatar.queueSign(label.toUpperCase().replace(/\s+/g, "_")); } catch (_) {}
      }
    }
    return;
  }

  if (data.type === "speech") {
    const text = data.text || "";
    const partial = Boolean(data.partial);
    setCaption(`"${text}"`, "var(--honey)");
    setTranscript(text);
    console.log(`[Call] Received ${partial ? "partial " : ""}speech: "${text}" -> queueing signs for avatar`);

    if (role === "deaf") {
      playTextOnAvatar(text, { partial });
    }

    return;
  }

  if (data.type === "text") {
    const text = data.text || "";
    setCaption(text, "var(--sage)");
    setTranscript(text);
    console.log(`[Call] Received text: "${text}" -> ${role === "deaf" ? "queueing avatar" : "speaking"}`);

    if (role === "hearing") {
      speakSign(text);
    } else {
      playTextOnAvatar(text);
    }
  }
}

function sendData(payload) {
  if (dataChannel && dataChannel.readyState === "open") {
    dataChannel.send(JSON.stringify(payload));
    return true;
  }
  pendingDataMessages.push(payload);
  return false;
}

function sendCaption(type, content, extra = {}) {
  const clean = String(content || "").trim();
  if (!clean) return;

  const payload = type === "sign"
    ? { type: "sign", label: clean, ...extra }
    : { type, text: clean, ...extra };

  sendData(payload);

  if (type === "sign") {
    setCaption(clean, "var(--sage)");
    setTranscript(`You signed: ${clean}`);
  } else {
    setCaption(type === "speech" ? `"${clean}"` : clean, "var(--sage)");
    setTranscript(clean);
    if (role === "hearing" && (type === "speech" || type === "text") && !extra.partial) {
      setAvatarStatus("Previewing your speech as signs");
      playTextOnAvatar(clean);
    }
  }
}

// ── Role logic: deaf = sign inference, hearing = speech recognition ─
function updateRoleLogic() {
  stopInferLoop();
  stopSpeechRecognition();

  if (role === "deaf") {
    // I am signing → run hand inference, my signs go to the hearing partner as speech
    startInferLoop();
    setAvatarStatus("Partner speech signs here");
    setTranscript("Camera signs will be sent to your partner as text and speech.");
    setCaption("Sign when ready - partner will hear you", "rgba(255,255,255,.78)");
    return;
  }

  // I am the hearing partner → use my microphone, my speech drives the deaf side avatar
  startSpeechRecognition();
  setAvatarStatus("Your speech previews here");
  setTranscript("Speech recognition is listening. Your speech will sign on your partner's avatar.");
  setCaption("Speak - partner's avatar will sign it", "rgba(255,255,255,.78)");
}

// ── Sign inference loop (deaf role) ───────────────────────────────
function startInferLoop() {
  if (_inferInterval) return;
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
  if (_inferInFlight || !localStream || cameraOff) return;
  const video = els.localVideo;
  if (!video || !video.videoWidth) return;
  _inferFrameCounter = (_inferFrameCounter + 1) % INFER_FRAME_SKIP;
  if (_inferFrameCounter !== 0) return;

  if (useClientInference && _localHands) {
    _inferInFlight = true;
    try {
      await _localHands.send({ image: video });
    } catch (err) {
      console.warn("[Call] Local Hands send error:", err);
      _inferInFlight = false;
    }
  } else {
    const canvas = $("inferCanvas");
    if (!canvas) return;
    drawVideoFrame(video, canvas, SERVER_FRAME_MAX_DIM);

    _inferInFlight = true;
    const startedAt = performance.now();
    canvas.toBlob(async (blob) => {
      if (!blob) {
        _inferInFlight = false;
        return;
      }

      const fd = new FormData();
      fd.append("frame", blob, "frame.jpg");
      fd.append("consecutive_threshold", callConsecutiveThreshold());
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), INFER_HARD_TIMEOUT_MS);
      try {
        const res = await fetch("/api/infer_frame", {
          method: "POST",
          body: fd,
          signal: controller.signal
        });
        const elapsed = performance.now() - startedAt;
        if (elapsed > INFER_SLOW_WARN_MS) {
          console.debug(`[Call] slow frame request ${Math.round(elapsed)}ms`);
        }
        const data = await res.json().catch(() => ({}));
        data._clientSeq = ++_inferSeq;
        if (res.ok) handleInferResponse(data);
      } catch (err) {
        if (err?.name === "AbortError") {
          console.warn(`[Call] frame request timed out after ${INFER_HARD_TIMEOUT_MS}ms; dropping frame.`);
        }
      } finally {
        clearTimeout(timeoutId);
      }
      _inferInFlight = false;
    }, "image/jpeg", SERVER_JPEG_QUALITY);
  }
}

function handleInferResponse(data) {
  const seq = Number(data._clientSeq || 0);
  if (seq && seq < _lastAppliedInferSeq) {
    console.debug(`[Call] stale inference response ignored seq=${seq}, latest=${_lastAppliedInferSeq}`);
    return;
  }
  if (seq) _lastAppliedInferSeq = seq;

  if (data.hand_state === "no_hand" && callWordModule) {
    const state = callWordModule.tick(false);
    if (state.completedWord) {
      sendCaption("sign", state.completedWord, { unit: "word" });
    }
    return;
  }

  const completed = data.completed_sentence || data.completed_word || "";
  const label = completed || data.label || "";
  if (!label || data.hand_state !== "recognised") return;
  if (!completed && !isFirstConfirmedFrame(data)) return;

  const now = Date.now();
  if (label === lastSign && now - lastSignTime < SIGN_REPEAT_MS) return;

  lastSign = label;
  lastSignTime = now;

  if (completed || !callWordModule) {
    sendCaption("sign", label, { confidence: data.confidence || 0, unit: completed ? "word" : "sign" });
    return;
  }

  if (["UNDO", "BACKSPACE", "DELETE", "WAVE_LEFT"].includes(String(label).toUpperCase())) {
    const removed = callWordModule.undoWord();
    setTranscript(removed ? `Removed word: ${removed}` : "No word to undo");
    console.log(`[WordModule] Undo gesture -> removed word: "${removed}"`);
    return;
  }

  const state = callWordModule.pushSign(label, now);
  setTranscript(state.currentWordBuffer
    ? `Building word: ${state.currentWordBuffer}`
    : state.finalPhrase || "Signing...");

  if (state.completedWord) {
    sendCaption("sign", state.completedWord, { confidence: data.confidence || 0, unit: "word" });
  }
}

// ── Speech recognition (hearing role) ─────────────────────────────
function startSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    setTranscript("Speech recognition is not supported in this browser.");
    return;
  }

  _recognition = new SR();
  _recognition.continuous = true;
  _recognition.interimResults = true;
  _recognition.lang = "en-US";

  _recognition.onresult = (event) => {
    let interimTranscript = "";
    let finalTranscript = "";

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript;
      else interimTranscript += event.results[i][0].transcript;
    }

    if (finalTranscript) {
      const cleanFinal = finalTranscript.trim();
      lastInterimSent = "";
      lastInterimSentAt = 0;
      console.log(`[Call] Speech final: "${cleanFinal}" -> sending to partner`);
      sendCaption("speech", cleanFinal, { partial: false });
    } else if (interimTranscript) {
      const cleanInterim = interimTranscript.trim();
      setCaption(`"${cleanInterim}"`, "rgba(255,255,255,.72)");
      setTranscript(cleanInterim);

      const now = Date.now();
      if (
        cleanInterim
        && cleanInterim !== lastInterimSent
        && now - lastInterimSentAt >= SPEECH_PARTIAL_SEND_MS
      ) {
        lastInterimSent = cleanInterim;
        lastInterimSentAt = now;
        console.log(`[Call] Speech partial: "${cleanInterim}" -> sending to partner`);
        sendData({ type: "speech", text: cleanInterim, partial: true });
      }
    }
  };

  _recognition.onerror = (event) => {
    if (["not-allowed", "service-not-allowed", "audio-capture"].includes(event.error)) {
      setTranscript("Microphone permission is blocked.");
      return;
    }
    console.warn("Speech recognition error:", event.error);
  };

  _recognition.onend = () => {
    if (role === "hearing" && _recognition) {
      try { _recognition.start(); } catch (_) {}
    }
  };

  try {
    _recognition.start();
  } catch (err) {
    console.warn("Speech recognition could not start:", err);
  }
}

function stopSpeechRecognition() {
  if (!_recognition) return;
  _recognition.onend = null;
  try { _recognition.stop(); } catch (_) {}
  _recognition = null;
}

// ── Media toggles ──────────────────────────────────────────────────
function toggleMute() {
  if (!localStream) return;
  isMuted = !isMuted;
  localStream.getAudioTracks().forEach(track => { track.enabled = !isMuted; });
  updateMediaButtons();
}

function toggleCamera() {
  if (!localStream) return;
  cameraOff = !cameraOff;
  localStream.getVideoTracks().forEach(track => { track.enabled = !cameraOff; });
  if (els.localVideo) els.localVideo.style.opacity = cameraOff ? "0.35" : "1";
  updateMediaButtons();
}

function updateMediaButtons() {
  if (els.muteBtn) {
    els.muteBtn.innerHTML = isMuted
      ? '<i data-lucide="mic-off" width="16" height="16"></i> Unmute'
      : '<i data-lucide="mic" width="16" height="16"></i> Mute';
  }
  if (els.cameraBtn) {
    els.cameraBtn.innerHTML = cameraOff
      ? '<i data-lucide="video-off" width="16" height="16"></i> Camera On'
      : '<i data-lucide="video" width="16" height="16"></i> Camera Off';
  }
  if (typeof lucide !== "undefined") lucide.createIcons();
}

// ── Typed message ──────────────────────────────────────────────────
function sendTypedMessage() {
  const input = els.typedMessageInput;
  const text = input ? input.value.trim() : "";
  if (!text) return;
  sendCaption("text", text);
  if (input) input.value = "";
}

// ── Copy link ──────────────────────────────────────────────────────
async function copyLink() {
  const btn = els.copyLinkBtn;
  try {
    await navigator.clipboard.writeText(window.location.href);
    if (btn) btn.innerHTML = '<i data-lucide="check" width="16" height="16"></i> Copied';
  } catch (_) {
    if (btn) btn.textContent = window.ROOM_ID;
  }

  if (typeof lucide !== "undefined") lucide.createIcons();
  setTimeout(() => {
    if (!btn) return;
    btn.innerHTML = '<i data-lucide="copy" width="16" height="16"></i> Copy Link';
    if (typeof lucide !== "undefined") lucide.createIcons();
  }, 1800);
}

// ── Partner left ───────────────────────────────────────────────────
function handlePeerLeft() {
  setStatus("Partner left. Waiting...");
  setConnection(`Room ${window.ROOM_ID} - waiting for partner`);
  setCaption("Waiting...", "rgba(255,255,255,.78)");
  setTranscript("Partner left the room.");
  if (dataChannel) dataChannel.close();
  dataChannel = null;
  if (pc) pc.close();
  pc = null;
  if (els.remoteVideo) els.remoteVideo.srcObject = null;
}

// ── End call ───────────────────────────────────────────────────────
function endCall() {
  stopInferLoop();
  stopSpeechRecognition();
  if (dataChannel) dataChannel.close();
  if (pc) pc.close();
  if (ws) ws.close();
  if (localStream) localStream.getTracks().forEach(track => track.stop());
  window.location.href = "/";
}

startCall();
