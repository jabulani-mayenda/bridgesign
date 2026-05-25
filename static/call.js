/* ── BridgeSign call.js ────────────────────────────────────────────
   Fixes in this version:
   1. Flip camera  – toggles front / back camera (phone-friendly)
   2. Sign-to-speech – deaf partner's signs are spoken aloud on hearing side
   3. Speech-to-sign – hearing partner's speech drives the avatar on deaf side
   4. Role-aware UI labels and avatar status messages
─────────────────────────────────────────────────────────────────── */

const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
const wsUrl = `${wsScheme}://${window.location.host}/ws/call/${window.ROOM_ID}`;

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
let lastSign = "";
let lastSignTime = 0;

let _localHands = null;
let useClientInference = false;

const INFER_INTERVAL_MS = 140;
const SIGN_REPEAT_MS = 1600;
const ICE_SERVERS = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "stun:stun1.l.google.com:19302" },
    { urls: "stun:stun2.l.google.com:19302" },
    { urls: "stun:stun3.l.google.com:19302" },
    { urls: "stun:stun4.l.google.com:19302" }
  ]
};

// ── Initialize MediaPipe Hands locally if available ──
function initLocalHands() {
  if (typeof Hands !== "undefined") {
    try {
      _localHands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });
      _localHands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.65,
        minTrackingConfidence: 0.55
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
function handleLocalHandsResults(results) {
  _inferInFlight = false; // Release lock
  if (!localStream || cameraOff || role !== "deaf") return;

  const video = els.localVideo;
  if (!video || !video.videoWidth) return;

  let landmarksList = null;
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const hand = results.multiHandLandmarks[0];
    const w = video.videoWidth;
    const h = video.videoHeight;
    
    // Mirror X-coordinates (1.0 - x) because the server model was
    // trained on mirrored front-camera coordinates.
    landmarksList = hand.map((lm, idx) => [
      idx,
      Math.round((1.0 - lm.x) * w),
      Math.round(lm.y * h)
    ]);
  }

  if (landmarksList) {
    sendLocalLandmarks(landmarksList);
  }
}

// Send local landmarks to the landmarks infer endpoint
async function sendLocalLandmarks(landmarks) {
  try {
    const res = await fetch("/api/infer_landmarks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ landmarks })
    });
    if (res.ok) {
      handleInferResponse(await res.json());
    }
  } catch (err) {
    console.error("[Call] Landmarks infer error:", err);
  }
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

// ── Web Speech API speak helper (browser TTS) ──────────────────────
function speak(text) {
  if (!text || !window.speechSynthesis) return;
  const normalized = text === text.toUpperCase()
    ? text.toLowerCase().replace(/\b\w/g, c => c.toUpperCase())
    : text;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(normalized);
  utt.rate = 0.95;
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

async function playTextOnAvatar(text) {
  const cleanText = String(text || "").trim();
  if (!cleanText) return;

  initCallAvatar();
  const avatar = window.callAvatar;
  if (!avatar) return;

  try {
    const res = await fetch("/api/stt/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: cleanText })
    });
    if (res.ok) {
      const data = await res.json();
      if (data.guidance && data.guidance.length) {
        queueGuidanceOnAvatar(data.guidance);
        return;
      }
    }
  } catch (_) {}

  if (typeof avatar.queueText === "function") avatar.queueText(cleanText);
}

// ── Main startup ───────────────────────────────────────────────────
async function startCall() {
  cacheElements();
  bindControls();
  initCallAvatar();
  
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

    // If WE are the hearing partner, speak the sign aloud so we hear it
    if (role === "hearing") {
      speak(label);
    }

    // If WE are the deaf partner receiving a sign echo, show on avatar
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
    setCaption(`"${text}"`, "var(--honey)");
    setTranscript(text);

    // If WE are the deaf partner receiving partner's speech, animate the avatar
    if (role === "deaf") {
      playTextOnAvatar(text);
    }

    // If WE are the hearing partner receiving a speech echo, just show it
    return;
  }

  if (data.type === "text") {
    const text = data.text || "";
    setCaption(text, "var(--sage)");
    setTranscript(text);

    // Typed text from deaf side → speak it on hearing side
    if (role === "hearing") {
      speak(text);
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
  }
}

// ── Role logic: deaf = sign inference, hearing = speech recognition ─
function updateRoleLogic() {
  stopInferLoop();
  stopSpeechRecognition();

  if (role === "deaf") {
    // I am signing → run hand inference, my signs go to the hearing partner as speech
    startInferLoop();
    if (els.callAvatarStatus) els.callAvatarStatus.textContent = "Partner speech will sign here";
    setCaption("Sign when ready – partner will hear you", "rgba(255,255,255,.78)");
    return;
  }

  // I am the hearing partner → use my microphone, my speech drives the deaf side avatar
  startSpeechRecognition();
  if (els.callAvatarStatus) els.callAvatarStatus.textContent = "Your speech becomes signs for partner";
  setCaption("Speak – partner's signs will appear here", "rgba(255,255,255,.78)");
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
}

async function captureAndInfer() {
  if (_inferInFlight || !localStream || cameraOff) return;
  const video = els.localVideo;
  if (!video || !video.videoWidth) return;

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
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.save();

    // Only flip horizontally for front camera (model trained with flipped front cam frames)
    if (facingMode === "user") {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }

    ctx.drawImage(video, 0, 0);
    ctx.restore();

    _inferInFlight = true;
    canvas.toBlob(async (blob) => {
      if (!blob) {
        _inferInFlight = false;
        return;
      }

      const fd = new FormData();
      fd.append("frame", blob, "frame.jpg");
      try {
        const res = await fetch("/api/infer_frame", { method: "POST", body: fd });
        if (res.ok) handleInferResponse(await res.json());
      } catch (_) {}
      _inferInFlight = false;
    }, "image/jpeg", 0.72);
  }
}

function handleInferResponse(data) {
  // Prefer completed sentence > completed word > current label
  const label = data.completed_sentence || data.completed_word || data.label || "";
  if (!label || data.hand_state !== "recognised") return;

  const now = Date.now();
  if (label === lastSign && now - lastSignTime < SIGN_REPEAT_MS) return;

  lastSign = label;
  lastSignTime = now;

  // Send the sign/word over the data channel to the hearing partner
  // who will then have it spoken aloud by their browser
  sendCaption("sign", label, { confidence: data.confidence || 0 });
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
      // Send speech to the deaf partner — their avatar will sign it
      sendCaption("speech", finalTranscript.trim());
    } else if (interimTranscript) {
      setCaption(`"${interimTranscript.trim()}"`, "rgba(255,255,255,.72)");
      setTranscript(interimTranscript.trim());
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
