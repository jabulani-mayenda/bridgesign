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

let _recognition = null;
let _inferInterval = null;
let _inferInFlight = false;
let lastSign = "";
let lastSignTime = 0;

const INFER_INTERVAL_MS = 140;
const SIGN_REPEAT_MS = 1400;
const ICE_SERVERS = {
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
};

const els = {};

function $(id) {
  return document.getElementById(id);
}

function cacheElements() {
  [
    "roleSelect", "localVideo", "remoteVideo", "callStatus", "captionBar",
    "connectionState", "muteBtn", "cameraBtn", "endBtn", "copyLinkBtn",
    "typedMessageInput", "sendTextBtn", "callTranscript", "callAvatarStatus"
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

async function startCall() {
  cacheElements();
  bindControls();
  initCallAvatar();
  setStatus("Starting camera and microphone...");

  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 960, height: 540, facingMode: "user" },
      audio: true
    });
  } catch (err) {
    try {
      localStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 960, height: 540, facingMode: "user" },
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
    els.localVideo.srcObject = localStream;
    await els.localVideo.play().catch(() => {});
  }

  updateMediaButtons();
  connectWebSocket();
  updateRoleLogic();
}

function bindControls() {
  if (els.roleSelect) {
    role = els.roleSelect.value;
    els.roleSelect.addEventListener("change", (e) => {
      role = e.target.value;
      updateRoleLogic();
    });
  }

  if (els.muteBtn) els.muteBtn.addEventListener("click", toggleMute);
  if (els.cameraBtn) els.cameraBtn.addEventListener("click", toggleCamera);
  if (els.endBtn) els.endBtn.addEventListener("click", endCall);
  if (els.copyLinkBtn) els.copyLinkBtn.addEventListener("click", copyLink);
  if (els.sendTextBtn) els.sendTextBtn.addEventListener("click", sendTypedMessage);
  if (els.typedMessageInput) {
    els.typedMessageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendTypedMessage();
    });
  }
}

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

function handleDataMessage(data) {
  if (data.type === "sign") {
    const label = data.label || "";
    setCaption(label, "var(--terracotta)");
    setTranscript(`Partner signed: ${label}`);
    return;
  }

  if (data.type === "speech") {
    const text = data.text || "";
    setCaption(`"${text}"`, "var(--honey)");
    setTranscript(text);
    playTextOnAvatar(text);
    return;
  }

  if (data.type === "text") {
    const text = data.text || "";
    setCaption(text, "var(--sage)");
    setTranscript(text);
    playTextOnAvatar(text);
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

function updateRoleLogic() {
  stopInferLoop();
  stopSpeechRecognition();

  if (role === "deaf") {
    startInferLoop();
    if (els.callAvatarStatus) els.callAvatarStatus.textContent = "Speech will sign here";
    setCaption("Sign when ready", "rgba(255,255,255,.78)");
    return;
  }

  startSpeechRecognition();
  if (els.callAvatarStatus) els.callAvatarStatus.textContent = "Partner speech signs here";
  setCaption("Speak when ready", "rgba(255,255,255,.78)");
}

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
  const canvas = $("inferCanvas");
  if (!video || !canvas || !video.videoWidth) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
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

function handleInferResponse(data) {
  const label = data.completed_sentence || data.completed_word || data.label || "";
  if (!label || data.hand_state !== "recognised") return;

  const now = Date.now();
  if (label === lastSign && now - lastSignTime < SIGN_REPEAT_MS) return;

  lastSign = label;
  lastSignTime = now;
  sendCaption("sign", label, { confidence: data.confidence || 0 });
}

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

function sendTypedMessage() {
  const input = els.typedMessageInput;
  const text = input ? input.value.trim() : "";
  if (!text) return;
  sendCaption("text", text);
  if (input) input.value = "";
}

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
