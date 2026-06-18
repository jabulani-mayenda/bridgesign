(function () {
  const PLAYER_URL = "/static/avatar/cwasa_player.html?v=cwasa-asl-dict-20260521";
  const STILL_WAITING_MS = 15000;
  let nextId = 0;

  function labelFromText(text) {
    return String(text || "")
      .trim()
      .toUpperCase()
      .replace(/[^A-Z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
  }

  function status(options, text) {
    const el = options.statusId ? document.getElementById(options.statusId) : null;
    if (el) el.textContent = text;
  }

  function create(containerId, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) return null;

    const id = `cwasaAvatar${++nextId}`;
    container.innerHTML = "";
    container.classList.add("cwasa-avatar-host");

    const iframe = document.createElement("iframe");
    iframe.id = id;
    iframe.className = "cwasa-avatar-frame";
    iframe.title = "CWASA signing avatar";
    iframe.src = PLAYER_URL;
    iframe.setAttribute("allow", "fullscreen");
    iframe.setAttribute("loading", "eager");
    container.appendChild(iframe);

    const fallbackHost = document.createElement("div");
    fallbackHost.className = "cwasa-vrm-fallback";
    fallbackHost.style.opacity = "0";
    fallbackHost.style.pointerEvents = "none";
    container.appendChild(fallbackHost);

    const loader = options.loaderId ? document.getElementById(options.loaderId) : null;
    if (loader) {
      loader.style.display = "block";
      loader.textContent = "Loading CWASA avatar...";
    }

    let readyResolve;
    let readyReject;
    const readyPromise = new Promise((resolve, reject) => {
      readyResolve = resolve;
      readyReject = reject;
    });
    let isReady = false;
    const pendingPayloads = [];

    const driver = {
      isCWASA: true,
      _bridgeSignAvatarContainer: containerId,
      _readyPromise: readyPromise,
      queueSign(signLabel) {
        showCWASA();
        postWhenReady({ label: labelFromText(signLabel) });
      },
      queueLetters(text) {
        for (const ch of String(text || "").replace(/[^A-Za-z]/g, "").toUpperCase()) {
          showCWASA();
          postWhenReady({ label: ch });
        }
      },
      queueText(text) {
        const words = String(text || "").match(/[A-Za-z_]+/g) || [];
        for (const word of words) {
          showCWASA();
          postWhenReady({ label: labelFromText(word) });
        }
      },
      playSiGMLText(sigml, label = "CUSTOM") {
        showCWASA();
        postWhenReady({ sigml, label: labelFromText(label) || "CUSTOM" });
      },
      resize() {},
      destroy() {
        window.removeEventListener("message", onMessage);
        iframe.remove();
      }
    };

    let fallbackAvatar = null;
    let cwasaMotionTimer = null;

    function showCWASA() {
      iframe.style.opacity = "1";
      iframe.style.pointerEvents = "";
      fallbackHost.style.opacity = "0";
      fallbackHost.style.pointerEvents = "none";
    }

    function showFallback() {
      iframe.style.opacity = "0";
      iframe.style.pointerEvents = "none";
      fallbackHost.style.opacity = "1";
      fallbackHost.style.pointerEvents = "";
    }

    function ensureFallbackAvatar() {
      if (fallbackAvatar) return fallbackAvatar;
      if (!window.AvatarController) return null;
      const fallbackId = `${id}Fallback`;
      fallbackHost.id = fallbackId;
      try {
        fallbackAvatar = new window.AvatarController(fallbackId);
        return fallbackAvatar;
      } catch (err) {
        console.warn("[CWASA] VRM fallback could not start:", err);
        return null;
      }
    }

    function clearMotionWatch() {
      if (cwasaMotionTimer) {
        clearTimeout(cwasaMotionTimer);
        cwasaMotionTimer = null;
      }
    }

    function startMotionWatch(label) {
      clearMotionWatch();
      const startedAt = Date.now();
      const cleanLabel = label.replace(/_/g, " ");

      const tick = () => {
        const elapsedSec = Math.round((Date.now() - startedAt) / 1000);
        console.info(`[CWASA] Still waiting for ${label} after ${elapsedSec}s.`);
        status(options, `Still signing: ${cleanLabel} (${elapsedSec}s)`);
        cwasaMotionTimer = setTimeout(tick, STILL_WAITING_MS);
      };

      cwasaMotionTimer = setTimeout(tick, STILL_WAITING_MS);
    }

    function post(payload) {
      iframe.contentWindow?.postMessage({
        source: "bridgesign-cwasa-parent",
        type: "play",
        payload
      }, "*");
    }

    function postWhenReady(payload) {
      if (!isReady) {
        pendingPayloads.push(payload);
        status(options, "Loading CWASA avatar...");
        return;
      }
      post(payload);
    }

    function flushPending() {
      while (pendingPayloads.length) post(pendingPayloads.shift());
    }

    function onMessage(event) {
      if (event.source !== iframe.contentWindow) return;
      const data = event.data || {};
      if (data.source !== "bridgesign-cwasa") return;

      if (data.type === "ready") {
        isReady = true;
        if (loader) loader.style.display = "none";
        status(options, options.readyText || "CWASA avatar ready");
        readyResolve(driver);
        flushPending();
        return;
      }

      if (data.type === "signing") {
        const label = data.detail?.label || "SIGN";
        status(options, `Signing: ${label.replace(/_/g, " ")}`);
        startMotionWatch(label);
        window.dispatchEvent(new CustomEvent("avatar-signing", {
          detail: { label, engine: "cwasa", queueLength: 0 }
        }));
        return;
      }

      if (data.type === "debug") {
        if (data.detail?.event === "animactive" || data.detail?.event === "animidle") clearMotionWatch();
        if (data.detail?.event === "still-playing") {
          const label = data.detail?.label || "SIGN";
          const elapsedSec = Math.round((data.detail?.elapsedMs || 0) / 1000);
          status(options, `Still signing: ${label.replace(/_/g, " ")} (${elapsedSec}s)`);
        }
        console.info("[CWASA]", data.detail || {});
        return;
      }

      if (data.type === "unsupported") {
        const label = data.detail?.label || "SIGN";
        clearMotionWatch();
        status(options, `Needs SiGML: ${label.replace(/_/g, " ")}`);
        showCWASA();
        window.dispatchEvent(new CustomEvent("avatar-cwasa-unsupported", {
          detail: { label, containerId }
        }));
        return;
      }

      if (data.type === "error") {
        const message = data.detail?.message || "CWASA avatar error.";
        const label = data.detail?.label || "";
        clearMotionWatch();
        const fallback = options.allowVRMFallback === true && label ? ensureFallbackAvatar() : null;
        if (fallback) {
          showFallback();
          fallback.queueSign(label);
        } else {
          showCWASA();
        }
        if (loader) {
          loader.style.display = "block";
          loader.textContent = message;
        }
        status(options, message);
        readyReject(new Error(message));
        window.dispatchEvent(new CustomEvent("avatar-error", {
          detail: { containerId, message, engine: "cwasa" }
        }));
      }
    }

    window.addEventListener("message", onMessage);
    return driver;
  }

  window.CWASABridge = { create };
})();
