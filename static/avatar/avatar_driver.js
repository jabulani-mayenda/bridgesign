(function () {
  const DEFAULT_MAX_WAIT_MS = 10000;
  const RETRY_MS = 150;

  function getContainer(containerId) {
    return document.getElementById(containerId);
  }

  function isContainerVisible(container) {
    if (!container) return false;
    const rect = container.getBoundingClientRect();
    return rect.width >= 80 && rect.height >= 80;
  }

  function safeId(containerId, suffix) {
    return `${containerId.replace(/[^A-Za-z0-9_-]/g, "")}${suffix}`;
  }

  function toDisplayLabel(signLabel) {
    return String(signLabel || "").replace(/_/g, " ").trim() || "Waiting for a sign...";
  }

  function signImageSrc(signLabel) {
    const imageKey = String(signLabel || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    return `/static/signs/sign_${imageKey}.png`;
  }

  function ensureFallback(containerId, message, options = {}) {
    const container = getContainer(containerId);
    if (!container) return null;

    const fallbackId = safeId(containerId, "Fallback");
    const wordId = safeId(containerId, "FallbackWord");
    const imageId = safeId(containerId, "FallbackImage");
    const noteId = safeId(containerId, "FallbackNote");

    let stage = document.getElementById(fallbackId);
    if (!stage) {
      stage = document.createElement("div");
      stage.id = fallbackId;
      stage.className = "avatar-fallback";
      stage.innerHTML = `
        <div class="avatar-fallback-badge">${options.fallbackLabel || "Sign Preview"}</div>
        <div class="avatar-fallback-word" id="${wordId}">Waiting for a sign...</div>
        <img class="avatar-fallback-image" id="${imageId}" alt="" style="display:none">
        <div class="avatar-fallback-note" id="${noteId}"></div>
      `;
      container.appendChild(stage);
    }

    const loader = options.loaderId ? document.getElementById(options.loaderId) : null;
    if (loader) loader.style.display = "none";

    const note = document.getElementById(noteId);
    if (note) {
      note.textContent = message || "The 3D avatar is unavailable here, so BridgeSign is showing sign cards instead.";
    }

    return stage;
  }

  function removeFallback(containerId) {
    const stage = document.getElementById(safeId(containerId, "Fallback"));
    if (stage) stage.remove();
  }

  function updateFallback(containerId, signLabel, options = {}) {
    const stage = ensureFallback(containerId, "", options);
    if (!stage) return;

    const wordId = safeId(containerId, "FallbackWord");
    const imageId = safeId(containerId, "FallbackImage");
    const noteId = safeId(containerId, "FallbackNote");
    const wordEl = document.getElementById(wordId);
    const imgEl = document.getElementById(imageId);
    const noteEl = document.getElementById(noteId);

    if (wordEl) wordEl.textContent = toDisplayLabel(signLabel);
    if (noteEl) noteEl.textContent = options.activeNote || "Following the current sign sequence.";

    if (imgEl) {
      imgEl.onload = () => { imgEl.style.display = "block"; };
      imgEl.onerror = () => { imgEl.style.display = "none"; };
      imgEl.src = signImageSrc(signLabel);
      imgEl.alt = toDisplayLabel(signLabel);
    }
  }

  function createFallbackDriver(containerId, message, options = {}) {
    ensureFallback(containerId, message, options);
    return {
      isFallback: true,
      queueSign(signLabel) {
        updateFallback(containerId, signLabel, options);
      },
      queueLetters(text) {
        for (const ch of String(text || "").replace(/[^A-Za-z]/g, "").toUpperCase()) {
          updateFallback(containerId, ch, options);
        }
      },
      queueText(text) {
        const words = String(text || "").match(/[A-Za-z]+/g) || [];
        if (!words.length) return;
        updateFallback(containerId, words[0].toUpperCase(), options);
      },
      resize() {}
    };
  }

  function addTextQueue(driver) {
    if (!driver || driver.queueText) return driver;
    driver.queueText = async function queueText(text) {
      const words = String(text || "").match(/[A-Za-z]+/g) || [];
      for (const word of words) {
        await driver.queueSign(word.toUpperCase());
      }
    };
    return driver;
  }

  function setStatus(options, text) {
    const status = options.statusId ? document.getElementById(options.statusId) : null;
    if (status) status.textContent = text;
  }

  function init(containerId, options = {}) {
    const container = getContainer(containerId);
    if (!container) return null;

    const globalName = options.globalName || "appAvatar";
    const existing = window[globalName];
    if (existing && existing._bridgeSignAvatarContainer === containerId && !existing.isFallback) {
      if (typeof existing.resize === "function") existing.resize();
      return existing;
    }
    if (existing && existing._bridgeSignAvatarContainer === containerId && existing.isFallback) {
      removeFallback(containerId);
      window[globalName] = null;
    }

    const loader = options.loaderId ? document.getElementById(options.loaderId) : null;
    if (loader) {
      loader.style.display = "block";
      loader.textContent = options.preferCWASA === false ? "Loading 3D Avatar..." : "Loading CWASA avatar...";
    }

    if (options.preferCWASA !== false && window.CWASABridge) {
      try {
        removeFallback(containerId);
        const driver = window.CWASABridge.create(containerId, options);
        if (driver) {
          driver._bridgeSignAvatarContainer = containerId;
          window[globalName] = driver;
          return driver;
        }
      } catch (err) {
        console.warn("[AvatarDriver] CWASA startup failed, trying VRM avatar:", err);
      }
    }

    const maxWaitMs = options.maxWaitMs || DEFAULT_MAX_WAIT_MS;
    const startedAt = Date.now();

    const attempt = () => {
      if (!isContainerVisible(container)) {
        if (Date.now() - startedAt < maxWaitMs) {
          setTimeout(attempt, RETRY_MS);
          return;
        }
        const driver = createFallbackDriver(
          containerId,
          "The avatar panel is not visible yet. Open this panel again to retry the 3D avatar.",
          options
        );
        driver._bridgeSignAvatarContainer = containerId;
        window[globalName] = driver;
        return;
      }

      if (window.AvatarController) {
        try {
          removeFallback(containerId);
          const driver = addTextQueue(new window.AvatarController(containerId));
          driver._bridgeSignAvatarContainer = containerId;
          window[globalName] = driver;
          if (loader) loader.style.display = "none";
          setStatus(options, options.readyText || "Avatar ready");
          return;
        } catch (err) {
          console.warn("[AvatarDriver] 3D avatar startup failed:", err);
          const driver = createFallbackDriver(
            containerId,
            "3D avatar could not start on this browser, so BridgeSign switched to sign cards.",
            options
          );
          driver._bridgeSignAvatarContainer = containerId;
          window[globalName] = driver;
          return;
        }
      }

      if (Date.now() - startedAt < maxWaitMs) {
        setTimeout(attempt, RETRY_MS);
        return;
      }

      console.warn("[AvatarDriver] Avatar module did not load in time; using fallback.");
      const driver = createFallbackDriver(
        containerId,
        "3D avatar files did not load in time, so BridgeSign switched to sign cards.",
        options
      );
      driver._bridgeSignAvatarContainer = containerId;
      window[globalName] = driver;
    };

    setTimeout(attempt, 50);
    return window[globalName] || null;
  }

  window.BridgeSignAvatar = {
    init,
    ensureFallback,
    updateFallback,
    createFallbackDriver,
    removeFallback
  };
})();
