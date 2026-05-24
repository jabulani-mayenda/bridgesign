/* BridgeSign Service Worker
   - Caches app shell on install (HTML/CSS/JS/fonts)
   - API calls always go to network (no stale inference)
   - Shows offline page when network unreachable
*/
const CACHE  = "bridgesign-v17-speech-fallback";
const SHELL  = [
  "/",
  "/static/app.js",
  "/static/style.css",
  "/static/manifest.json",
  "/static/icon-192.png",
  "/static/icon-512.png",
  "/static/avatar/avatar_controller.js",
  "/static/avatar/avatar_driver.js",
  "/static/avatar/cwasa_driver.js",
  "/static/vendor/cwasa/vhg2026/cwa/allcsa.js",
  "/static/vendor/cwasa/vhg2026/cwa/cwasa.css",
  "/static/vendor/cwasa/vhg2026/cwa/h2s.xsl",
  "/static/vendor/cwasa/vhg2026/cwa/shaders/qskin.vert",
  "/static/vendor/cwasa/vhg2026/cwa/shaders/qskin.frag",
  "/static/vendor/cwasa/vhg2026/avatars/COMMON.jar",
  "/static/vendor/cwasa/vhg2026/avatars/anna.jar",
  "/static/call.js",
  "/static/lib/three.module.js",
  "/static/lib/GLTFLoader.js",
  "/static/lib/three-vrm.module.min.js",
  "/static/utils/BufferGeometryUtils.js",
];
const OFFLINE_HTML = `<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BridgeSign – Offline</title>
<style>
  body{margin:0;display:flex;align-items:center;justify-content:center;min-height:100vh;
       font-family:system-ui,sans-serif;background:#1a1612;color:#e8ddd0;text-align:center;padding:2rem}
  h1{font-size:2rem;margin-bottom:.5rem}p{opacity:.6}
  .icon{font-size:4rem;margin-bottom:1rem}
</style></head>
<body>
  <div><div class="icon">◎</div>
  <h1>You're offline</h1>
  <p>BridgeSign needs a connection to run inference.<br>Connect to the internet and try again.</p></div>
</body></html>`;

self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);

  // Never cache API calls or video streams
  if (url.pathname.startsWith("/api/") || url.pathname === "/video_feed") {
    e.respondWith(fetch(e.request).catch(() =>
      new Response(JSON.stringify({ error: "offline" }), {
        headers: { "Content-Type": "application/json" }
      })
    ));
    return;
  }

  // For navigation (HTML pages): network-first, fall back to offline page
  if (e.request.mode === "navigate") {
    e.respondWith(
      fetch(e.request).catch(() =>
        new Response(OFFLINE_HTML, { headers: { "Content-Type": "text/html" } })
      )
    );
    return;
  }

  // Core code changes often during local development. Keep it network-first so
  // camera fixes do not get hidden behind an old PWA cache.
  if (
    url.pathname === "/static/app.js" ||
    url.pathname === "/static/style.css" ||
    url.pathname === "/static/call.js" ||
    url.pathname === "/static/avatar/cwasa_driver.js" ||
    url.pathname === "/static/avatar/avatar_controller.js" ||
    url.pathname === "/static/avatar/avatar_driver.js"
  ) {
    e.respondWith(
      fetch(e.request).then(response => {
        const resClone = response.clone();
        caches.open(CACHE).then(cache => cache.put(e.request, resClone));
        return response;
      }).catch(() => caches.match(e.request))
    );
    return;
  }

  // Static assets: cache-first with dynamic caching for 3D animations & images
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(response => {
        if (response.ok && (url.pathname.includes('/animations/') || url.pathname.includes('/signs/'))) {
          const resClone = response.clone();
          caches.open(CACHE).then(cache => cache.put(e.request, resClone));
        }
        return response;
      });
    })
  );
});
