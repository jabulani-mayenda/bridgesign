from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _cwasa_browser_test import CDP, PLAYER_URL, PORT, chrome_args, event_text, http_json, wait_for_debugger


READY_WAIT_SECONDS = int(os.environ.get("CWASA_FULL_READY_WAIT_SECONDS", "90"))
SIGN_TIMEOUT_MS = int(os.environ.get("CWASA_SIGN_TIMEOUT_MS", "12000"))
MANIFEST = ROOT / "static" / "avatar" / "sigml" / "asl" / "manifest.json"
REPORT = ROOT / "cwasa_sigml_validation_report.json"


def labels_to_test() -> list[str]:
    env_labels = os.environ.get("CWASA_TEST_LABELS", "")
    if env_labels.strip():
        return [part.strip().upper() for part in env_labels.split(",") if part.strip()]

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return [str(label).upper() for label in manifest["signs"]]


def main() -> int:
    labels = labels_to_test()
    if not labels:
        print("No labels to test.")
        return 1

    profile = ROOT / f".cwasa-full-chrome-{int(time.time())}-{random.randint(1000, 9999)}"
    profile.mkdir(exist_ok=True)
    chrome_log = (ROOT / "_cwasa_full_chrome.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(chrome_args(str(profile), PORT), stdout=chrome_log, stderr=chrome_log)

    try:
        wait_for_debugger(PORT)
        target = http_json(f"http://127.0.0.1:{PORT}/json/new?{PLAYER_URL}", method="PUT")
        cdp = CDP(target["webSocketDebuggerUrl"])
        for domain in ("Page", "Runtime", "Log", "Network"):
            cdp.call(f"{domain}.enable")

        cdp.call("Page.navigate", {"url": PLAYER_URL}, timeout=30)
        script = r"""new Promise(resolve => {
          const labels = __LABELS__;
          const readyWaitMs = __READY_WAIT_MS__;
          const signTimeoutMs = __SIGN_TIMEOUT_MS__;

          function normalize(label) {
            return String(label || "").trim().toUpperCase().replace(/[^A-Z0-9]+/g, "_").replace(/^_+|_+$/g, "");
          }

          function isPlayerEvent(data) {
            return data && data.source === "bridgesign-cwasa";
          }

          function canvasInfo() {
            const c = document.querySelector("canvas");
            return c ? { exists: true, w: c.width, h: c.height, clientW: c.clientWidth, clientH: c.clientHeight } : { exists: false };
          }

          function waitForSurface() {
            return new Promise(resolveReady => {
              const started = performance.now();
              function tick() {
                const readyEnough = !!window.CWASA && !!document.querySelector("canvas");
                if (readyEnough || performance.now() - started > readyWaitMs) {
                  resolveReady(readyEnough);
                  return;
                }
                setTimeout(tick, 250);
              }
              tick();
            });
          }

          function playOne(label) {
            label = normalize(label);
            return new Promise(resolveOne => {
              const record = {
                label,
                ok: false,
                error: "",
                frames: null,
                sawSigning: false,
                sawLoaded: false,
                sawActive: false,
                sawIdle: false,
                events: [],
                canvas: null
              };
              let finished = false;

              function finish(ok, error) {
                if (finished) return;
                finished = true;
                clearTimeout(timer);
                window.removeEventListener("message", onMessage);
                record.ok = !!ok;
                record.error = error || "";
                record.canvas = canvasInfo();
                resolveOne(record);
              }

              function remember(type, detail) {
                record.events.push({
                  type,
                  event: detail && detail.event || "",
                  label: detail && detail.label || "",
                  frames: detail && detail.frames != null ? detail.frames : null
                });
              }

              function onMessage(event) {
                const data = event.data || {};
                if (!isPlayerEvent(data)) return;
                const detail = data.detail || {};

                if (data.type === "signing") {
                  remember(data.type, detail);
                  if (normalize(detail.label) === label) record.sawSigning = true;
                  return;
                }

                if (data.type === "debug") {
                  remember(data.type, detail);
                  if (detail.event === "sigmlloaded") {
                    record.sawLoaded = true;
                    record.frames = detail.frames == null ? null : detail.frames;
                  }
                  if (detail.event === "animactive") record.sawActive = true;
                  if (detail.event === "animidle") {
                    record.sawIdle = true;
                    finish(
                      record.sawSigning && record.sawLoaded && record.sawActive,
                      record.sawSigning && record.sawLoaded && record.sawActive ? "" : "missing lifecycle event"
                    );
                  }
                  return;
                }

                if (data.type === "unsupported" && normalize(detail.label) === label) {
                  remember(data.type, detail);
                  finish(false, "unsupported");
                  return;
                }

                if (data.type === "error") {
                  remember(data.type, detail);
                  finish(false, detail.message || "player error");
                }
              }

              const timer = setTimeout(() => finish(false, "timeout"), signTimeoutMs);
              window.addEventListener("message", onMessage);
              window.postMessage({
                source: "bridgesign-cwasa-parent",
                type: "play",
                payload: { label }
              }, "*");
            });
          }

          (async () => {
            const surfaceReady = await waitForSurface();
            const results = [];
            for (const label of labels) {
              results.push(await playOne(label));
            }
            resolve({
              href: location.href,
              hasCWASA: !!window.CWASA,
              surfaceReady,
              canvas: canvasInfo(),
              results
            });
          })().catch(err => {
            resolve({ error: err && err.message ? err.message : String(err), results: [] });
          });
        })"""

        script = (
            script.replace("__LABELS__", json.dumps(labels))
            .replace("__READY_WAIT_MS__", str(READY_WAIT_SECONDS * 1000))
            .replace("__SIGN_TIMEOUT_MS__", str(SIGN_TIMEOUT_MS))
        )
        result = cdp.eval(
            script,
            timeout=READY_WAIT_SECONDS + (len(labels) * (SIGN_TIMEOUT_MS / 1000)) + 30,
        )

        REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if os.environ.get("CWASA_VERBOSE") == "1":
            print(json.dumps(result, indent=2))

        results = result.get("results", []) if isinstance(result, dict) else []
        failed = [item for item in results if not item.get("ok")]
        passed = len(results) - len(failed)
        print(f"REPORT: {REPORT}")
        print(f"SUMMARY: {passed}/{len(labels)} signs passed")
        if failed:
            print("FAILED:")
            for item in failed:
                print(f"- {item.get('label')}: {item.get('error')}")
            return 1

        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        chrome_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
