import json
import os
import random
import subprocess
import time

from _cwasa_browser_test import CDP, PORT, chrome_args, event_text, http_json, wait_for_debugger


APP_URL = "http://127.0.0.1:5000"
APP_READY_WAIT_SECONDS = int(os.environ.get("CWASA_APP_READY_WAIT_SECONDS", "60"))


def main():
    profile = os.path.abspath(f".cwasa-app-chrome-{int(time.time())}-{random.randint(1000, 9999)}")
    os.makedirs(profile, exist_ok=True)
    chrome_log = open("_cwasa_app_chrome.log", "w", encoding="utf-8")
    proc = subprocess.Popen(chrome_args(profile, PORT), stdout=chrome_log, stderr=chrome_log)

    try:
        wait_for_debugger(PORT)
        target = http_json(f"http://127.0.0.1:{PORT}/json/new?{APP_URL}/register", method="PUT")
        cdp = CDP(target["webSocketDebuggerUrl"])
        for domain in ("Page", "Runtime", "Log", "Network"):
            cdp.call(f"{domain}.enable")

        username = f"cwasa_{int(time.time())}_{random.randint(100, 999)}"
        password = "cwasa-test-123"
        cdp.call("Page.navigate", {"url": APP_URL + "/register"}, timeout=30)
        cdp.drain(2)
        register_result = cdp.eval(f"""(() => {{
          document.querySelector('[name=username]').value = {json.dumps(username)};
          document.querySelector('[name=password]').value = {json.dumps(password)};
          document.querySelector('form').submit();
          return true;
        }})()""")
        cdp.drain(3)

        cdp.call("Page.navigate", {"url": APP_URL + "/login"}, timeout=30)
        cdp.drain(2)
        login_result = cdp.eval(f"""(() => {{
          document.querySelector('[name=username]').value = {json.dumps(username)};
          document.querySelector('[name=password]').value = {json.dumps(password)};
          document.querySelector('form').submit();
          return true;
        }})()""")
        # Wait for the redirect to complete naturally so cookie is saved
        for _ in range(10):
            current_href = cdp.eval("location.href")
            if current_href == APP_URL + "/" or current_href == APP_URL:
                break
            cdp.drain(1)
        cdp.drain(2)

        result = cdp.eval("""new Promise(resolve => {
          const seen = [];
          function handler(event) {
            if (event.data && event.data.source === 'bridgesign-cwasa') seen.push(event.data);
          }
          function canvasInfo() {
            const frame = document.querySelector('#motionAvatarContainer iframe');
            const doc = frame && frame.contentDocument;
            const canvas = doc && doc.querySelector('canvas');
            return {
              hasFrame: !!frame,
              hasCanvas: !!canvas,
              w: canvas ? canvas.width : 0,
              h: canvas ? canvas.height : 0,
              status: document.getElementById('motionStatus')?.textContent || '',
              clipStatus: document.getElementById('motionClipStatus')?.textContent || '',
              driverReady: !!window.motionAvatar
            };
          }
          window.addEventListener('message', handler);
          const motionTab = document.querySelector('[data-tab="motion"]');
          if (!motionTab) {
            window.removeEventListener('message', handler);
            resolve({
              error: "motion tab not found",
              href: location.href,
              body: document.body.innerText.slice(0, 600),
              seen,
              info: canvasInfo()
            });
            return;
          }
          motionTab.click();
          const started = performance.now();
          const maxReadyMs = __APP_READY_WAIT_MS__;

          function waitForDriver() {
            return new Promise(resolveDriver => {
              const tick = () => {
                if (window.motionAvatar && window.motionAvatar._readyPromise) {
                  resolveDriver(window.motionAvatar);
                  return;
                }
                if (performance.now() - started >= maxReadyMs) {
                  resolveDriver(null);
                  return;
                }
                setTimeout(tick, 250);
              };
              tick();
            });
          }

          async function run() {
            const avatar = await waitForDriver();
            if (avatar && avatar._readyPromise) {
              try {
                await Promise.race([
                  avatar._readyPromise,
                  new Promise(resolveReadyTimeout => setTimeout(resolveReadyTimeout, maxReadyMs))
                ]);
              } catch (_) {
                // The result below captures the current page state.
              }
              if (avatar.isCWASA) {
                avatar.queueSign('HELP');
                setTimeout(() => avatar.queueSign('HELLO'), 2500);
              }
            }
            setTimeout(() => {
              window.removeEventListener('message', handler);
              resolve({seen, info: canvasInfo()});
            }, 9000);
          }

          run();
        })""".replace("__APP_READY_WAIT_MS__", str(APP_READY_WAIT_SECONDS * 1000)), timeout=APP_READY_WAIT_SECONDS + 20)

        print("REGISTER", register_result)
        print("LOGIN", login_result)
        print("APP_PLAY", json.dumps(result, indent=2))
        print("EVENTS")
        for event in cdp.events:
            text = event_text(event)
            if text:
                try:
                    print(text[:1000])
                except UnicodeEncodeError:
                    print(text[:1000].encode("ascii", "replace").decode("ascii"))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        chrome_log.close()


if __name__ == "__main__":
    main()
