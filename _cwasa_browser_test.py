import base64
import json
import os
import random
import socket
import struct
import subprocess
import time
import urllib.request


CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PLAYER_URL = "http://127.0.0.1:5000/static/avatar/cwasa_player.html?v=cwasa-asl-dict-20260521"
TEST_LABELS = [part.strip().upper() for part in os.environ.get("CWASA_TEST_LABELS", "HELP,HELLO").split(",") if part.strip()]
READY_WAIT_SECONDS = int(os.environ.get("CWASA_READY_WAIT_SECONDS", "120"))


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


PORT = int(os.environ.get("CWASA_DEBUG_PORT") or get_free_port())


def chrome_args(profile, port):
    return [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--disable-crash-reporter",
        "--disable-crashpad",
        "--disable-breakpad",
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-sync",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-sandbox",
        "--autoplay-policy=no-user-gesture-required",
        "--remote-debugging-address=127.0.0.1",
        "--remote-allow-origins=*",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "about:blank",
    ]


class CDP:
    def __init__(self, ws_url):
        self.sock = self._connect(ws_url)
        self.next_id = 0
        self.pending = {}
        self.events = []

    def _connect(self, ws_url):
        if not ws_url.startswith("ws://"):
            raise ValueError(ws_url)
        rest = ws_url[5:]
        host_port, path = rest.split("/", 1)
        host, port = host_port.split(":", 1)
        sock = socket.create_connection((host, int(port)), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET /{path} HTTP/1.1\r\n"
            f"Host: {host_port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(request.encode("ascii"))
        response = sock.recv(4096)
        if b" 101 " not in response:
            raise RuntimeError(response.decode("latin1", "replace"))
        sock.settimeout(0.5)
        return sock

    def _send_frame(self, text):
        payload = text.encode("utf-8")
        mask = os.urandom(4)
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        masked = bytes(payload[i] ^ mask[i % 4] for i in range(length))
        self.sock.sendall(bytes(header) + mask + masked)

    def _recv_frame(self):
        first = self.sock.recv(2)
        if len(first) < 2:
            raise EOFError()
        opcode = first[0] & 0x0F
        length = first[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self.sock.recv(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self.sock.recv(8))[0]
        if first[1] & 0x80:
            mask = self.sock.recv(4)
        else:
            mask = None
        payload = b""
        while len(payload) < length:
            payload += self.sock.recv(length - len(payload))
        if mask:
            payload = bytes(payload[i] ^ mask[i % 4] for i in range(length))
        if opcode == 8:
            raise EOFError()
        if opcode == 9:
            self._send_frame("")
            return None
        if opcode != 1:
            return None
        return json.loads(payload.decode("utf-8"))

    def call(self, method, params=None, timeout=10):
        self.next_id += 1
        msg_id = self.next_id
        self._send_frame(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self._recv_frame()
            except socket.timeout:
                continue
            if not msg:
                continue
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})
            self.events.append(msg)
        raise TimeoutError(method)

    def drain(self, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                msg = self._recv_frame()
            except socket.timeout:
                continue
            if msg:
                self.events.append(msg)

    def eval(self, expression, timeout=10):
        result = self.call("Runtime.evaluate", {
            "expression": expression,
            "awaitPromise": True,
            "returnByValue": True,
            "userGesture": True,
        }, timeout=timeout)
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            return {"exception": details.get("text", ""), "details": details}
        value = result.get("result", {})
        if value.get("subtype") == "error":
            return {"error": value.get("description")}
        return value.get("value")


def http_json(url, method="GET"):
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=10) as res:
        return json.loads(res.read().decode("utf-8"))


def wait_for_debugger(port=PORT):
    for _ in range(80):
        try:
            return http_json(f"http://127.0.0.1:{port}/json/version")
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Chrome remote debugger did not start")


def event_text(event):
    method = event.get("method")
    params = event.get("params", {})
    if method == "Runtime.consoleAPICalled":
        args = []
        for arg in params.get("args", []):
            args.append(arg.get("value", arg.get("description", "")))
        return f"console.{params.get('type')}: " + " ".join(str(a) for a in args)
    if method == "Runtime.exceptionThrown":
        details = params.get("exceptionDetails", {})
        return "exception: " + details.get("text", "")
    if method == "Log.entryAdded":
        entry = params.get("entry", {})
        return f"log.{entry.get('level')}: {entry.get('text')}"
    if method == "Network.loadingFailed":
        return f"network failed: {params.get('errorText')} {params.get('blockedReason', '')}"
    if method == "Network.responseReceived":
        response = params.get("response", {})
        status = response.get("status", 0)
        if status >= 400:
            return f"network {status}: {response.get('url')}"
    return ""


def main():
    if not os.path.exists(CHROME):
        raise RuntimeError(f"Chrome not found at {CHROME}")

    profile = os.path.abspath(f".cwasa-chrome-{int(time.time())}-{random.randint(1000, 9999)}")
    os.makedirs(profile, exist_ok=True)
    chrome_log = open("_cwasa_chrome.log", "w", encoding="utf-8")
    proc = subprocess.Popen(chrome_args(profile, PORT), stdout=chrome_log, stderr=chrome_log)

    try:
        wait_for_debugger(PORT)
        target = http_json(f"http://127.0.0.1:{PORT}/json/new?{PLAYER_URL}", method="PUT")
        cdp = CDP(target["webSocketDebuggerUrl"])
        for domain in ("Page", "Runtime", "Log", "Network"):
            cdp.call(f"{domain}.enable")
        cdp.call("Page.navigate", {"url": PLAYER_URL}, timeout=30)
        state = cdp.eval("""new Promise(resolve => {
          const started = performance.now();
          const tick = () => {
            const ready = !!window.CWASA && !!document.querySelector('canvas');
            if (ready || performance.now() - started > __READY_WAIT_MS__) {
              resolve({
                href: location.href,
                hasCWASA: !!window.CWASA,
                hasLgr: !!window.lgr,
                body: document.body.innerText,
                canvases: [...document.querySelectorAll('canvas')].map(c => ({w:c.width, h:c.height, clientW:c.clientWidth, clientH:c.clientHeight})),
                avatarHtml: document.querySelector('.CWASAAvatar')?.innerHTML?.slice(0, 400) || ''
              });
            } else {
              setTimeout(tick, 250);
            }
          };
          tick();
        })""".replace("__READY_WAIT_MS__", str(READY_WAIT_SECONDS * 1000)), timeout=READY_WAIT_SECONDS + 5)

        print("STATE", json.dumps(state, indent=2))

        play_script = """new Promise(resolve => {
          const labels = __LABELS__;
          const seen = [];
          const samples = [];
          function canvasSample() {
            const c = document.querySelector('canvas');
            if (!c) return {exists:false};
            try {
              const ctx = c.getContext('2d');
              if (!ctx) return {exists:true, webgl:true, w:c.width, h:c.height, url:c.toDataURL('image/png').slice(0, 80)};
              const img = ctx.getImageData(0, 0, Math.min(40, c.width), Math.min(40, c.height)).data;
              let sum = 0;
              for (let i = 0; i < img.length; i += 4) sum += img[i] + img[i + 1] + img[i + 2] + img[i + 3];
              return {exists:true, w:c.width, h:c.height, sum};
            } catch (err) {
              return {exists:true, w:c.width, h:c.height, error:String(err.message || err), url:c.toDataURL('image/png').slice(0, 80)};
            }
          }
          function handler(event) {
            if (event.data && event.data.source === 'bridgesign-cwasa') seen.push(event.data);
          }
          window.addEventListener('message', handler);
          samples.push({t:0, sample:canvasSample()});
          setTimeout(() => window.postMessage({source:'bridgesign-cwasa-parent', type:'play', payload:{label:labels[0] || 'HELP'}}, '*'), 250);
          setTimeout(() => samples.push({t:1, sample:canvasSample()}), 1000);
          setTimeout(() => samples.push({t:3, sample:canvasSample()}), 3000);
          setTimeout(() => samples.push({t:6, sample:canvasSample()}), 6000);
          setTimeout(() => window.postMessage({source:'bridgesign-cwasa-parent', type:'play', payload:{label:labels[1] || labels[0] || 'HELLO'}}, '*'), 7000);
          setTimeout(() => samples.push({t:9, sample:canvasSample()}), 9000);
          setTimeout(() => samples.push({t:12, sample:canvasSample()}), 12000);
          setTimeout(() => {
            window.removeEventListener('message', handler);
            resolve({
              seen,
              samples,
              status: document.getElementById('cwasaStatus')?.textContent || '',
              canvases: [...document.querySelectorAll('canvas')].map(c => ({w:c.width, h:c.height})),
              hasCWASA: !!window.CWASA,
              avatarHtml: document.querySelector('.CWASAAvatar')?.innerHTML?.slice(0, 800) || ''
            });
          }, 15000);
        })""".replace("__LABELS__", json.dumps(TEST_LABELS))
        play = cdp.eval(play_script, timeout=25)

        print("PLAY", json.dumps(play, indent=2))
        print("EVENTS")
        for event in cdp.events:
            text = event_text(event)
            if text:
                print(text[:1000])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        chrome_log.close()


if __name__ == "__main__":
    main()
