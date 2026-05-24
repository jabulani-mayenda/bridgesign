"""BridgeSign PWA smoke test — tests all new endpoints."""
import urllib.request, urllib.parse, json, io, time
import numpy as np, cv2

BASE   = "http://127.0.0.1:5000"
jar    = urllib.request.HTTPCookieProcessor()
opener = urllib.request.build_opener(jar)
opener.addheaders = [("User-Agent", "SmokeTest/1.0")]
OK = "[PASS]"; FAIL = "[FAIL]"

def post_form(path, fields):
    data = urllib.parse.urlencode(fields).encode()
    req  = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    return opener.open(req)

def post_json(path, payload):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    r = opener.open(req)
    return json.loads(r.read())

def get(path):
    return opener.open(BASE + path)

print("=== BridgeSign PWA Smoke Test ===\n")

# 1. Register fresh user
uname = f"smoke_{int(time.time())}"
try:
    post_form("/register", {"username": uname, "password": "testpass"})
    print(f"{OK} 1. Register user '{uname}'")
except Exception as e:
    print(f"{FAIL} 1. Register: {e}")

# 2. Login
try:
    post_form("/login", {"username": uname, "password": "testpass"})
    print(f"{OK} 2. Login")
except Exception as e:
    print(f"{FAIL} 2. Login: {e}")

# 3. camera/start
try:
    req = urllib.request.Request(BASE + "/api/camera/start", data=b"", method="POST")
    d = json.loads(opener.open(req).read())
    assert d.get("ok"), f"not ok: {d}"
    print(f"{OK} 3. camera/start: {d}")
except Exception as e:
    print(f"{FAIL} 3. camera/start: {e}")

# 4. infer_frame with blank image -> expect hand_state=no_hand
try:
    blank = np.zeros((480, 640, 3), np.uint8)
    _, buf = cv2.imencode(".jpg", blank)
    img_bytes = buf.tobytes()
    B = "BndryXYZ"
    body  = f"--{B}\r\n".encode()
    body += b'Content-Disposition: form-data; name="frame"; filename="f.jpg"\r\n'
    body += b"Content-Type: image/jpeg\r\n\r\n"
    body += img_bytes
    body += f"\r\n--{B}--\r\n".encode()
    req = urllib.request.Request(BASE + "/api/infer_frame", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={B}")
    d = json.loads(opener.open(req).read())
    assert "hand_state" in d, f"missing hand_state: {d}"
    print(f"{OK} 4. infer_frame: hand_state={d['hand_state']} label={repr(d['label'])}")
except Exception as e:
    print(f"{FAIL} 4. infer_frame: {e}")

# 5. stt/text -> expect guidance list
try:
    d = post_json("/api/stt/text", {"text": "help me please"})
    g = d.get("guidance", [])
    print(f"{OK} 5. stt/text: {len(g)} guidance items returned")
except Exception as e:
    print(f"{FAIL} 5. stt/text: {e}")

# 6. mode switch
try:
    d = post_json("/api/mode", {"mode": "word"})
    assert d.get("ok") and d.get("mode") == "word"
    print(f"{OK} 6. mode switch: {d}")
except Exception as e:
    print(f"{FAIL} 6. mode switch: {e}")

# 7. camera/stop
try:
    req = urllib.request.Request(BASE + "/api/camera/stop", data=b"", method="POST")
    d = json.loads(opener.open(req).read())
    assert d.get("ok")
    print(f"{OK} 7. camera/stop: {d}")
except Exception as e:
    print(f"{FAIL} 7. camera/stop: {e}")

# 8. manifest.json
try:
    m = json.loads(get("/static/manifest.json").read())
    assert m.get("name") == "BridgeSign" and m.get("display") == "standalone"
    print(f"{OK} 8. manifest: name={m['name']} display={m['display']}")
except Exception as e:
    print(f"{FAIL} 8. manifest: {e}")

# 9. sw.js route
try:
    r = get("/sw.js")
    ct = r.headers.get("Content-Type", "")
    sw = r.read().decode()
    assert "serviceWorker" in sw or "fetch" in sw
    print(f"{OK} 9. sw.js: status={r.status} content-type={ct[:30]}")
except Exception as e:
    print(f"{FAIL} 9. sw.js: {e}")

print("\n=== Done ===")
