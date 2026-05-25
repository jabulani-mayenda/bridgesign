import threading
import time
import json
import os
import re
import secrets
import string
import traceback
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
from flask_cors import CORS
from flask_sock import Sock
from werkzeug.exceptions import HTTPException
from werkzeug.security import generate_password_hash, check_password_hash
import config
from session_tracker import SessionTracker
from emergency_phrases import EmergencyPhrases
from call_room import room_manager

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "bridgesign_secret_key_2024")
app.config["JSON_SORT_KEYS"] = False

# ── Session cookie config for cloud deployment ────────────────────────
# Behind a reverse proxy (Render, Railway) the app must trust forwarded
# headers so Flask knows the request is HTTPS and sets cookies correctly.
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
_is_cloud = os.environ.get("RENDER") or os.environ.get("RAILWAY_ENVIRONMENT")
if _is_cloud:
    app.config["SESSION_COOKIE_SECURE"] = True
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

CORS(app)
sock = Sock(app)

ROOM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,40}$")

# ── Gesture thresholds (tightened to fix confusion between signs and gestures) ──
GESTURE_MOTION_THRESHOLD = 10.0          # was 8.0  – require more movement before gesture even runs
GESTURE_STATIC_SUPPRESS_THRESHOLD = 9.0  # was 6.0  – suppress static sign only with clear motion
GESTURE_FRAME_MOTION_THRESHOLD = 1.0     # was 0.75 – higher per-frame delta required
GESTURE_INFERENCE_EVERY_N = 2
GESTURE_MIN_CONFIDENCE   = 0.62          # was 0.45 – gesture must be more certain
GESTURE_MARGIN           = 3.0           # was 2.5  – gesture needs a wider lead over static
GESTURE_LSTM_MIN_CONF    = 0.65          # was 0.50 – LSTM gesture model must be confident
GESTURE_DECISION_MARGIN  = 0.12          # was 0.05 – bigger lead needed to beat static sign
GESTURE_COOLDOWN_SEC     = 1.8           # was 1.2  – longer pause between gesture emissions
GESTURE_ONLY_LABELS      = {"J", "Z"}

# Debug: print pipeline internals every N frames (set 0 to disable)
_DEBUG_EVERY_N = 0   # set to 30 to enable verbose frame logging

# ── User Store (file-based) ───────────────────────────────────────
USERS_FILE = os.path.join(config.DATA_DIR, "users.json")
_users_lock = threading.Lock()

def load_users():
    try:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[Auth] Could not load users from {USERS_FILE}: {e}")
    return {}

def save_users(users):
    try:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        tmp_path = f"{USERS_FILE}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
        try:
            os.replace(tmp_path, USERS_FILE)
        except OSError as e:
            print(f"[Auth] os.replace failed ({e}) — falling back to direct write to {USERS_FILE}")
            # Fallback to direct write if os.replace fails across different partitions/volumes
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(users, f, indent=2)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    except Exception as e:
        print(f"[Auth] Could not save users to {USERS_FILE}: {e}")
        raise RuntimeError(f"Database write failed: {e}")

def load_user_phrases(username):
    path = os.path.join(config.DATA_DIR, f"phrases_{username}.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError as e:
            print(f"[Phrases] Could not parse {path}: {e}")
            return {}
    return {}

def save_user_phrases(username, phrases):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = os.path.join(config.DATA_DIR, f"phrases_{username}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(phrases, f, indent=2)


def _request_payload():
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form


def _data_dir_probe():
    try:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        probe_path = os.path.join(config.DATA_DIR, ".write_probe")
        with open(probe_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe_path)
        return {"ok": True, "path": config.DATA_DIR}
    except Exception as e:
        return {"ok": False, "path": config.DATA_DIR, "error": str(e)}

# ── Singleton inference modules ─────────────────────────────────
# Load these after the web app has imported so Render can bind a port quickly.
_detector           = None
_motion_detector    = None
_pose_detector      = None
_extractor          = None
_classifier         = None
_gesture_classifier = None
_detector_lock      = threading.Lock()
_motion_detector_lock = threading.Lock()
_pose_detector_lock = threading.Lock()
_inference_init_lock = threading.Lock()
_inference_status = {"ready": False, "loading": False, "error": ""}


def _load_inference_modules():
    """Initialize MediaPipe/model objects once, without blocking app import."""
    global _detector, _motion_detector, _pose_detector
    global _extractor, _classifier, _gesture_classifier

    if _inference_status["ready"]:
        return

    with _inference_init_lock:
        if _inference_status["ready"]:
            return

        _inference_status.update({"ready": False, "loading": True, "error": ""})
        try:
            from hand_detector import HandDetector
            from feature_extractor import FeatureExtractor
            from classifier import Classifier
            from gesture_classifier import GestureClassifier
            from pose_detector import PoseDetector

            print("[BridgeSign] Loading inference modules...")
            _detector           = HandDetector()
            _motion_detector    = HandDetector(max_hands=2, detection_con=0.55, track_con=0.45)
            _pose_detector      = PoseDetector(detection_con=0.5, track_con=0.5)
            _extractor          = FeatureExtractor()
            _classifier         = Classifier()
            _gesture_classifier = GestureClassifier()
            print("[BridgeSign] Modules ready.")
            _run_sanity_check()
            _inference_status.update({"ready": True, "loading": False, "error": ""})
        except Exception as e:
            _inference_status.update({"ready": False, "loading": False, "error": str(e)})
            print(f"[BridgeSign] Inference module load failed: {e}")
            raise


def _warm_inference_modules():
    try:
        _load_inference_modules()
    except Exception:
        pass


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "inference_ready": bool(_inference_status["ready"]),
        "inference_loading": bool(_inference_status["loading"]),
        "inference_error": _inference_status["error"],
        "data_dir": config.DATA_DIR,
        "data_dir_exists": os.path.isdir(config.DATA_DIR),
        "data_dir_writable": os.access(config.DATA_DIR, os.W_OK),
    }), 200


@app.route("/test-api", methods=["GET", "POST"])
def test_api():
    payload = _request_payload()
    return jsonify({
        "status": "ok",
        "method": request.method,
        "content_type": request.content_type,
        "is_json": request.is_json,
        "payload_keys": sorted(payload.keys()) if hasattr(payload, "keys") else [],
        "data_dir_probe": _data_dir_probe(),
    }), 200

# ── Startup model sanity check ────────────────────────────────────
# Run a few known samples through the classifier to catch corrupted
# model files (the root cause of the "only translating J" cloud bug).
_SANITY_SAMPLES = {
    "A": [0.0,0.0,0.44396728,-0.13154586,0.78105354,-0.5508483,0.88793457,-0.9701507,1.0441452,-1.2496856,0.6495077,-0.88793457,0.6988374,-1.1756911,0.600178,-0.8714913,0.5508483,-0.6659509,0.36997274,-0.92904264,0.38641596,-1.1921344,0.32064304,-0.7974968,0.3124214,-0.5919564,0.09043778,-0.93726426,0.098659396,-1.2085776,0.082216166,-0.78927517,0.106881015,-0.5755131,-0.18909718,-0.9125994,-0.16443233,-1.134583,-0.13154586,-0.83038324,-0.098659396,-0.6577293],
    "B": [0.0,0.0,0.2739964,-0.13348544,0.4566607,-0.44260958,0.3161497,-0.7025549,0.10538323,-0.793887,0.46368623,-0.88521916,0.52691615,-1.2224455,0.53394175,-1.4261864,0.5409673,-1.6018251,0.29507306,-0.9554747,0.35127744,-1.3348544,0.3864052,-1.5807486,0.40045628,-1.7844894,0.13348544,-0.934398,0.18266428,-1.292701,0.22481757,-1.5245441,0.24589421,-1.7212595,-0.042153295,-0.8360403,-0.014051098,-1.1170623,0.021076648,-1.2997266,0.049178842,-1.4753653],
    "H": [0.0,0.0,0.15822981,-0.5142469,0.6230299,-0.7911491,1.0680512,-0.6823661,1.3845109,-0.543915,0.69225544,-1.0482725,1.4339577,-1.1076087,1.8592004,-1.1471661,2.205328,-1.1768342,0.71203417,-0.7021448,1.5130726,-0.83070654,2.0075407,-0.8999321,2.422894,-0.93948954,0.74170226,-0.33623835,1.4240683,-0.42524263,1.226281,-0.42524263,0.96915764,-0.4054639,0.77137035,-0.019778727,1.2955066,-0.17800854,1.1669449,-0.19778727,0.9592683,-0.17800854],
    "W": [0.0,0.0,0.24238947,-0.121194735,0.37873355,-0.3332855,0.25753883,-0.51507765,0.098470725,-0.68929505,0.40145755,-0.93168455,0.60597366,-1.2876941,0.7120191,-1.5073595,0.7877658,-1.7194504,0.17421743,-0.98470724,0.26511347,-1.4543368,0.2878375,-1.7345996,0.2878375,-1.9769892,-0.037873354,-0.9241099,-0.13634408,-1.333142,-0.20451611,-1.5982555,-0.26511347,-1.8254957,-0.21966545,-0.7574671,-0.17421743,-1.0225806,-0.06817204,-0.89381117,0.015149342,-0.7120191],
}

def _run_sanity_check():
    """Test the model on known samples at startup."""
    if _classifier.pipeline is None:
        print("[SanityCheck] WARNING: No model loaded -- skipping sanity check.")
        return
    passed, failed = 0, 0
    for expected_label, features in _SANITY_SAMPLES.items():
        pred_label, conf = _classifier.predict(features)
        if pred_label == expected_label:
            passed += 1
        else:
            failed += 1
            print(f"[SanityCheck] FAIL: Expected '{expected_label}', got '{pred_label}' (conf={conf:.2f})")
    if failed == 0:
        print(f"[SanityCheck] OK: All {passed} test samples passed -- model is healthy.")
    else:
        print("=" * 52)
        print("  !!!  MODEL SANITY CHECK FAILED  !!!")
        print(f"  {failed}/{passed+failed} test predictions were WRONG.")
        print("  The model file is likely corrupted or was trained")
        print("  with a different scikit-learn version.")
        print("  Re-run: python model_trainer.py")
        print("=" * 52)


if os.environ.get("BRIDGESIGN_WARM_INFERENCE", "1") == "1":
    threading.Thread(target=_warm_inference_modules, daemon=True).start()

tts       = None  # Browser speech synthesis handles speaking in the web UI.
tracker   = SessionTracker()
emergency = EmergencyPhrases()
learning  = None

# ── Per-user inference session state ─────────────────────────────
# Each user gets their own pipeline state so multiple users don't interfere.
_inference_sessions = {}
_inf_lock = threading.Lock()

def _get_inference_session(username):
    """Return (or create) the per-user ML pipeline state dict."""
    with _inf_lock:
        if username not in _inference_sessions:
            from gesture_feature_extractor import GestureFeatureExtractor
            from word_assembler import WordAssembler

            _inference_sessions[username] = {
                "gesture_ext":            GestureFeatureExtractor(),
                "assembler":              WordAssembler(),
                "prev_result_key":        "",
                "consecutive":            0,
                "last_emitted_key":       "",
                "last_hand_ts":           0.0,
                "gesture_cooldown_until": 0.0,
                "prev_features":           None,
                "last_gesture_frame":      -999,
                "cached_gesture_label":    "",
                "cached_gesture_conf":     0.0,
                "mode":                   "letter",
                "stt_history":            [],
                "frame_count":            0,
                "tracked_count":          0,
            }
        return _inference_sessions[username]

def _reset_inference_session(username):
    """Reset pipeline state (called on camera start/stop and mode switch)."""
    with _inf_lock:
        if username in _inference_sessions:
            s = _inference_sessions[username]
            s["gesture_ext"].clear()
            s["assembler"].reset()
            s["prev_result_key"]        = ""
            s["consecutive"]            = 0
            s["last_emitted_key"]       = ""
            s["last_hand_ts"]           = 0.0
            s["gesture_cooldown_until"] = 0.0
            s["prev_features"]          = None
            s["last_gesture_frame"]      = -999
            s["cached_gesture_label"]    = ""
            s["cached_gesture_conf"]     = 0.0
            s["frame_count"]            = 0
            s["tracked_count"]          = 0


def _display_label(label):
    return str(label or "").replace("_", " ").strip()


def _label_unit(label):
    clean = str(label or "").replace("_", "")
    return "letter" if len(clean) == 1 and clean.isalpha() else "word"


def _predict_static_sign(features):
    """Run the alphabet/static sign classifier and apply confidence gating."""
    label, conf = _classifier.predict(features)
    if conf < config.MIN_PREDICTION_CONFIDENCE or label in ("", "Unknown", "Error"):
        return "", 0.0
    # J and Z are motion letters; let the gesture model own them in live use.
    if label in GESTURE_ONLY_LABELS and _gesture_classifier.is_available():
        return "", 0.0
    return label, conf


def _count_direction_changes(values, min_step=0.015):
    """Count meaningful sign changes in a 1D trajectory."""
    changes = 0
    last_sign = 0
    for delta in np.diff(values):
        if abs(delta) < min_step:
            continue
        sign = 1 if delta > 0 else -1
        if last_sign and sign != last_sign:
            changes += 1
        last_sign = sign
    return changes


def _motion_letter_scores(raw_seq):
    """
    Return heuristic scores for distinguishing Z vs J.
    Z should look like a mostly horizontal zig-zag of the index finger.
    J should look like a mostly downward curved hook of the pinky finger.
    """
    index_x = raw_seq[:, 16]
    index_y = raw_seq[:, 17]
    pinky_x = raw_seq[:, 40]
    pinky_y = raw_seq[:, 41]

    index_x_range = float(np.ptp(index_x))
    index_y_range = float(np.ptp(index_y))
    pinky_x_range = float(np.ptp(pinky_x))
    pinky_y_range = float(np.ptp(pinky_y))

    z_score = 0.0
    if index_x_range > 0.10:
        z_score += 0.40
    if index_x_range > index_y_range * 1.15:
        z_score += 0.25
    z_score += min(0.35, 0.18 * _count_direction_changes(index_x))
    if abs(float(index_x[-1] - index_x[0])) > 0.08:
        z_score += 0.10

    j_score = 0.0
    pinky_drop = float(pinky_y[-1] - pinky_y[0])
    if pinky_drop > 0.10:
        j_score += 0.35
    if pinky_y_range > 0.12:
        j_score += 0.25
    if pinky_y_range > pinky_x_range * 0.80:
        j_score += 0.15
    j_score += min(0.20, 0.20 * _count_direction_changes(pinky_x))
    if abs(float(pinky_x[-1] - pinky_x[0])) > 0.04:
        j_score += 0.10

    return z_score, j_score


def _refine_motion_letter_prediction(raw_seq, label, conf):
    """Use simple path heuristics to reduce J/Z confusion."""
    if label not in GESTURE_ONLY_LABELS or raw_seq is None:
        return label, conf

    z_score, j_score = _motion_letter_scores(raw_seq)

    if z_score >= j_score + 0.18:
        return "Z", conf if label == "Z" else max(conf - 0.06, 0.0)
    if j_score >= z_score + 0.18:
        return "J", conf if label == "J" else max(conf - 0.06, 0.0)

    # If the gesture model itself is unsure and the path shape is weak too,
    # drop the result rather than forcing a bad J/Z guess.
    if conf < 0.68 and max(z_score, j_score) < 0.55:
        return "", 0.0

    return label, conf


def _clear_gesture_cache(state):
    state["cached_gesture_label"] = ""
    state["cached_gesture_conf"] = 0.0


def _cache_gesture_candidate(state, label, conf):
    state["last_gesture_frame"] = state.get("frame_count", 0)
    state["cached_gesture_label"] = label or ""
    state["cached_gesture_conf"] = float(conf or 0.0)
    return state["cached_gesture_label"], state["cached_gesture_conf"]


def _predict_gesture_candidate(state, motion=None):
    """Run the motion-gesture classifier without mixing it into the static path."""
    if not _gesture_classifier.is_available() or not state["gesture_ext"].is_ready():
        _clear_gesture_cache(state)
        return "", 0.0

    if motion is None:
        motion = state["gesture_ext"].get_motion_magnitude()
    if motion <= GESTURE_MOTION_THRESHOLD:
        _clear_gesture_cache(state)
        return "", 0.0

    frame_count = state.get("frame_count", 0)
    if frame_count - state.get("last_gesture_frame", -999) < GESTURE_INFERENCE_EVERY_N:
        return state.get("cached_gesture_label", ""), state.get("cached_gesture_conf", 0.0)

    if _gesture_classifier.is_lstm():
        raw_seq = state["gesture_ext"].get_raw_sequence()
        if raw_seq is None:
            return _cache_gesture_candidate(state, "", 0.0)
        g_label, g_conf = _gesture_classifier.predict(raw_seq)
        g_label, g_conf = _refine_motion_letter_prediction(raw_seq, g_label, g_conf)
        if not g_label or g_label == "STATIC" or g_conf < GESTURE_LSTM_MIN_CONF:
            return _cache_gesture_candidate(state, "", 0.0)
        return _cache_gesture_candidate(state, g_label, g_conf)

    g_feats = state["gesture_ext"].extract_gesture_features()
    g_label, g_conf = _gesture_classifier.predict(g_feats)
    raw_seq = state["gesture_ext"].get_raw_sequence()
    g_label, g_conf = _refine_motion_letter_prediction(raw_seq, g_label, g_conf)
    if not g_label or g_conf < GESTURE_MIN_CONFIDENCE:
        return _cache_gesture_candidate(state, "", 0.0)
    return _cache_gesture_candidate(state, g_label, g_conf)


def _select_live_output(static_label, static_conf, gesture_label, gesture_conf):
    """
    Choose the live result while keeping hand signs and gestures separate.
    A gesture can win even if the static classifier is unsure, but it must
    beat a confident static sign by a small margin to avoid mode crossover.
    """
    if gesture_label:
        gesture_wins = not static_label
        if static_label:
            if _gesture_classifier.is_lstm():
                gesture_wins = gesture_conf >= max(
                    GESTURE_LSTM_MIN_CONF,
                    static_conf + GESTURE_DECISION_MARGIN,
                )
            else:
                gesture_wins = gesture_conf > static_conf * GESTURE_MARGIN
        if gesture_wins:
            return gesture_label, gesture_conf, "gesture"

    if static_label:
        return static_label, static_conf, "handsign"

    return "", 0.0, ""


def _decode_posted_frame():
    f = request.files.get("frame")
    if f is None:
        return None, "No frame"

    import cv2

    img_bytes = f.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, "Invalid frame"
    return frame, ""


def _landmark_payload(hand_landmarks):
    return [
        {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(getattr(lm, "z", 0.0)),
            "visibility": float(getattr(lm, "visibility", 1.0)),
        }
        for lm in hand_landmarks
    ]


def _bbox_from_landmarks(hand_landmarks):
    xs = [float(lm.x) for lm in hand_landmarks]
    ys = [float(lm.y) for lm in hand_landmarks]
    if not xs or not ys:
        return None
    return {
        "x": min(xs),
        "y": min(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
    }


def _handedness_label(results, index):
    try:
        handedness = results.handedness[index]
        if handedness:
            return handedness[0].category_name
    except Exception:
        pass
    return "Hand"


def _hand_tracking_payload(results):
    hands = []
    if results is None or not getattr(results, "hand_landmarks", None):
        return hands

    for index, hand_lm in enumerate(results.hand_landmarks):
        hands.append({
            "index": index,
            "label": _handedness_label(results, index),
            "landmarks": _landmark_payload(hand_lm),
            "bbox": _bbox_from_landmarks(hand_lm),
        })
    return hands

# ── Auth helpers ──────────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return jsonify({"error": "Not logged in"}), 401
        return f(*args, **kwargs)
    return decorated


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    if isinstance(e, HTTPException):
        return e
    print("[BridgeSign] Unhandled exception:")
    traceback.print_exc()
    wants_json = request.path.startswith("/api/") or request.path in {"/health", "/test-api"}
    if wants_json:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500
    return "Internal server error", 500

# ── Auth routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session["username"])

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    message = None
    just_registered = request.method == "GET" and request.args.get("registered") == "1"
    if just_registered:
        session.clear()
        message = "Account created! Please sign in with your new credentials."
    elif request.method == "GET" and "username" in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        payload = _request_payload()
        username = payload.get("username", "").strip()
        password = payload.get("password", "")
        users = load_users()
        print(f"[Auth] Login attempt user={username!r} found={username in users}")
        if username in users and check_password_hash(users[username]["password"], password):
            session["username"] = username
            if request.is_json:
                return jsonify({"ok": True, "redirect": url_for("index")})
            return redirect(url_for("index"))
        error = "Invalid username or password."
        if request.is_json:
            return jsonify({"ok": False, "error": error}), 401
    return render_template("login.html", error=error, message=message)

@app.route("/register", methods=["GET", "POST"])
def register():
    if "username" in session:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        payload = _request_payload()
        username = payload.get("username", "").strip()
        password = payload.get("password", "")
        if not username or not password:
            error = "Username and password are required."
        else:
            with _users_lock:
                try:
                    users = load_users()
                    if username in users:
                        error = "Username already taken."
                    else:
                        users[username] = {"password": generate_password_hash(password)}
                        save_users(users)
                        print(f"[Auth] Created account user={username!r} users_file={USERS_FILE}")
                except Exception as e:
                    print(f"[Auth] Registration error: {e}")
                    error = f"Database write error: Could not save credentials. Check BRIDGESIGN_DATA_DIR permissions on your server."
            if not error:
                session.clear()
                if request.is_json:
                    return jsonify({"ok": True, "redirect": url_for("login", registered="1")})
                return redirect(url_for("login", registered="1"))
        if request.is_json:
            return jsonify({"ok": False, "error": error}), 400
    return render_template("register.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ── Camera session management (no thread — browser owns the camera) ──
@app.route("/api/camera/start", methods=["POST"])
@login_required
def start_camera():
    """Reset per-user ML state to prepare for a new session."""
    _reset_inference_session(session["username"])
    return jsonify({"ok": True})

@app.route("/api/camera/stop", methods=["POST"])
@login_required
def stop_camera():
    _reset_inference_session(session["username"])
    return jsonify({"ok": True})

# ── Core: per-frame inference endpoint ───────────────────────────
@app.route("/api/infer_frame", methods=["POST"])
@login_required
def infer_frame():
    """
    Receive a single JPEG frame from the browser camera, run the full
    MediaPipe + classifier pipeline, and return the result as JSON.

    The browser captures its own camera feed via getUserMedia and POSTs
    one frame every ~100 ms (≈10 fps). This replaces the old MJPEG stream
    + background camera thread entirely.
    """
    username = session["username"]
    s = _get_inference_session(username)
    s["frame_count"] = s.get("frame_count", 0) + 1

    # ── Decode incoming JPEG ──────────────────────────────────────
    frame, decode_error = _decode_posted_frame()
    if decode_error == "No frame":
        return jsonify({"error": decode_error}), 400

    empty_response = {
        "hand_state": "no_hand", "label": "", "confidence": 0.0,
        "result_source": "", "result_unit": "",
        "static_label": "", "static_confidence": 0.0,
        "gesture_label": "", "gesture_confidence": 0.0,
        "motion_magnitude": 0.0, "frame_motion": 0.0,
        "mode": s["mode"], "word_buffer": "", "last_word": "",
        "sentence": "", "completed_sentence": "", "completed_word": "",
        "frame_count": s.get("frame_count", 0),
        "tracked_count": s.get("tracked_count", 0),
        "consecutive": s.get("consecutive", 0),
        "landmark_count": 0,
        "tracking_parts": [],
        "hand_bbox": None,
    }
    if frame is None:
        return jsonify(empty_response)

    try:
        _load_inference_modules()
    except Exception as e:
        return jsonify({
            "error": "Inference modules failed to load",
            "detail": str(e),
        }), 503

    now = time.time()
    hand_state          = "no_hand"
    confirmed_label     = ""
    confirmed_conf      = 0.0
    confirmed_source    = ""
    confirmed_unit      = ""
    static_label        = ""
    static_conf         = 0.0
    gesture_label       = ""
    gesture_conf        = 0.0
    motion_magnitude    = 0.0
    frame_motion        = 0.0
    lm_list             = []
    landmark_count      = 0
    tracking_parts      = []
    hand_bbox           = None

    # ── Hand detection ────────────────────────────────────────────
    with _detector_lock:
        frame, results = _detector.find_hands(frame, draw=False)
        hand_detected = (
            results is not None
            and results.hand_landmarks
            and len(results.hand_landmarks) > 0
        )
        if hand_detected:
            lm_list = _detector.get_landmarks(frame, hand_no=0)

    hand_detected = (
        results is not None
        and results.hand_landmarks
        and len(results.hand_landmarks) > 0
    )

    if hand_detected:
        s["last_hand_ts"] = now
        s["tracked_count"] = s.get("tracked_count", 0) + 1
        landmark_count = len(results.hand_landmarks[0]) if results.hand_landmarks else 0
        tracking_parts = ["hand"]
        hand_bbox = _bbox_from_landmarks(results.hand_landmarks[0])
        if lm_list:
            features = _extractor.extract_features(lm_list)
            if features is not None:
                prev_features = s.get("prev_features")
                if prev_features is not None:
                    frame_motion = float(np.sum(np.abs(np.asarray(features) - np.asarray(prev_features))))
                s["prev_features"] = np.asarray(features).copy()
                s["gesture_ext"].push_frame(features)
                motion_magnitude = s["gesture_ext"].get_motion_magnitude()
                active_motion = (
                    motion_magnitude > GESTURE_STATIC_SUPPRESS_THRESHOLD
                    or frame_motion > GESTURE_FRAME_MOTION_THRESHOLD
                )
                if active_motion:
                    tracking_parts = ["hand", "motion"]
                static_label, static_conf = _predict_static_sign(features)
                if (
                    active_motion
                    and _gesture_classifier.is_available()
                    and _label_unit(static_label) == "letter"
                ):
                    static_label, static_conf = "", 0.0

                if now > s["gesture_cooldown_until"]:
                    gesture_label, gesture_conf = _predict_gesture_candidate(s, motion_magnitude)

                label_raw, conf, source = _select_live_output(
                    static_label,
                    static_conf,
                    gesture_label,
                    gesture_conf,
                )

                if not label_raw:
                    s["consecutive"]     = 0
                    s["prev_result_key"] = ""
                    hand_state = "low_confidence"
                else:
                    hand_state = "recognised"
                    result_key = f"{source}:{label_raw}"
                    if result_key == s["prev_result_key"]:
                        s["consecutive"] += 1
                    else:
                        s["consecutive"]     = 1
                        s["prev_result_key"] = result_key

                    if s["consecutive"] >= config.CONSECUTIVE_THRESHOLD:
                        confirmed_label  = _display_label(label_raw)
                        confirmed_conf   = conf
                        confirmed_source = source
                        confirmed_unit   = _label_unit(label_raw)

                        if result_key != s["last_emitted_key"]:
                            s["last_emitted_key"] = result_key
                            tracker.log_translation(
                                confirmed_label,
                                conf,
                                f"camera_{source}",
                            )
                            if source == "gesture":
                                s["gesture_cooldown_until"] = now + GESTURE_COOLDOWN_SEC
                            if s["mode"] == "word":
                                if confirmed_unit == "letter":
                                    s["assembler"].push_letter(label_raw)
                                else:
                                    s["assembler"].push_word(label_raw)
    else:
        s["consecutive"]      = 0
        s["prev_result_key"]  = ""
        s["last_emitted_key"] = ""
        s["prev_features"]    = None
        s["gesture_ext"].clear()

    # ── Word assembler tick ───────────────────────────────────────
    grace_present = hand_detected or (now - s["last_hand_ts"]) < config.HAND_LOST_GRACE_SEC
    asm = s["assembler"].tick(hand_present=grace_present)

    # ── Optional debug logging ────────────────────────────────────
    if _DEBUG_EVERY_N > 0:
        s.setdefault("_dbg_frame", 0)
        s["_dbg_frame"] = (s["_dbg_frame"] + 1) % _DEBUG_EVERY_N
        if s["_dbg_frame"] == 0 and hand_detected:
            print(f"[DBG] hand_state={hand_state} | static=({static_label},{static_conf:.2f}) "
                  f"| gesture=({gesture_label},{gesture_conf:.2f}) "
                  f"| motion={motion_magnitude:.2f}/{frame_motion:.2f} "
                  f"| consec={s['consecutive']} | key={s['prev_result_key']!r}")

    return jsonify({
        "hand_state":         hand_state,
        "label":              confirmed_label,
        "confidence":         confirmed_conf,
        "result_source":      confirmed_source,
        "result_unit":        confirmed_unit,
        "static_label":       _display_label(static_label),
        "static_confidence":  static_conf,
        "gesture_label":      _display_label(gesture_label),
        "gesture_confidence": gesture_conf,
        "motion_magnitude":   motion_magnitude,
        "frame_motion":       frame_motion,
        "mode":               s["mode"],
        "word_buffer":        asm["word_buffer"]        if s["mode"] == "word" else "",
        "last_word":          asm["last_word"]          if s["mode"] == "word" else "",
        "sentence":           asm["sentence"]           if s["mode"] == "word" else "",
        "completed_sentence": (asm["completed_sentence"] or "") if s["mode"] == "word" else "",
        "completed_word":     (asm["completed_word"]     or "") if s["mode"] == "word" else "",
        "frame_count":        s.get("frame_count", 0),
        "tracked_count":      s.get("tracked_count", 0),
        "consecutive":        s.get("consecutive", 0),
        "landmark_count":     landmark_count,
        "tracking_parts":     tracking_parts,
        "hand_bbox":          hand_bbox,
    })


@app.route("/api/infer_landmarks", methods=["POST"])
@login_required
def infer_landmarks():
    """
    Receive pre-extracted landmarks from the browser, run the classification
    pipeline directly without running MediaPipe on the server.

    Expected JSON payload:
    {
        "landmarks": [[id, cx, cy], ...]
    }
    """
    username = session["username"]
    s = _get_inference_session(username)
    s["frame_count"] = s.get("frame_count", 0) + 1

    data = request.get_json() or {}
    lm_list = data.get("landmarks")

    empty_response = {
        "hand_state": "no_hand", "label": "", "confidence": 0.0,
        "result_source": "", "result_unit": "",
        "static_label": "", "static_confidence": 0.0,
        "gesture_label": "", "gesture_confidence": 0.0,
        "motion_magnitude": 0.0, "frame_motion": 0.0,
        "mode": s["mode"], "word_buffer": "", "last_word": "",
        "sentence": "", "completed_sentence": "", "completed_word": "",
        "frame_count": s.get("frame_count", 0),
        "tracked_count": s.get("tracked_count", 0),
        "consecutive": s.get("consecutive", 0),
        "landmark_count": 0,
        "tracking_parts": [],
        "hand_bbox": None,
    }

    if not lm_list or len(lm_list) < 21:
        s["consecutive"]      = 0
        s["prev_result_key"]  = ""
        s["last_emitted_key"] = ""
        s["prev_features"]    = None
        s["gesture_ext"].clear()

        # Word assembler tick
        now = time.time()
        grace_present = (now - s.get("last_hand_ts", 0.0)) < config.HAND_LOST_GRACE_SEC
        asm = s["assembler"].tick(hand_present=grace_present)

        empty_response["word_buffer"] = asm["word_buffer"] if s["mode"] == "word" else ""
        empty_response["last_word"] = asm["last_word"] if s["mode"] == "word" else ""
        empty_response["sentence"] = asm["sentence"] if s["mode"] == "word" else ""
        empty_response["completed_sentence"] = (asm["completed_sentence"] or "") if s["mode"] == "word" else ""
        empty_response["completed_word"] = (asm["completed_word"] or "") if s["mode"] == "word" else ""
        return jsonify(empty_response)

    try:
        _load_inference_modules()
    except Exception as e:
        return jsonify({
            "error": "Inference modules failed to load",
            "detail": str(e),
        }), 503

    now = time.time()
    s["last_hand_ts"] = now
    s["tracked_count"] = s.get("tracked_count", 0) + 1
    landmark_count = len(lm_list)
    tracking_parts = ["hand"]

    # Calculate bounding box
    xs = [float(lm[1]) for lm in lm_list]
    ys = [float(lm[2]) for lm in lm_list]
    hand_bbox = {
        "x": min(xs),
        "y": min(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
    }

    hand_state       = "no_hand"
    confirmed_label  = ""
    confirmed_conf   = 0.0
    confirmed_source = ""
    confirmed_unit   = ""
    static_label     = ""
    static_conf      = 0.0
    gesture_label    = ""
    gesture_conf     = 0.0
    motion_magnitude = 0.0
    frame_motion     = 0.0

    features = _extractor.extract_features(lm_list)
    if features is not None:
        prev_features = s.get("prev_features")
        if prev_features is not None:
            frame_motion = float(np.sum(np.abs(np.asarray(features) - np.asarray(prev_features))))
        s["prev_features"] = np.asarray(features).copy()
        s["gesture_ext"].push_frame(features)
        motion_magnitude = s["gesture_ext"].get_motion_magnitude()
        active_motion = (
            motion_magnitude > GESTURE_STATIC_SUPPRESS_THRESHOLD
            or frame_motion > GESTURE_FRAME_MOTION_THRESHOLD
        )
        if active_motion:
            tracking_parts = ["hand", "motion"]
        static_label, static_conf = _predict_static_sign(features)
        if (
            active_motion
            and _gesture_classifier.is_available()
            and _label_unit(static_label) == "letter"
        ):
            static_label, static_conf = "", 0.0

        if now > s["gesture_cooldown_until"]:
            gesture_label, gesture_conf = _predict_gesture_candidate(s, motion_magnitude)

        label_raw, conf, source = _select_live_output(
            static_label,
            static_conf,
            gesture_label,
            gesture_conf,
        )

        if not label_raw:
            s["consecutive"]     = 0
            s["prev_result_key"] = ""
            hand_state = "low_confidence"
        else:
            hand_state = "recognised"
            result_key = f"{source}:{label_raw}"
            if result_key == s["prev_result_key"]:
                s["consecutive"] += 1
            else:
                s["consecutive"]     = 1
                s["prev_result_key"] = result_key

            if s["consecutive"] >= config.CONSECUTIVE_THRESHOLD:
                confirmed_label  = _display_label(label_raw)
                confirmed_conf   = conf
                confirmed_source = source
                confirmed_unit   = _label_unit(label_raw)

                if result_key != s["last_emitted_key"]:
                    s["last_emitted_key"] = result_key
                    tracker.log_translation(
                        confirmed_label,
                        conf,
                        f"camera_{source}",
                    )
                    if source == "gesture":
                        s["gesture_cooldown_until"] = now + GESTURE_COOLDOWN_SEC
                    if s["mode"] == "word":
                        if confirmed_unit == "letter":
                            s["assembler"].push_letter(label_raw)
                        else:
                            s["assembler"].push_word(label_raw)

    # ── Word assembler tick ───────────────────────────────────────
    asm = s["assembler"].tick(hand_present=True)

    return jsonify({
        "hand_state":         hand_state,
        "label":              confirmed_label,
        "confidence":         confirmed_conf,
        "result_source":      confirmed_source,
        "result_unit":        confirmed_unit,
        "static_label":       _display_label(static_label),
        "static_confidence":  static_conf,
        "gesture_label":      _display_label(gesture_label),
        "gesture_confidence": gesture_conf,
        "motion_magnitude":   motion_magnitude,
        "frame_motion":       frame_motion,
        "mode":               s["mode"],
        "word_buffer":        asm["word_buffer"]        if s["mode"] == "word" else "",
        "last_word":          asm["last_word"]          if s["mode"] == "word" else "",
        "sentence":           asm["sentence"]           if s["mode"] == "word" else "",
        "completed_sentence": (asm["completed_sentence"] or "") if s["mode"] == "word" else "",
        "completed_word":     (asm["completed_word"]     or "") if s["mode"] == "word" else "",
        "frame_count":        s.get("frame_count", 0),
        "tracked_count":      s.get("tracked_count", 0),
        "consecutive":        s.get("consecutive", 0),
        "landmark_count":     landmark_count,
        "tracking_parts":     tracking_parts,
        "hand_bbox":          hand_bbox,
    })


@app.route("/api/track_frame", methods=["POST"])
@login_required
def track_frame():
    """Return normalized hand + pose landmark coordinates for motion recording."""
    frame, decode_error = _decode_posted_frame()
    if decode_error == "No frame":
        return jsonify({"error": decode_error}), 400
    if frame is None:
        return jsonify({
            "ok": False,
            "source": "server_hand",
            "hands": [],
            "pose": [],
            "parts": [],
            "landmark_count": 0,
            "pose_landmark_count": 0,
            "error": decode_error or "Invalid frame",
        })

    try:
        _load_inference_modules()
    except Exception as e:
        return jsonify({
            "ok": False,
            "source": "server_hand_pose",
            "hands": [],
            "pose": [],
            "parts": [],
            "landmark_count": 0,
            "pose_landmark_count": 0,
            "error": "Inference modules failed to load",
            "detail": str(e),
        }), 503

    # ── Hand detection ────────────────────────────────────────
    with _motion_detector_lock:
        _, results = _motion_detector.find_hands(frame, draw=False)

    hands = _hand_tracking_payload(results)
    parts = [f"{str(hand['label']).lower()} hand" for hand in hands]
    hand_landmark_count = sum(len(hand["landmarks"]) for hand in hands)

    # ── Pose detection (upper body) ──────────────────────────
    pose_landmarks_raw = None
    pose_payload = []
    pose_parts = []
    try:
        with _pose_detector_lock:
            pose_landmarks_raw = _pose_detector.detect(frame)
        if pose_landmarks_raw:
            pose_payload = _pose_detector.get_landmark_payload(pose_landmarks_raw)
            pose_parts = _pose_detector.get_upper_body_parts(pose_landmarks_raw)
    except Exception as e:
        print(f"[track_frame] Pose detection error: {e}")

    all_parts = list(set(parts + pose_parts))
    total_landmarks = hand_landmark_count + len(pose_payload)

    return jsonify({
        "ok": True,
        "source": "server_hand_pose",
        "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "hands": hands,
        "hand_count": len(hands),
        "pose": pose_payload,
        "pose_landmark_count": len(pose_payload),
        "parts": all_parts,
        "landmark_count": total_landmarks,
    })

# ── Word assembler: manual flush (Next Word button) ───────────────
@app.route("/api/assembler/flush", methods=["POST"])
@login_required
def assembler_flush():
    """
    Immediately commit the current letter buffer as a word.
    Called when the user presses the 'Next Word' button in the UI.
    Returns the updated assembler state so the frontend can refresh.
    """
    username = session["username"]
    s = _get_inference_session(username)
    word = s["assembler"].manual_flush()
    asm  = s["assembler"].tick(hand_present=False)
    return jsonify({
        "ok":               True,
        "flushed_word":     word,
        "word_buffer":      asm["word_buffer"],
        "last_word":        asm["last_word"],
        "sentence":         asm["sentence"],
        "completed_word":   asm["completed_word"] or word,
        "completed_sentence": asm["completed_sentence"] or "",
    })

# ── Mode toggle ───────────────────────────────────────────────────
@app.route("/api/mode", methods=["POST"])
@login_required
def set_mode():
    data = request.get_json()
    mode = (data or {}).get("mode", "letter")
    if mode not in ("letter", "word"):
        return jsonify({"error": "Invalid mode. Use 'letter' or 'word'."}), 400
    username = session["username"]
    s = _get_inference_session(username)
    with _inf_lock:
        s["mode"] = mode
    _reset_inference_session(username)
    # Re-apply mode after reset
    with _inf_lock:
        _inference_sessions[username]["mode"] = mode
    return jsonify({"ok": True, "mode": mode})

# ── Speech-to-Text (text comes from browser Web Speech API) ───────
@app.route("/api/stt/text", methods=["POST"])
@login_required
def stt_receive_text():
    """
    The browser's SpeechRecognition fires, sends the transcript here.
    We generate sign guidance and return it immediately — no polling needed.
    """
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"guidance": [], "history": []}), 200

    from sign_text_processor import process_speech_gloss

    gloss_str, guidance = process_speech_gloss(text)
    username = session["username"]
    s = _get_inference_session(username)
    entry = {
        "text":     text,
        "gloss_order": gloss_str,
        "guidance": guidance,
        "ts":       time.strftime("%H:%M:%S"),
    }
    with _inf_lock:
        s["stt_history"].insert(0, entry)
        s["stt_history"] = s["stt_history"][:20]
        history_snapshot = s["stt_history"][:10]

    return jsonify({"guidance": guidance, "gloss_order": gloss_str, "history": history_snapshot})

@app.route("/api/avatar/animations")
@login_required
def list_avatar_animations():
    """Return saved VRM clips and local CWASA SiGML signs."""
    anim_dir = os.path.join(app.static_folder, "avatar", "animations")
    sigml_dir = os.path.join(app.static_folder, "avatar", "sigml", "asl")
    vrm_labels = []
    cwasa_labels = []
    if os.path.isdir(anim_dir):
        for filename in os.listdir(anim_dir):
            if not filename.lower().endswith(".json"):
                continue
            label = os.path.splitext(filename)[0].upper()
            if re.fullmatch(r"[A-Z][A-Z0-9_]{0,39}", label):
                vrm_labels.append(label)
    if os.path.isdir(sigml_dir):
        for filename in os.listdir(sigml_dir):
            if not filename.lower().endswith(".sigml"):
                continue
            label = os.path.splitext(filename)[0].upper()
            if re.fullmatch(r"[A-Z][A-Z0-9_]{0,39}", label):
                cwasa_labels.append(label)
    labels = vrm_labels + cwasa_labels
    labels = sorted(set(labels), key=lambda value: (len(value) > 1, value))
    return jsonify({
        "animations": labels,
        "cwasa": sorted(set(cwasa_labels), key=lambda value: (len(value) > 1, value)),
        "vrm": sorted(set(vrm_labels), key=lambda value: (len(value) > 1, value)),
        "count": len(labels),
    })

@app.route("/api/avatar/animation", methods=["POST"])
@login_required
def save_avatar_animation():
    data = request.get_json() or {}
    label = str(data.get("label", "")).strip().upper().replace(" ", "_")
    clip = data.get("clip") or {}

    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,39}", label):
        return jsonify({"error": "Invalid sign label."}), 400

    if not isinstance(clip, dict) or not isinstance(clip.get("tracks"), list):
        return jsonify({"error": "Invalid animation clip."}), 400

    duration = clip.get("duration", 0)
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid clip duration."}), 400
    if duration <= 0 or duration > 20:
        return jsonify({"error": "Clip duration must be between 0 and 20 seconds."}), 400

    safe_clip = {
        "name": label,
        "duration": duration,
        "source": "motion_recorder",
        "tracks": [],
    }

    for track in clip["tracks"]:
        if not isinstance(track, dict):
            return jsonify({"error": "Invalid track."}), 400
        name = str(track.get("name", ""))
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*\.quaternion", name):
            return jsonify({"error": f"Invalid track name: {name}"}), 400
        times = track.get("times")
        values = track.get("values")
        if not isinstance(times, list) or not isinstance(values, list):
            return jsonify({"error": f"Invalid data for track {name}."}), 400
        if len(times) < 2 or len(times) > 240 or len(values) != len(times) * 4:
            return jsonify({"error": f"Invalid keyframe count for track {name}."}), 400
        try:
            clean_times = [float(v) for v in times]
            clean_values = [float(v) for v in values]
        except (TypeError, ValueError):
            return jsonify({"error": f"Non-numeric values in track {name}."}), 400
        if any(not np.isfinite(v) for v in clean_times + clean_values):
            return jsonify({"error": f"Non-finite values in track {name}."}), 400
        safe_clip["tracks"].append({
            "name": name,
            "type": "quaternion",
            "times": clean_times,
            "values": clean_values,
        })

    out_dir = os.path.join(app.static_folder, "avatar", "animations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(safe_clip, f, indent=2)

    return jsonify({
        "ok": True,
        "label": label,
        "path": f"/static/avatar/animations/{label}.json",
        "tracks": len(safe_clip["tracks"]),
    })

# ── Existing API endpoints (unchanged) ───────────────────────────
@app.route("/api/stats")
@login_required
def get_stats():
    return jsonify(tracker.get_stats())

@app.route("/api/translate_image", methods=["POST"])
@login_required
def translate_image():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    f = request.files["image"]
    path = os.path.join(config.DATA_DIR, "upload_tmp.jpg")
    f.save(path)
    from image_translator import ImageTranslator
    label, conf, _ = ImageTranslator().translate(path)
    tracker.log_translation(label, conf, "image")
    return jsonify({"label": label, "confidence": f"{conf:.0%}"})

@app.route("/api/learn/lesson")
@login_required
def get_lesson():
    global learning
    if learning is None:
        from learning_mode import LearningMode
        learning = LearningMode()
    sign = request.args.get("sign", "Hello")
    return jsonify({"sign": sign, "tip": learning.get_lesson(sign)})

@app.route("/api/emergency/all")
@login_required
def emergency_all():
    builtin = emergency.get_phrases()
    custom  = load_user_phrases(session["username"])
    result  = {str(k): v for k, v in builtin.items()}
    result.update(custom)
    return jsonify(result)

@app.route("/api/emergency/custom", methods=["POST"])
@login_required
def add_custom_phrase():
    data   = request.get_json()
    phrase = (data or {}).get("phrase", "").strip()
    if not phrase:
        return jsonify({"error": "Empty phrase"}), 400
    custom = load_user_phrases(session["username"])
    cid    = f"c_{int(time.time())}"
    custom[cid] = phrase
    save_user_phrases(session["username"], custom)
    return jsonify({"ok": True, "id": cid, "phrase": phrase})

@app.route("/api/emergency/custom/<cid>", methods=["DELETE"])
@login_required
def delete_custom_phrase(cid):
    custom = load_user_phrases(session["username"])
    custom.pop(cid, None)
    save_user_phrases(session["username"], custom)
    return jsonify({"ok": True})

# ── PWA: serve service worker from root scope ─────────────────────
@app.route("/sw.js")
def service_worker():
    from flask import send_from_directory
    resp = send_from_directory("static", "sw.js")
    resp.headers["Content-Type"] = "application/javascript"
    resp.headers["Service-Worker-Allowed"] = "/"
    return resp

# ── WebRTC Call Room ──────────────────────────────────────────────
@app.route("/call/new")
@login_required
def new_call_room():
    # Generate random 8-character room ID
    alphabet = string.ascii_lowercase + string.digits
    room_id = ''.join(secrets.choice(alphabet) for _ in range(8))
    return redirect(url_for("call_room_view", room_id=room_id))

@app.route("/call/<room_id>")
@login_required
def call_room_view(room_id):
    if not ROOM_ID_RE.fullmatch(room_id):
        return redirect(url_for("index"))
    return render_template("call.html", room_id=room_id, username=session["username"])

@sock.route("/ws/call/<room_id>")
def call_signaling(ws, room_id):
    if "username" not in session or not ROOM_ID_RE.fullmatch(room_id):
        try:
            ws.send(json.dumps({"type": "error", "message": "Unauthorized call room"}))
        except Exception:
            pass
        ws.close()
        return

    if not room_manager.join_room(room_id, ws):
        ws.close()
        return

    try:
        while True:
            data = ws.receive()
            if data:
                room_manager.broadcast(room_id, ws, data)
    except Exception as e:
        print(f"[WebSocket] Connection error/close: {e}")
    finally:
        room_manager.leave_room(room_id, ws)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("BRIDGESIGN_HOST", "127.0.0.1")
    print("=" * 52)
    print(f"  BridgeSign PWA")
    print(f"  Open in browser: http://127.0.0.1:{port}")
    print(f"  Also works     : http://localhost:{port}")
    print("  Do NOT use     : http://0.0.0.0:{port}".format(port=port))
    print("  Do NOT use LAN/IP URLs for camera/mic unless HTTPS is enabled.")
    print("=" * 52)
    app.run(debug=False, host=host, port=port, threaded=True)
