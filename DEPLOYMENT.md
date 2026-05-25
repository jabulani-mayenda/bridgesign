# BridgeSign Deployment

## Phone And Call Testing Requirements

Camera and microphone access require HTTPS on phones. A plain LAN URL such as `http://192.168.x.x:5000` will usually block media access.

The call module also uses WebSockets for signaling, so deploy as a dynamic web service, not a static site.

## Railway Deployment

1. Push this project to a GitHub repository.
2. Go to [railway.com](https://railway.app) and create a new project from that repository.
3. Railway will auto-detect the `Dockerfile` and `railway.json`.
4. In your Railway service settings, add these environment variables:
   - `SECRET_KEY` — set to a random string
   - `BRIDGESIGN_HOST` = `0.0.0.0`
   - `BRIDGESIGN_WARM_INFERENCE` = `0`
   - `BRIDGESIGN_DATA_DIR` = `/tmp/bridgesign-data`
   - `RAILWAY_ENVIRONMENT` = `production`
5. Railway auto-assigns `PORT` — do not set it manually.
6. Deploy. Railway will build the Docker image and give you a public HTTPS URL.
7. Open the deployed URL on your phone or browser.

## Render Web Service

1. Push this project to a GitHub repository.
2. In Render, create a new Web Service from that repository.
3. Use the Docker runtime. The included `Dockerfile` installs the Linux shared libraries required by MediaPipe, including `libGLESv2.so.2`.
4. If Render asks for a start command, leave it blank so it uses the Dockerfile `CMD`, or use:
   `gunicorn app:app --worker-class gthread --workers 1 --threads 8 --bind 0.0.0.0:$PORT --timeout 120`
5. Set the health check path to `/health` if Render does not read it from `render.yaml`.
6. Confirm these environment variables if Render does not read them from `render.yaml`:
   `PORT=10000`, `HOSTNAME=0.0.0.0`, `BRIDGESIGN_HOST=0.0.0.0`, `BRIDGESIGN_WARM_INFERENCE=0`, `BRIDGESIGN_DATA_DIR=/tmp/bridgesign-data`, `RENDER=true`, `GUNICORN_TIMEOUT=180`.
7. If you see **502 Bad Gateway**, check Render Logs for `WORKER TIMEOUT`, `SIGKILL`, or `Out of memory`. The sign model must stay under ~80MB on disk; use at least the **Starter** plan (512MB RAM) or disable warm inference (`BRIDGESIGN_WARM_INFERENCE=0`).
7. Set `SECRET_KEY` to a random value if Render does not generate it from `render.yaml`.
8. Open the deployed `https://...onrender.com` URL on your phone.

If an existing Render service was created with the Python native runtime and Render will not let you change its runtime, create a new Web Service with Docker selected. MediaPipe needs OS-level packages that are not present in Render's native Python image.

The Render service sets `BRIDGESIGN_WARM_INFERENCE=0` so the web server can pass health checks immediately. MediaPipe and the sign models load lazily the first time a camera inference endpoint receives a frame.

User accounts, custom emergency phrases, session stats, and temporary uploads use `BRIDGESIGN_DATA_DIR`. The Docker/Render default is `/tmp/bridgesign-data`, which is writable but ephemeral. For persistent accounts, add a Render Disk or Railway Volume and point `BRIDGESIGN_DATA_DIR` at the mount path.

## Why One Worker

The call room signaling state is currently stored in memory in `call_room.py`. Running multiple workers can split two callers across different processes, so they may not find each other. Use one threaded worker until signaling is moved to Redis or another shared store.

## Call Module Note

The current WebRTC config uses public STUN only. It should work for many phone and laptop tests, but some mobile networks or strict NATs may need a TURN server later.

