# BridgeSign Deployment

## Phone And Call Testing Requirements

Camera and microphone access require HTTPS on phones. A plain LAN URL such as `http://192.168.x.x:5000` will usually block media access.

The call module also uses WebSockets for signaling, so deploy as a dynamic web service, not a static site.

## Render Web Service

1. Push this project to a GitHub repository.
2. In Render, create a new Web Service from that repository.
3. Use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --worker-class gthread --workers 1 --threads 8 --bind 0.0.0.0:$PORT --timeout 120`
4. Set `SECRET_KEY` to a random value if Render does not generate it from `render.yaml`.
5. Open the deployed `https://...onrender.com` URL on your phone.

## Why One Worker

The call room signaling state is currently stored in memory in `call_room.py`. Running multiple workers can split two callers across different processes, so they may not find each other. Use one threaded worker until signaling is moved to Redis or another shared store.

## Call Module Note

The current WebRTC config uses public STUN only. It should work for many phone and laptop tests, but some mobile networks or strict NATs may need a TURN server later.
