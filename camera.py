import cv2
import threading
import queue
import config

class Camera:
    def __init__(self, index=config.CAMERA_INDEX, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT):
        self.index = index
        self.width = width
        self.height = height
        self.cap = None
        self.active_index = None
        self.active_backend = None
        # Background reader thread — always serves the freshest frame
        self._frame_queue = queue.Queue(maxsize=1)
        self._reader_thread = None
        self._reading = False

    def _candidate_indices(self):
        candidates = [self.index]
        for idx in range(config.CAMERA_MAX_INDEX + 1):
            if idx not in candidates:
                candidates.append(idx)
        return candidates

    def _candidate_backends(self):
        backends = []
        for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
            value = getattr(cv2, name, None)
            if value is not None and value not in backends:
                backends.append(value)
        return backends or [0]

    def _open_capture(self, index, backend):
        if backend == getattr(cv2, "CAP_ANY", 0):
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)

        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            return None

        # ── Key perf setting: keep buffer at 1 frame so we always get fresh frames
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None

        return cap

    def start(self):
        backend_names = {
            getattr(cv2, "CAP_DSHOW", -1): "DirectShow",
            getattr(cv2, "CAP_MSMF", -2): "MediaFoundation",
            getattr(cv2, "CAP_ANY", 0): "Auto",
        }

        for index in self._candidate_indices():
            for backend in self._candidate_backends():
                cap = self._open_capture(index, backend)
                if cap is not None:
                    self.cap = cap
                    self.active_index = index
                    self.active_backend = backend_names.get(backend, str(backend))
                    print(f"[Camera] Using index {index} via {self.active_backend}")
                    self._start_reader()
                    return

        raise RuntimeError(
            f"Failed to open camera. Tried indices 0-{config.CAMERA_MAX_INDEX} "
            "with DirectShow, MediaFoundation, and Auto backends."
        )

    def _start_reader(self):
        """Spin up a background thread that continuously reads frames
           and keeps only the latest one (queue maxsize=1 auto-discards stale frames)."""
        self._reading = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self):
        while self._reading and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)   # flip once here, not in get_frame
                # Drop old frame if consumer is slow; keep only the newest
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put((True, frame))
            else:
                self._frame_queue.put((False, None))

    def get_frame(self):
        """Return the freshest frame from the background reader thread."""
        try:
            return self._frame_queue.get(timeout=0.1)
        except queue.Empty:
            return False, None

    def stop(self):
        self._reading = False
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.active_index = None
            self.active_backend = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
