import pyttsx3
import threading
import queue
import config

class TextToSpeech:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TextToSpeech, cls).__new__(cls)
                cls._instance._init_engine()
        return cls._instance

    def _init_engine(self):
        self.q = queue.Queue()
        self.is_speaking = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        # Engine initialized inside the worker thread to be thread-safe on Windows
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', config.TTS_RATE)
            engine.setProperty('volume', config.TTS_VOLUME)
            
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                # Often a female voice at index 1
                engine.setProperty('voice', voices[1].id)
        except Exception as e:
            print(f"[TTS Init Error] {e}")
            return

        while True:
            text = self.q.get()
            if text is None:
                break
            self.is_speaking = True
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS Error] {e}")
            finally:
                self.is_speaking = False
                self.q.task_done()

    @staticmethod
    def _normalize(text: str) -> str:
        """Convert all-caps strings to Title Case so pyttsx3 reads them as
        real words rather than spelling them out letter by letter.
        e.g.  'HELLO'       -> 'Hello'
              'HELLO WORLD' -> 'Hello World'
        Mixed-case strings (emergency phrases etc.) pass through unchanged."""
        if text and text == text.upper():
            return text.title()
        return text

    def speak(self, text):
        """Speak the text synchronously by waiting for the queue."""
        if text:
            self.q.put(self._normalize(text))
            self.q.join()

    def speak_async(self, text):
        """Speak the text asynchronously to avoid blocking."""
        if text:
            self.q.put(self._normalize(text))

