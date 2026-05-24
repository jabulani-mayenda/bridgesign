try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

import threading

class SpeechToText:
    def __init__(self):
        self.is_listening = False
        if not SPEECH_AVAILABLE:
            print("[SpeechToText] PyAudio/SpeechRecognition not available. Feature disabled.")
            self.recognizer = None
            self.microphone = None
            return

        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
            # Adjust for ambient noise on initialization
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
        except (AttributeError, OSError) as e:
            print(f"[SpeechToText] Microphone not available ({e}). Feature disabled.")
            self.recognizer = None
            self.microphone = None

    def listen(self, callback):
        """
        Listen for speech in the background and call the given callback 
        function with the transcribed text.
        """
        if not SPEECH_AVAILABLE or self.is_listening:
            return

        self.is_listening = True

        def _listen_thread():
            while self.is_listening:
                try:
                    with self.microphone as source:
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

                    text = self.recognizer.recognize_google(audio)
                    if callback:
                        callback(text)
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"Speech recognition service error: {e}")
                except Exception as e:
                    print(f"Error in speech recognition: {e}")

        thread = threading.Thread(target=_listen_thread)
        thread.daemon = True
        thread.start()

    def stop(self):
        self.is_listening = False

