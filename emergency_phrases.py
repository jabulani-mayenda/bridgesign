class EmergencyPhrases:
    def __init__(self):
        self.phrases = {
            1: "I need help.",
            2: "Call an ambulance.",
            3: "Where is the nearest hospital?",
            4: "I cannot hear. Please communicate via this app or text.",
            5: "I am deaf or hard of hearing."
        }
        self.tts = None

    def get_phrases(self):
        return self.phrases

    def _get_tts(self):
        if self.tts is None:
            from text_to_speech import TextToSpeech
            self.tts = TextToSpeech()
        return self.tts

    def play_phrase(self, phrase_id):
        # Handle string or int IDs
        try:
            phrase_id = int(phrase_id)
        except ValueError:
            pass # might be custom custom ID like "c_12345"
            
        phrase = self.phrases.get(phrase_id)
        if phrase:
            self._get_tts().speak_async(phrase)
            return True
        return False
