import json
import os
import time
import config

class SessionTracker:
    def __init__(self):
        self.session_file = os.path.join(config.DATA_DIR, "sessions.json")
        self.history = []
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def log_translation(self, label, confidence, source="camera"):
        """
        Log a translated sign to the session history.
        """
        entry = {
            "timestamp": time.time(),
            "label": label,
            "confidence": confidence,
            "source": source
        }
        self.history.append(entry)
        self._save_history()

    def _save_history(self):
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            print(f"Error saving session history: {e}")

    def get_stats(self):
        """
        Returns basic statistics about tracked sessions.
        """
        total_translations = len(self.history)
        if total_translations == 0:
            return {"total": 0, "most_common": "None"}
            
        # Tally translations
        counts = {}
        for item in self.history:
            lbl = item.get("label", "Unknown")
            counts[lbl] = counts.get(lbl, 0) + 1
            
        most_common = max(counts, key=counts.get)
        
        return {
            "total": total_translations,
            "most_common": most_common,
            "counts": counts
        }
