"""
BridgeSign – Classifier
Loads the trained scikit-learn pipeline (sign_model.pkl) and exposes a
single predict(features) → (label, confidence) interface so the rest of
the application doesn't need to know about the underlying model type.
"""

import os
import json
import pickle
import numpy as np
import config


class Classifier:
    def __init__(self, model_path=None):
        self.pipeline   = None
        self.categories = {}   # {0: 'A', 1: 'B', …}
        self.n_features_in_ = None

        if model_path is None:
            model_path = os.path.join(config.MODELS_DIR, "sign_model.pkl")

        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("[Classifier] No trained model found – run model_trainer.py first.")

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_model(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.pipeline  = data["pipeline"]
            classes_list   = data["classes"]          # list of label strings
            self.categories = {i: c for i, c in enumerate(classes_list)}
            self.n_features_in_ = getattr(self.pipeline, "n_features_in_", None)
            self._force_single_thread_inference()

            print(f"[Classifier] Model loaded from {path}")
            print(f"[Classifier] {len(self.categories)} classes: {classes_list}")

        except Exception as e:
            print(f"[Classifier] Error loading model: {e}")
            self.pipeline = None

    def _force_single_thread_inference(self):
        """Avoid joblib thread-pool creation failures in locked-down Windows shells."""
        if self.pipeline is None:
            return
        try:
            for _, step in getattr(self.pipeline, "steps", []):
                if hasattr(step, "n_jobs"):
                    step.n_jobs = 1
        except Exception as e:
            print(f"[Classifier] Could not set single-thread inference: {e}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features):
        """
        Parameters
        ----------
        features : list or array of floats (hand-landmark features)

        Returns
        -------
        label      : str   – predicted sign letter / word
        confidence : float – probability in [0, 1]
        """
        if features is None or len(features) == 0:
            return "", 0.0

        if self.pipeline is None:
            return "No Model (Train First)", 0.0

        try:
            if self.n_features_in_ is not None and len(features) != self.n_features_in_:
                print(
                    f"[Classifier] Feature length mismatch: got {len(features)}, "
                    f"model expects {self.n_features_in_}. Rebuild dataset/model."
                )
                return "Error", 0.0

            X = np.array([features], dtype=np.float32)

            # predict_proba available because all estimators use soft-voting
            proba     = self.pipeline.predict_proba(X)[0]
            class_id  = int(np.argmax(proba))
            confidence = float(proba[class_id])
            label      = self.categories.get(class_id, "Unknown")

            return label, confidence

        except Exception as e:
            print(f"[Classifier] Predict error: {e}")
            return "Error", 0.0

    def predict_topk(self, features, k=5):
        """Return the top-k class probabilities for debugging live inference."""
        if features is None or len(features) == 0 or self.pipeline is None:
            return []

        try:
            if self.n_features_in_ is not None and len(features) != self.n_features_in_:
                return []

            X = np.array([features], dtype=np.float32)
            proba = self.pipeline.predict_proba(X)[0]
            top_ids = np.argsort(proba)[::-1][:k]
            return [
                {
                    "label": self.categories.get(int(i), "Unknown"),
                    "confidence": float(proba[int(i)]),
                }
                for i in top_ids
            ]
        except Exception as e:
            print(f"[Classifier] Top-k predict error: {e}")
            return []
