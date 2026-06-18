"""
BridgeSign – Classifier
Loads the trained scikit-learn pipeline (sign_model.pkl) and exposes a
single predict(features) → (label, confidence) interface so the rest of
the application doesn't need to know about the underlying model type.
"""

import os
import json
import pickle
from collections import OrderedDict
import numpy as np
import config


class Classifier:
    def __init__(self, model_path=None):
        self.pipeline   = None
        self.categories = {}   # {0: 'A', 1: 'B', …}
        self.n_features_in_ = None
        self._predict_cache = OrderedDict()
        self._cache_size = int(getattr(config, "CLASSIFIER_CACHE_SIZE", 2048))
        self._cache_decimals = int(getattr(config, "CLASSIFIER_CACHE_DECIMALS", 2))

        if model_path is None:
            preferred_model = os.path.join(config.MODELS_DIR, "sign_model_light.pkl")
            legacy_model = os.path.join(config.MODELS_DIR, "sign_model.pkl")
            model_path = preferred_model if os.path.exists(preferred_model) else legacy_model

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

    def _prepare_features(self, features):
        if features is None or len(features) == 0:
            return None
        if self.n_features_in_ is not None and len(features) != self.n_features_in_:
            print(
                f"[Classifier] Feature length mismatch: got {len(features)}, "
                f"model expects {self.n_features_in_}. Rebuild dataset/model."
            )
            return None
        return np.asarray(features, dtype=np.float32)

    def _cache_key(self, arr):
        rounded = np.round(arr, self._cache_decimals).astype(np.float32, copy=False)
        return rounded.tobytes()

    def _cache_get(self, key):
        if key not in self._predict_cache:
            return None
        value = self._predict_cache.pop(key)
        self._predict_cache[key] = value
        return value

    def _cache_put(self, key, value):
        if self._cache_size <= 0:
            return
        self._predict_cache[key] = value
        while len(self._predict_cache) > self._cache_size:
            self._predict_cache.popitem(last=False)

    def _result_from_proba(self, proba):
        class_id = int(np.argmax(proba))
        confidence = float(proba[class_id])
        label = self.categories.get(class_id, "Unknown")
        return label, confidence

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
        if self.pipeline is None:
            return "No Model (Train First)", 0.0

        try:
            arr = self._prepare_features(features)
            if arr is None:
                return "Error", 0.0

            key = self._cache_key(arr)
            cached = self._cache_get(key)
            if cached is not None:
                return cached

            X = arr.reshape(1, -1)
            proba = self.pipeline.predict_proba(X)[0]
            result = self._result_from_proba(proba)
            self._cache_put(key, result)

            return result

        except Exception as e:
            print(f"[Classifier] Predict error: {e}")
            return "Error", 0.0

    def predict_many(self, feature_rows):
        """Batch predict several feature vectors, using the same cache as predict()."""
        if self.pipeline is None:
            return [("No Model (Train First)", 0.0) for _ in feature_rows]

        missing_rows = []
        missing_keys = []
        results = [("", 0.0) for _ in feature_rows]

        try:
            for index, features in enumerate(feature_rows):
                arr = self._prepare_features(features)
                if arr is None:
                    results[index] = ("Error", 0.0)
                    continue
                key = self._cache_key(arr)
                cached = self._cache_get(key)
                if cached is not None:
                    results[index] = cached
                else:
                    missing_rows.append((index, arr))
                    missing_keys.append(key)

            if missing_rows:
                X = np.vstack([arr for _, arr in missing_rows]).astype(np.float32, copy=False)
                probas = self.pipeline.predict_proba(X)
                for (index, _), key, proba in zip(missing_rows, missing_keys, probas):
                    result = self._result_from_proba(proba)
                    self._cache_put(key, result)
                    results[index] = result

            return results
        except Exception as e:
            print(f"[Classifier] Batch predict error: {e}")
            return [("Error", 0.0) for _ in feature_rows]

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
