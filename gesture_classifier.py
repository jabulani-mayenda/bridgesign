"""
BridgeSign -- Gesture Classifier (LSTM + sklearn fallback)
============================================================
Loads the trained gesture model and exposes a
predict(gesture_features) -> (label, confidence) interface.

Supports two model backends:
  1. LSTM (gesture_lstm.pt)  -- preferred, temporal-aware
  2. sklearn (gesture_model.pkl) -- fallback for backward compat

This runs ALONGSIDE the existing static Classifier -- it does NOT
replace it. The app.py pipeline decides which result to use.
"""

import os
import json
import pickle
import numpy as np
import config

# Gesture window settings
from gesture_feature_extractor import GESTURE_WINDOW, STATIC_FEATURES

LSTM_MODEL_PATH   = os.path.join(config.MODELS_DIR, "gesture_lstm.pt")
LSTM_META_PATH    = os.path.join(config.MODELS_DIR, "gesture_lstm_meta.json")
ONNX_MODEL_PATH   = os.path.join(config.MODELS_DIR, "gesture_lstm.onnx")
SKLEARN_MODEL_PATH = os.path.join(config.MODELS_DIR, "gesture_model.pkl")


class GestureClassifier:
    def __init__(self, model_path=None):
        self.backend    = None   # "onnx", "lstm", or "sklearn"
        self.model      = None
        self.pipeline   = None   # sklearn only
        self.ort_session = None  # onnx only
        self.categories = {}     # {0: 'J', 1: 'Z', ...}
        self.n_features_in_ = None
        self._device    = "cpu"

        # Try ONNX first, then LSTM, then fall back to sklearn
        if os.path.exists(ONNX_MODEL_PATH):
            self._load_onnx(ONNX_MODEL_PATH)
        elif os.path.exists(LSTM_MODEL_PATH):
            self._load_lstm(LSTM_MODEL_PATH)
        elif model_path and os.path.exists(model_path):
            self._load_sklearn(model_path)
        elif os.path.exists(SKLEARN_MODEL_PATH):
            self._load_sklearn(SKLEARN_MODEL_PATH)
        else:
            print("[GestureClassifier] No gesture model found -- gesture recognition disabled.")
            print("[GestureClassifier] Run lstm_gesture_trainer.py or gesture_trainer.py.")

    def _load_lstm(self, path):
        """Load the PyTorch LSTM model."""
        try:
            import torch
            from lstm_gesture_model import GestureLSTM

            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            classes = checkpoint["classes"]
            self.categories = {i: c for i, c in enumerate(classes)}

            self.model = GestureLSTM(
                input_size=checkpoint.get("input_size", 42),
                hidden_size=checkpoint.get("hidden_size", 64),
                num_layers=checkpoint.get("num_layers", 2),
                num_classes=len(classes),
                dropout=0.0,  # no dropout at inference
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.backend = "lstm"

            print(f"[GestureClassifier] LSTM model loaded from {path}")
            print(f"[GestureClassifier] {len(classes)} gestures: {classes}")

        except Exception as e:
            print(f"[GestureClassifier] Error loading LSTM model: {e}")
            # Try sklearn fallback
            if os.path.exists(SKLEARN_MODEL_PATH):
                print("[GestureClassifier] Falling back to sklearn model...")
                self._load_sklearn(SKLEARN_MODEL_PATH)

    def _load_sklearn(self, path):
        """Load the legacy scikit-learn gesture model."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.pipeline = data["pipeline"]
            classes_list = data["classes"]
            self.categories = {i: c for i, c in enumerate(classes_list)}
            self.n_features_in_ = getattr(self.pipeline, "n_features_in_", None)
            self.backend = "sklearn"

            print(f"[GestureClassifier] sklearn model loaded from {path}")
            print(f"[GestureClassifier] {len(self.categories)} gestures: {classes_list}")

        except Exception as e:
            print(f"[GestureClassifier] Error loading sklearn model: {e}")
            self.pipeline = None

    def _load_onnx(self, path):
        try:
            import onnxruntime as ort
            self.ort_session = ort.InferenceSession(path)
            
            # Load metadata for classes
            with open(LSTM_META_PATH, "r") as f:
                meta = json.load(f)
            classes = meta["classes"]
            self.categories = {i: c for i, c in enumerate(classes)}
            
            self.backend = "onnx"
            print(f"[GestureClassifier] ONNX model loaded from {path}")
            print(f"[GestureClassifier] {len(classes)} gestures: {classes}")
        except Exception as e:
            print(f"[GestureClassifier] Error loading ONNX model: {e}")
            if os.path.exists(LSTM_MODEL_PATH):
                print("[GestureClassifier] Falling back to PyTorch LSTM model...")
                self._load_lstm(LSTM_MODEL_PATH)
            elif os.path.exists(SKLEARN_MODEL_PATH):
                print("[GestureClassifier] Falling back to sklearn model...")
                self._load_sklearn(SKLEARN_MODEL_PATH)

    def is_available(self):
        """True if a gesture model is loaded and ready."""
        if self.backend == "onnx":
            return self.ort_session is not None
        if self.backend == "lstm":
            return self.model is not None
        return self.pipeline is not None

    def predict(self, gesture_features):
        """
        Parameters
        ----------
        gesture_features : np.ndarray
            For LSTM: shape (15, 42) -- raw frame sequence from the buffer
            For sklearn: shape (N,) -- flattened temporal features from
                         GestureFeatureExtractor.extract_gesture_features()

        Returns
        -------
        label      : str   -- predicted gesture label (e.g., 'J', 'Z', 'STATIC')
        confidence : float -- probability in [0, 1]
        """
        if gesture_features is None or len(gesture_features) == 0:
            return "", 0.0

        if self.backend == "onnx":
            return self._predict_onnx(gesture_features)
        elif self.backend == "lstm":
            return self._predict_lstm(gesture_features)
        elif self.backend == "sklearn":
            return self._predict_sklearn(gesture_features)
        else:
            return "", 0.0

    def _predict_lstm(self, features):
        """Run inference through the LSTM model."""
        try:
            import torch

            # Accept either (15, 42) or flat (630,) input
            features = np.array(features, dtype=np.float32)
            if features.ndim == 1:
                # Flat input from extract_gesture_features -- take only the raw portion
                raw_len = GESTURE_WINDOW * STATIC_FEATURES  # 630
                if len(features) >= raw_len:
                    features = features[:raw_len]
                features = features.reshape(GESTURE_WINDOW, STATIC_FEATURES)
            elif features.ndim == 2 and features.shape == (GESTURE_WINDOW, STATIC_FEATURES):
                pass  # already correct shape
            else:
                return "", 0.0

            X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, 15, 42)

            with torch.no_grad():
                logits = self.model(X)
                probs = torch.softmax(logits, dim=1)[0]
                class_id = int(probs.argmax())
                confidence = float(probs[class_id])
                label = self.categories.get(class_id, "Unknown")

            return label, confidence

        except Exception as e:
            print(f"[GestureClassifier] LSTM predict error: {e}")
            return "", 0.0

    def _predict_sklearn(self, gesture_features):
        """Run inference through the sklearn pipeline (legacy)."""
        try:
            X = np.array([gesture_features], dtype=np.float32)
            proba = self.pipeline.predict_proba(X)[0]
            class_id = int(np.argmax(proba))
            confidence = float(proba[class_id])
            label = self.categories.get(class_id, "Unknown")
            return label, confidence
        except Exception as e:
            print(f"[GestureClassifier] sklearn predict error: {e}")
            return "", 0.0

    def _predict_onnx(self, features):
        try:
            features = np.array(features, dtype=np.float32)
            if features.ndim == 1:
                raw_len = GESTURE_WINDOW * STATIC_FEATURES
                if len(features) >= raw_len:
                    features = features[:raw_len]
                features = features.reshape(GESTURE_WINDOW, STATIC_FEATURES)
            elif features.ndim == 2 and features.shape == (GESTURE_WINDOW, STATIC_FEATURES):
                pass
            else:
                return "", 0.0

            # ONNX expects (batch_size, seq_len, input_size)
            X = np.expand_dims(features, axis=0)
            input_name = self.ort_session.get_inputs()[0].name

            try:
                logits = self.ort_session.run(None, {input_name: X})[0][0]
            except Exception:
                # Older exported models may have a fixed sequence length baked in.
                # Retry once with padding/trimming so the live app keeps working
                # even before the model is re-exported.
                input_shape = self.ort_session.get_inputs()[0].shape
                expected_seq_len = input_shape[1] if len(input_shape) > 1 else None
                if not isinstance(expected_seq_len, int):
                    raise
                if features.shape[0] < expected_seq_len:
                    pad_source = features[-1:] if len(features) else np.zeros((1, STATIC_FEATURES), dtype=np.float32)
                    pad = np.repeat(pad_source, expected_seq_len - features.shape[0], axis=0)
                    features = np.concatenate([features, pad], axis=0)
                elif features.shape[0] > expected_seq_len:
                    features = features[-expected_seq_len:]
                X = np.expand_dims(features, axis=0)
                logits = self.ort_session.run(None, {input_name: X})[0][0]
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            label = self.categories.get(class_id, "Unknown")
            
            return label, confidence
        except Exception as e:
            print(f"[GestureClassifier] ONNX predict error: {e}")
            return "", 0.0

    def is_lstm(self):
        """True if the active backend is the LSTM model or ONNX."""
        return self.backend in ["lstm", "onnx"]
