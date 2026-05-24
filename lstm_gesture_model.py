"""
BridgeSign -- LSTM Gesture Model
=================================
A lightweight LSTM network that classifies hand gesture sequences.

Input:  (batch, seq_len=15, features=42)  -- 15 frames of 42 landmarks
Output: (batch, num_classes)              -- softmax over gesture classes

The key innovation over the old Random Forest approach:
  - The LSTM *learns* temporal patterns, not just flat feature vectors
  - A dedicated "STATIC" class explicitly tells the system
    "this is NOT a gesture, let the static classifier handle it"
  - No more hand-tuned thresholds for motion magnitude or confidence margins
"""

import torch
import torch.nn as nn


class GestureLSTM(nn.Module):
    """
    Lightweight LSTM for gesture classification.
    ~50K parameters -- runs in <5ms on CPU.
    """

    def __init__(self, input_size=42, hidden_size=64, num_layers=2,
                 num_classes=3, dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Batch-norm on input features for stable training
        self.input_bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Attention layer: learn which frames matter most
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, 42)

        Returns
        -------
        logits : Tensor of shape (batch, num_classes)
        """
        batch_size, seq_len, feat_size = x.shape

        # Apply batch norm per-feature across the sequence
        # Reshape: (batch * seq_len, features) for BN, then back
        x_flat = x.reshape(-1, feat_size)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(batch_size, seq_len, feat_size)

        # LSTM forward
        lstm_out, _ = self.lstm(x)   # (batch, seq_len, hidden)

        # Attention-weighted pooling over time steps
        attn_weights = self.attention(lstm_out)          # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)   # (batch, hidden)

        return self.classifier(context)

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_classes, device="cpu"):
    """Factory function to create and initialise the model."""
    model = GestureLSTM(
        input_size=42,
        hidden_size=64,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    n_params = GestureLSTM.count_parameters(model)
    print(f"[GestureLSTM] Created model with {n_params:,} trainable parameters")
    print(f"[GestureLSTM] Classes: {num_classes}, Device: {device}")

    return model
