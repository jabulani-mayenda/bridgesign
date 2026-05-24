"""
BridgeSign -- LSTM Gesture Trainer
====================================
Trains the GestureLSTM model on gesture data collected by gesture_collector.py.

Usage:
  python lstm_gesture_trainer.py

Expects gesture data in: data/gesture_dataset.csv
Each row: label, f1, f2, ..., f42, f1, f2, ..., f42, ...  (15 frames x 42 features = 630 values)

Output:
  models/gesture_lstm.pt         -- trained model weights
  models/gesture_lstm_meta.json  -- class labels + training metrics
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config
from gesture_feature_extractor import GESTURE_WINDOW, STATIC_FEATURES
from lstm_gesture_model import create_model

DATA_PATH   = os.path.join(config.DATA_DIR, "gesture_dataset.csv")
MODEL_PATH  = os.path.join(config.MODELS_DIR, "gesture_lstm.pt")
META_PATH   = os.path.join(config.MODELS_DIR, "gesture_lstm_meta.json")

# Training hyperparameters
EPOCHS        = 80
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15   # early stopping


class GestureSequenceDataset(Dataset):
    """PyTorch dataset that loads gesture sequences from the CSV."""

    def __init__(self, sequences, labels):
        """
        Parameters
        ----------
        sequences : np.ndarray of shape (N, 15, 42)
        labels    : np.ndarray of shape (N,)  -- integer class IDs
        """
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_gesture_data(path):
    """
    Load gesture sequences from CSV.
    
    The gesture_collector saves each sample as:
      label, f1, f2, ..., f630  (15 frames * 42 features = 630 values)
    
    We reshape each sample back into (15, 42) for the LSTM.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gesture dataset not found at '{path}'.\n"
            "Run gesture_collector.py first to collect training data."
        )

    labels_raw = []
    sequences  = []
    expected_len = GESTURE_WINDOW * STATIC_FEATURES  # 15 * 42 = 630

    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            label = parts[0]
            features = [float(x) for x in parts[1:]]

            if len(features) != expected_len:
                print(f"  WARNING: Line {line_no}: expected {expected_len} features, "
                      f"got {len(features)} -- skipping")
                continue

            labels_raw.append(label)
            seq = np.array(features, dtype=np.float32).reshape(GESTURE_WINDOW, STATIC_FEATURES)
            sequences.append(seq)

    X = np.array(sequences, dtype=np.float32)    # (N, 15, 42)
    return X, labels_raw


def augment_sequences(X, y, copies=3, noise_std=0.02):
    """Add Gaussian noise and time-warping augmentation."""
    rng = np.random.default_rng(42)
    aug_X, aug_y = [X], [y]

    for i in range(copies):
        intensity = noise_std * (0.7 + 0.6 * i / max(copies - 1, 1))
        noisy = X + rng.normal(0, intensity, X.shape).astype(np.float32)
        aug_X.append(noisy)
        aug_y.append(y)

        # Time-shift augmentation: randomly shift sequence by 1-2 frames
        shift = rng.integers(1, 3)
        shifted = np.roll(X, shift, axis=1)
        shifted[:, :shift, :] = X[:, :shift, :]  # don't wrap, repeat first frames
        shifted += rng.normal(0, noise_std * 0.3, shifted.shape).astype(np.float32)
        aug_X.append(shifted)
        aug_y.append(y)

    return np.vstack(aug_X), np.concatenate(aug_y)


def main():
    print("\n-- BridgeSign LSTM Gesture Trainer --")
    print(f"  Dataset : {DATA_PATH}")
    print(f"  Output  : {MODEL_PATH}\n")

    # 1. Load data
    X, labels_raw = load_gesture_data(DATA_PATH)
    classes = sorted(set(labels_raw))
    label_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_id[l] for l in labels_raw], dtype=np.int64)

    print(f"  Samples     : {len(X)}")
    print(f"  Seq shape   : {X.shape[1:]}  (frames x features)")
    print(f"  Classes ({len(classes)}): {classes}")

    # Class distribution
    for c in classes:
        count = sum(1 for l in labels_raw if l == c)
        print(f"    {c:12s}: {count} samples")

    # 2. Train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train (raw) : {len(X_train)}")
    print(f"  Val (clean) : {len(X_val)}")

    # 3. Augment training set
    X_train, y_train = augment_sequences(X_train, y_train)
    print(f"  Train (aug) : {len(X_train)}\n")

    # 4. Create dataloaders
    train_ds = GestureSequenceDataset(X_train, y_train)
    val_ds   = GestureSequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(num_classes=len(classes), device=device)

    # 6. Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights /= class_weights.sum()
    class_weights *= len(classes)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    # 7. Training loop
    best_val_acc = 0.0
    patience_counter = 0

    print("  Training...")
    for epoch in range(1, EPOCHS + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        # -- Validate --
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={train_loss/train_total:.4f}  "
                  f"train={train_acc:.2%}  val={val_acc:.2%}  lr={lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "input_size": 42,
                "hidden_size": 64,
                "num_layers": 2,
                "num_classes": len(classes),
            }, MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # 8. Final evaluation
    print(f"\n  Best validation accuracy: {best_val_acc:.2%}")

    # Load best model for final report
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Per-class accuracy
    class_correct = {c: 0 for c in classes}
    class_total   = {c: 0 for c in classes}
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(1).cpu()
            for pred, true in zip(preds, y_batch):
                true_class = classes[true.item()]
                class_total[true_class] += 1
                if pred.item() == true.item():
                    class_correct[true_class] += 1

    print("\n  Per-class accuracy:")
    report = {}
    for c in classes:
        acc = class_correct[c] / max(class_total[c], 1)
        print(f"    {c:12s}: {acc:.0%} ({class_correct[c]}/{class_total[c]})")
        report[c] = {"accuracy": acc, "correct": class_correct[c], "total": class_total[c]}

    # Save metadata
    meta = {
        "classes": classes,
        "best_val_accuracy": best_val_acc,
        "per_class": report,
        "epochs_trained": epoch,
        "model_params": GestureLSTM.count_parameters(model) if 'GestureLSTM' in dir() else "N/A",
    }
    # Import for param count
    from lstm_gesture_model import GestureLSTM
    meta["model_params"] = GestureLSTM.count_parameters(model)

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n  Model saved  -> {MODEL_PATH}")
    print(f"  Meta saved   -> {META_PATH}")
    print("\n-- Training complete! --\n")


if __name__ == "__main__":
    main()
