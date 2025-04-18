#!/usr/bin/env python3
import sys
import os
import numpy as np
import librosa
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

# ——— Configuration ———
CLASSIFIER_PATH = "/home/grp4pi/AIFiles/cry_model/my_cry_classification_model_improved.keras"
CRY_CLASSES = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
CRY_THRESHOLD = 0.5              # p_cry above this => “cry”
TARGET_SHAPE = (128, 126)        # height × width for both models

# ——— Load Models ———
print("Loading cry-vs-not-cry model…")
cry_model = from_pretrained_keras("ericcbonet/cry-baby")
print("Loading 5‑way classification model…")
classify_model = tf.keras.models.load_model(CLASSIFIER_PATH)

# ——— Preprocessing ———
def preprocess_mel(wav_path, target_shape=TARGET_SHAPE):
    """Load WAV @16 kHz, make mel spectrogram, convert to dB, resize to target_shape, add batch+channel."""
    y, sr = librosa.load(wav_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, n_mels=target_shape[0])
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # (128, t) → (128, t, 1)
    t = tf.convert_to_tensor(mel_db[..., None], dtype=tf.float32)
    # → (128,126,1)
    t = tf.image.resize(t, target_shape, method="bilinear", antialias=True)
    # → (1,128,126,1)
    return t[None, ...]

# ——— Test Loop ———
def test_files(wav_paths):
    for path in wav_paths:
        if not os.path.isfile(path):
            print(f"[ERROR] File not found: {path}")
            continue

        print(f"\n=== {os.path.basename(path)} ===")
        # 1) cry-vs-not-cry (single sigmoid)
        x = preprocess_mel(path)
        preds = cry_model.predict(x)            # shape = (1,1)
        p_cry = float(preds[0,0])
        p_not = 1.0 - p_cry
        label = "cry" if p_cry > CRY_THRESHOLD else "not_cry"
        print(f" Cry-vs-not-cry → [not_cry={p_not:.3f}, cry={p_cry:.3f}] → {label}")

        # 2) if cry, run 5‑way classifier
        if p_cry > CRY_THRESHOLD:
            y = preprocess_mel(path)
            preds5 = classify_model.predict(y)[0]  # shape = (5,)
            idx5 = int(np.argmax(preds5))
            probs5 = ", ".join(f"{p:.2f}" for p in preds5)
            print(f" 5-way probs: [{probs5}]")
            print(f" Predicted type: {CRY_CLASSES[idx5]}")
        else:
            print(" Skipping 5‑way classification (no cry detected)")

# ——— Entry Point ———
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cry_individual_test.py file1.wav [file2.wav ...]")
        sys.exit(1)
    test_files(sys.argv[1:])
