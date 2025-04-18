#!/usr/bin/env python3
import sys
import os

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# ——— Configuration ———
CLASSIFIER_PATH = "/home/grp4pi/AIFiles/cry_model/my_cry_classification_model_improved.keras"
CLASSES         = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired', 'no_cry']
TARGET_SHAPE    = (128, 128)   # height, width

# ——— Preprocessing ———
def preprocess_audio(file_path, target_shape=TARGET_SHAPE):
    """
    1) Load audio at its native sampling rate
    2) Compute a 128‑band mel spectrogram
    3) Expand dims for channel, then resize to target_shape
    4) Add batch dimension → (1, H, W, 1)
    """
    y, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
    mel_spec = np.expand_dims(mel_spec, axis=-1)                      # (128, t, 1)
    mel_tensor = tf.convert_to_tensor(mel_spec, dtype=tf.float32)
    mel_resized = tf.image.resize(mel_tensor, target_shape)           # (128,128,1)
    return tf.expand_dims(mel_resized, axis=0)                        # (1,128,128,1)

# ——— Main Test Loop ———
def main(wav_paths):
    # Load your 6‑way classification model
    model = load_model(CLASSIFIER_PATH)

    for path in wav_paths:
        if not os.path.isfile(path):
            print(f"[ERROR] File not found: {path}")
            continue

        # Preprocess and predict
        x = preprocess_audio(path)
        preds = model.predict(x)[0]           # shape (6,)
        idx   = int(np.argmax(preds))
        probs = ", ".join(f"{p:.2f}" for p in preds)

        # Output
        print(f"\n{os.path.basename(path)}")
        print(f"  Probs: [{probs}]")
        print(f"  → Predicted: {CLASSES[idx]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cry_classify_only_test.py file1.wav [file2.wav ...]")
        sys.exit(1)
    main(sys.argv[1:])
