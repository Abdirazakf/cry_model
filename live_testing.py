#!/usr/bin/env python3
import sys
import queue
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import paho.mqtt.client as mqtt

MODEL_PATH      = "/home/grp4pi/AIFiles/cry_model/my_cry_classification_model_improved.keras"
CLASSES         = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired', 'no_cry']
TARGET_SHAPE    = (128, 128)
CHUNK_DURATION  = 2.0             # seconds per chunk
RMS_THRESHOLD   = 0.02            # filter out very quiet
CONF_THRESHOLD  = 0.85            # require ≥70% confidence to report a real cry

MQTT_HOST       = "693754a8789c4419b4d760a2653cd86e.s1.eu.hivemq.cloud"
MQTT_PORT       = 8883
MQTT_TOPIC      = "baby_cry/classification"
MQTT_USER       = "gp4pi"
MQTT_PASS       = "Group4pi"

print("Loading classification model…")
model = tf.keras.models.load_model(MODEL_PATH)

mqtt_client = mqtt.Client()
mqtt_client.tls_set()
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
mqtt_client.loop_start()

# ——— Prepare audio queue & callback ———
q = queue.Queue()
def audio_callback(indata, frames, time, status):
    if status:
        print(f"[Audio status] {status}", file=sys.stderr)
    q.put(indata.copy())

def is_significant(audio, thresh=RMS_THRESHOLD):
    return np.sqrt(np.mean(audio**2)) > thresh

def get_input_shape():
    dev = sd.query_devices(None, 'input')
    return int(dev['default_samplerate'])

def preprocess(chunk, sr_in, target_shape=TARGET_SHAPE):
    """
    1) Flatten
    2) Resample to 16kHz
    3) Compute 128-band mel spectrogram
    4) Convert to dB, expand to (H, W, 1), resize to target_shape
    5) Add batch dim → (1, H, W, 1)
    """
    audio = chunk.flatten()
    if sr_in != 16000:
        audio = librosa.resample(audio, orig_sr=sr_in, target_sr=16000)
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=512, n_mels=target_shape[0])
    mel_db = librosa.power_to_db(mel, ref=np.max)
    t = tf.convert_to_tensor(mel_db[..., None], dtype=tf.float32)
    t = tf.image.resize(t, target_shape, method="bilinear", antialias=True)
    return tf.expand_dims(t, 0)  # (1, H, W, 1)

def main():
    sr_in = get_input_shape()
    print(f"Mic default samplerate: {sr_in} Hz")
    blocksize = int(sr_in * CHUNK_DURATION)
    print(f"Listening in {CHUNK_DURATION}s chunks…")

    try:
        with sd.InputStream(channels=1, samplerate=sr_in,
                            blocksize=blocksize,
                            callback=audio_callback):
            while True:
                chunk = q.get()
                if not is_significant(chunk):
                    # too-quiet => no_cry
                    print("[no_cry] (quiet)")
                    continue

                x = preprocess(chunk, sr_in)
                preds = model.predict(x)[0]
                idx   = int(np.argmax(preds))
                label = CLASSES[idx]
                conf  = preds[idx]

                if label == "no_cry" or conf < CONF_THRESHOLD:
                    print("[no_cry]")
                    mqtt_client.publish(MQTT_TOPIC, label)
                else:
                    print(f"[{label:10s}] {conf*100:5.1f}%")
                    mqtt_client.publish(MQTT_TOPIC, label)

    except KeyboardInterrupt:
        print("\nStopping live classification.")
        sys.exit(0)
    finally:
       mqtt_client.loop_stop()
       mqtt_client.disconnect()
       sys.exit(0)

if __name__ == "__main__":
    main()
