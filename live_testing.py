import os
import time
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import paho.mqtt.client as mqtt
import tensorflow as tf
from tensorflow.image import resize
from huggingface_hub import from_pretrained_keras

# -------------------------------
# Load Models
# -------------------------------

# Load the cry vs. not cry model from Hugging Face.
# NOTE: This model expects an input shape of (1, 128, 126, 1)
cry_vs_not_cry_model = from_pretrained_keras("ericcbonet/cry-baby")
cry_vs_not_cry_model.summary()  # Optional

# Load your retrained classification model (saved in the new format, e.g. .keras)
classification_model_path = "/home/grp4pi/AIFiles/cry_model/my_cry_classification_model_improved.keras"
classification_model = tf.keras.models.load_model(classification_model_path)
# The classification model was trained on spectrograms with target shape (128,126,1)
classes = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# -------------------------------
# MQTT Setup
# -------------------------------
BROKER = "693754a8789c4419b4d760a2653cd86e.s1.eu.hivemq.cloud"
PORT = 8883
TOPIC = "baby_cry/classification"
USERNAME = "gp4pi"
PASSWORD = "Group4pi"

mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(USERNAME, PASSWORD)

def on_connect(client, userdata, flags, rc):
    print("MQTT: Connected" if rc == 0 else f"MQTT: Failed to connect, rc={rc}")

def on_disconnect(client, userdata, rc):
    print("MQTT: Disconnected")

mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.connect(BROKER, PORT, 60)
mqtt_client.loop_start()

# -------------------------------
# Preprocessing Functions for Cry Detection (Live)
# -------------------------------
def preprocess_for_cry_vs_not_cry(waveform, sample_rate=16000, target_shape=(128,126)):
    """
    Preprocess live audio for the cry vs. not cry model.
    This function:
      1. Resamples the waveform to 16kHz (if needed).
      2. Enforces a fixed duration (2 seconds = 32000 samples).
      3. Computes a mel spectrogram with 128 mel bands.
      4. Trims or pads the time axis to exactly 126 frames.
      5. Expands dimensions so the final shape is (1, 128, 126, 1).
    """
    # Resample if needed.
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    
    # Enforce fixed duration: 2 seconds = 32000 samples
    desired_samples = 32000
    if len(waveform) > desired_samples:
        waveform = waveform[:desired_samples]
    elif len(waveform) < desired_samples:
        waveform = np.pad(waveform, (0, desired_samples - len(waveform)), mode='constant')
    
    # Compute mel spectrogram with 128 mel bands.
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=16000, n_fft=512, n_mels=target_shape[0])
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Adjust time dimension to fixed length (126 frames).
    current_frames = mel_spec_db.shape[1]
    fixed_frames = target_shape[1]
    if current_frames > fixed_frames:
        mel_spec_db = mel_spec_db[:, :fixed_frames]
    elif current_frames < fixed_frames:
        mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, fixed_frames - current_frames)), mode='constant')
    
    # Expand dimensions: from (128,126) to (128,126,1)
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
    # Add batch dimension: final shape (1, 128,126,1)
    mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
    
    # Debug: print the shape
    print("Cry-detection input shape:", mel_spec_db.shape)
    return mel_spec_db

def is_crying(waveform, sample_rate=16000):
    input_data = preprocess_for_cry_vs_not_cry(waveform, sample_rate, target_shape=(128,126))
    predictions = cry_vs_not_cry_model.predict(input_data)
    pred_idx = np.argmax(predictions, axis=1)[0]
    return (pred_idx == 1)

# -------------------------------
# File-Based Classification Functions
# -------------------------------
def preprocess_audio_file(file_path, target_shape=(128,126)):
    # Load the audio file.
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Compute mel spectrogram.
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_fft=512, n_mels=target_shape[0])
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Expand dimensions to add a channel: shape (128, t, 1).
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
    # Use TensorFlow to resize the spectrogram to the target shape.
    mel_spec_resized = tf.image.resize(mel_spec_db, target_shape, method='bilinear', antialias=True).numpy()
    # Add a batch dimension: final shape (1, 128,126,1)
    processed = np.expand_dims(mel_spec_resized, axis=0)
    print("Classification input shape from file:", processed.shape)
    return processed

def classify_wav(file_path):
    processed_data = preprocess_audio_file(file_path, target_shape=(128,126))
    predictions = classification_model.predict(processed_data)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    return classes[predicted_class_idx]

# -------------------------------
# Utility: Enforce Fixed Duration on Audio
# -------------------------------
def enforce_fixed_duration(waveform, sample_rate, desired_samples):
    if len(waveform) > desired_samples:
        return waveform[:desired_samples]
    elif len(waveform) < desired_samples:
        return np.pad(waveform, (0, desired_samples - len(waveform)), mode='constant')
    else:
        return waveform

# -------------------------------
# Audio Callback: Live Processing
# -------------------------------
def audio_callback(indata, frames, time_info, status):
    waveform = indata.flatten()
    
    if not np.any(waveform):
        print("No audio data; skipping")
        return

    if not is_significant_audio(waveform):
       # print("Audio too quiet; skipping")
        return

    if not is_crying(waveform, sample_rate=41000):
        print("Not crying")
        return

    print("Cry detected: saving audio for classification...")

    # Force a fixed duration for classification; here, 2 seconds at 41000 Hz.
    desired_duration_secs = 2
    desired_samples = 41000 * desired_duration_secs
    waveform_fixed = enforce_fixed_duration(waveform, 41000, desired_samples)
    temp_filename = "temp_audio.wav"
    sf.write(temp_filename, waveform_fixed, 41000)
    # Short delay to ensure file writing completes.
    time.sleep(0.2)
    try:
        cry_label = classify_wav(temp_filename)
        print(f"Detected cry type: {cry_label}")
        send_notification(cry_label)
    except Exception as e:
        print("Error during classification:", e)

def is_significant_audio(waveform, threshold=0.03):
    rms = np.sqrt(np.mean(waveform**2))
    return rms > threshold

def send_notification(cry_label):
    message = f"The baby might have {cry_label}" if cry_label in ["belly_pain", "discomfort"] else f"The baby might be {cry_label}"
    print("Notification:", message)
    mqtt_client.publish(TOPIC, message)

# -------------------------------
# Main Loop: Start Live Audio Processing
# -------------------------------
if __name__ == "__main__":
    print("Listening for live crying... (Press Ctrl+C to exit)")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=41000):
        try:
            while True:
                sd.sleep(5000)
        except KeyboardInterrupt:
            print("Stopping audio stream...")
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
