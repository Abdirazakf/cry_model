import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.image import resize
import paho.mqtt.client as mqtt

model = load_model('my_cry_classification_model_improved.h5')
classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

BROKER = "693754a8789c4419b4d760a2653cd86e.s1.eu.hivemq.cloud"
PORT = 8883
TOPIC = "baby_cry/classification"
USERNAME = "gp4pi"
PASSWORD = "Group4pi"

mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(USERNAME, PASSWORD)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
       print("Connected")
    else:
       print("Failed to connect")

def on_disconnect(client, userdata, rc):
    print("Disconnected")

mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect

mqtt_client.connect(BROKER,PORT,60)
mqtt_client.loop_start()

def preprocess_audio(indata, sample_rate=16000, target_shape=(128, 128)):
    """Convert audio data to Mel Spectrogram."""
    if len(indata) < 512:
        return None

    mel_spectrogram = librosa.feature.melspectrogram(y=indata, sr=sample_rate, n_fft=512)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension
    return mel_spectrogram

def is_significant_audio(indata, threshold=0.03):
    """Check if audio has enough energy to be considered relevant."""
    rms = np.sqrt(np.mean(indata**2))  # Root Mean Square Energy
    return rms > threshold

def classify_audio(indata):
    """Run classification only if the audio is significant."""
    if not is_significant_audio(indata):
        return "no_cry"

    mel_spectrogram = preprocess_audio(indata)
    if mel_spectrogram is None:
        return "no_cry"

    predictions = model.predict(mel_spectrogram)
    confidence = np.max(predictions)
    predicted_class = classes[np.argmax(predictions)]

    if confidence < 0.6:  # Confidence threshold to reduce false positives
        return "no_cry"

    return predicted_class

def send_notification(class_name):
    """Send MQTT message only for crying categories."""
    if class_name == "no_cry":
        return  # Do nothing if no crying is detected

    message = f"The baby might be {class_name}" if class_name in ['burping', 'hungry', 'tired'] else f"The baby might have {class_name}"
    print(f"Notification: {message}")
    mqtt_client.publish(TOPIC, message)

def audio_callback(indata, frames, time, status):
    indata = indata.flatten()
    class_name = classify_audio(indata)
    print(f"Predicted Class: {class_name}")

    if class_name != "no_cry":
        send_notification(class_name)

with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
    print("Listening for live crying...")
    while True:
        sd.sleep(5000)

mqtt_client.loop_stop()
mqtt_client.disconnect()
