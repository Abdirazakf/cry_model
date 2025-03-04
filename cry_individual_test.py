import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.image import resize

model = load_model('/home/grp4pi/AIFiles/cry_model/my_cry_classification_model_improved.h5')

classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

# Function to convert to Mel spectrogram and resize
def preprocess_audio(file_path, target_shape=(128, 128)):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    return np.array([mel_spectrogram])

# Function to classify an individual .wav file
def classify_wav(file_path):
    processed_data = preprocess_audio(file_path)
    
    predictions = model.predict(processed_data)
    
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    
    predicted_class_label = classes[predicted_class_idx]
    
    return predicted_class_label

file_to_classify = '/home/grp4pi/AIFiles/cry_model/donateacry_corpus/tired/1309B82C-F146-46F0-A723-45345AFA6EA8-1430059864-1.0-f-04-ti.wav'  # Replace with the path to the .wav file
predicted_category = classify_wav(file_to_classify)
print(f'The predicted category for {os.path.basename(file_to_classify)} is: {predicted_category}')
