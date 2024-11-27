import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.image import resize

# Load the saved model
model = load_model('C:/Users/abdirazak/Downloads/my_cry_classification_model_improved.h5')

# Define the class labels in the same order as used during training
classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

# Function to preprocess individual audio files (convert to Mel spectrogram and resize)
def preprocess_audio(file_path, target_shape=(128, 128)):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    return np.array([mel_spectrogram])  # Return as batch (1 sample)

# Function to classify an individual .wav file
def classify_wav(file_path):
    # Preprocess the file
    processed_data = preprocess_audio(file_path)
    
    # Predict using the trained model
    predictions = model.predict(processed_data)
    
    # Get the predicted class index
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    
    # Map the predicted index to the corresponding class label
    predicted_class_label = classes[predicted_class_idx]
    
    return predicted_class_label

# Example of usage: classify an individual .wav file
file_to_classify = 'C:/Users/abdirazak/Downloads/donateacry_corpus/tired/1309B82C-F146-46F0-A723-45345AFA6EA8-1430059864-1.0-f-04-ti.wav'  # Replace with the path to the .wav file
predicted_category = classify_wav(file_to_classify)
print(f'The predicted category for {os.path.basename(file_to_classify)} is: {predicted_category}')
