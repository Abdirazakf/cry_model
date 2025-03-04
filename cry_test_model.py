import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.image import resize
from tensorflow.keras.utils import to_categorical

data_dir = '/home/grp4pi/AIFiles/cry_model/donateacry_corpus/'
classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
target_shape = (128, 128)

model = load_model('/home/grp4pi/AIFiles/cry_model/my_cry_classification_model_improved.h5')

def load_and_preprocess_data(data_dir, classes, target_shape=(128, 128)):
    data = []
    labels = []
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                data.append(mel_spectrogram)
                labels.append(i)
    
    return np.array(data), np.array(labels)

data, labels = load_and_preprocess_data(data_dir, classes)
labels = to_categorical(labels, num_classes=len(classes))

print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(data, labels)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

predictions = model.predict(data)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(labels, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
