import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
import sys
import os

# Load label classes
label_classes = np.load('label_classes.npy')

# Load trained model
model = load_model('emotion_model.h5')

# Feature extraction function (same as before)
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, contrast, tonnetz])

# Predict function
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    features = features.reshape(1, features.shape[0], 1)
    prediction = model.predict(features)
    predicted_label = label_classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_label, confidence

# Command-line usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_emotion.py <audio_file>")
    else:
        audio_path = sys.argv[1]
        if not os.path.exists(audio_path):
            print("File not found!")
        else:
            label, confidence = predict_emotion(audio_path)
            print(f"🎵 Predicted Emotion: {label} ({confidence*100:.2f}%)")
