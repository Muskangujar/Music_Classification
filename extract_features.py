import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set base directory
DATASET_PATH = "Marathi_Emotion_Songs"

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

    return np.hstack([mfcc, chroma, contrast, tonnetz])

# Output list
features = []
labels = []

# Traverse emotion folders
for emotion_folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, emotion_folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"Processing: {emotion_folder}")
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith('.wav') or file.endswith('.mp3'):
            file_path = os.path.join(folder_path, file)
            try:
                feat = extract_features(file_path)
                features.append(feat)
                labels.append(emotion_folder)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Convert to DataFrame
df = pd.DataFrame(features)
df['label'] = labels

# Save to CSV
df.to_csv("emotion_features.csv", index=False)
print("✅ Feature extraction complete! Saved to emotion_features.csv.")
