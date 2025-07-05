import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import soundfile as sf

# Load model and labels
model = load_model("emotion_model.h5")
label_classes = np.load("label_classes.npy", allow_pickle=True)

# Emotion meanings
emotion_meanings = {
    "Amused, Relaxed": "Cheerful",
    "Balanced, Serene": "Peaceful",
    "depressed": "Low Mood",
    "Excited, Balanced": "Energetic",
    "Happy, Amused": "Joyful",
    "Relaxed, Serene": "Calm",
    "Sad, Serene": "Melancholy",
    "Stressed, Tense": "Anxious",
    "Tense, Sad": "Distressed"
}

# Feature extraction
def extract_features(audio):
    y, sr = librosa.load(audio, duration=30)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, contrast, tonnetz])

# Rule-based prediction (simple tempo-energy based)
def rule_based_emotion(audio):
    y, sr = librosa.load(audio, duration=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = np.mean(librosa.feature.rms(y=y))

    if tempo > 130 and energy > 0.1:
        return "Excited, Balanced"
    elif tempo < 70 and energy < 0.03:
        return "depressed"
    elif tempo < 90 and energy < 0.06:
        return "Sad, Serene"
    elif 90 <= tempo <= 110:
        return "Relaxed, Serene"
    elif tempo > 110 and energy < 0.08:
        return "Happy, Amused"
    elif energy > 0.12:
        return "Stressed, Tense"
    else:
        return "Balanced, Serene"

# Model prediction
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    features = features.reshape(1, features.shape[0], 1)
    prediction = model.predict(features)
    predicted_label = label_classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_label, confidence, prediction[0]

# UI
st.title("🎵 Music Emotion Classifier")
st.markdown("Upload a **WAV or MP3** file. We'll use both a Machine learning model and audio cues to classify its emotion.")

uploaded_file = st.file_uploader("🎧 Upload your audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Spectrogram
    y, sr = librosa.load("temp_audio.wav", duration=30)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max),
                                   sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='Spectrogram')
    st.pyplot(fig)

    # Model prediction
    predicted_label, confidence, full_probs = predict_emotion("temp_audio.wav")
    model_meaning = emotion_meanings.get(predicted_label, predicted_label)
    st.success(f"🎯 Model Prediction: **{predicted_label}** (*{model_meaning}*) — {confidence*100:.2f}%")

    # Rule-based prediction
    rule_label = rule_based_emotion("temp_audio.wav")
    rule_meaning = emotion_meanings.get(rule_label, rule_label)
    st.info(f"🧠 Additional Prediction: **{rule_label}** (*{rule_meaning}*)")

    # Justification if predictions differ
    if predicted_label != rule_label:
        st.warning("🔄 Combined Interpretation:")
        st.markdown(f"The song shows elements of **both** *{model_meaning}* and *{rule_meaning}*. It may reflect **mixed emotions**, capturing both moods depending on tempo, rhythm, and tone.")

    # Confidence scores
    st.subheader("📊 Confidence Scores:")
    for i, label in enumerate(label_classes):
        meaning = emotion_meanings.get(label, "")
        st.write(f"{label} ({meaning}): {full_probs[i]*100:.2f}%")
