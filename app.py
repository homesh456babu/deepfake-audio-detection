import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .card-real {
        background-color: #D1FAE5;
        border-left: 6px solid #10B981;
        padding: 20px;
        border-radius: 8px;
        color: #065F46;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .card-fake {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 20px;
        border-radius: 8px;
        color: #991B1B;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_ok=True)

# App Title
st.markdown("<div class='main-title'>🎙️ Deepfake Audio Detector</div>", unsafe_ok=True)
st.markdown("<div class='subtitle'>Analyze speech recordings using SE-CNN-GRU Deep Learning models</div>", unsafe_ok=True)

# Sidebar Info
with st.sidebar:
    st.header("Pipeline Technical Specs")
    st.markdown("""
    - **Classifier**: SE-CNN + Bidirectional GRU
    - **Attention Block**: Channel Squeeze-and-Excitation
    - **Audio Features**: Log-Mel Spectrogram (128 bands)
    - **Target Sample Rate**: 16,000 Hz
    - **Window Duration**: 4.0 seconds
    """)
    st.divider()
    st.markdown("Designed for robust binary classification of human speech vs generative synthetic voices.")

# Constants
SR, DURATION, N_FFT, HOP_LENGTH = 16000, 4.0, 512, 500
TARGET_FRAMES, N_MELS = 128, 128

@st.cache_resource
def load_audio_model(model_path):
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

def extract_features(y):
    n = int(SR * DURATION)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)
    
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feat = librosa.power_to_db(S, ref=np.max)
    
    T = feat.shape[1]
    feat = np.pad(feat, ((0,0),(0, max(0, TARGET_FRAMES-T))))[:, :TARGET_FRAMES]
    feat = (feat - feat.mean(1, keepdims=True)) / (feat.std(1, keepdims=True) + 1e-6)
    return feat.astype(np.float32)[np.newaxis, ..., np.newaxis]

# Find model paths
model_path = None
config_path = None
for p in ["best_model.keras", "working/best_model.keras", "/kaggle/working/best_model.keras", "/content/working/best_model.keras"]:
    if os.path.exists(p):
        model_path = p
        break
for p in ["model_config.json", "working/model_config.json", "/kaggle/working/model_config.json", "/content/working/model_config.json"]:
    if os.path.exists(p):
        config_path = p
        break

model = load_audio_model(model_path) if model_path else None
threshold = 0.5
if config_path and os.path.exists(config_path):
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        threshold = cfg.get("threshold", 0.5)
    except Exception:
        pass

# File Uploader
uploaded_file = st.file_uploader("Upload an audio recording (.wav format)", type=["wav"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Audio Player")
        st.audio(uploaded_file, format='audio/wav')
        
        # Load and pre-process audio
        with st.spinner("Analyzing audio signals..."):
            y, sr = librosa.load(uploaded_file, sr=SR, duration=DURATION)
            features = extract_features(y)
            
            if model:
                prob = model.predict(features, verbose=0)[0][0]
                is_fake = prob > threshold
                confidence = prob if is_fake else (1.0 - prob)
                
                st.subheader("Detection Result")
                if is_fake:
                    st.markdown(f"<div class='card-fake'>🤖 Prediction: DEEPFAKE (AI-Generated)<br><span style='font-size: 15px; font-weight: normal;'>Confidence: {confidence * 100:.1f}% (Probability: {prob:.4f})</span></div>", unsafe_ok=True)
                else:
                    st.markdown(f"<div class='card-real'>👤 Prediction: GENUINE (Human Speech)<br><span style='font-size: 15px; font-weight: normal;'>Confidence: {confidence * 100:.1f}% (Probability: {prob:.4f})</span></div>", unsafe_ok=True)
            else:
                st.warning("Prediction model not found. Please train and save 'best_model.keras' to run classification.")
                
    with col2:
        st.subheader("Spectral Analysis")
        # Visualizing log-mel spectrogram
        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("Log-Mel Spectrogram")
        st.pyplot(fig)
else:
    st.info("Please upload a .wav audio file to start deepfake analysis.")
