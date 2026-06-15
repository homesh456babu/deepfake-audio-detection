"""
predict.py - Deepfake Audio Detection Inference
Usage: python predict.py path/to/audio.wav [model_path]
"""
import sys, json, os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

SR, DURATION, N_FFT, HOP_LENGTH = 16000, 4.0, 512, 500
TARGET_FRAMES, N_MELS = 128, 128

def extract_features(file_path):
    n = int(SR * DURATION)
    try:
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    except Exception:
        audio = np.zeros(n, dtype=np.float32)
    audio = np.pad(audio, (0, max(0, n - len(audio))))[:n]
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)
    
    S = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feat = librosa.power_to_db(S, ref=np.max)
    
    T = feat.shape[1]
    feat = np.pad(feat, ((0,0),(0, max(0, TARGET_FRAMES-T))))[:, :TARGET_FRAMES]
    feat = (feat - feat.mean(1, keepdims=True)) / (feat.std(1, keepdims=True) + 1e-6)
    return feat.astype(np.float32)[np.newaxis, ..., np.newaxis]

def predict_audio(file_path, model_path=None, config_path=None):
    if model_path is None:
        for p in ["best_model.keras", "working/best_model.keras", "/kaggle/working/best_model.keras", "/content/working/best_model.keras"]:
            if os.path.exists(p):
                model_path = p
                break
        if model_path is None:
            model_path = "best_model.keras"
            
    if config_path is None:
        for p in ["model_config.json", "working/model_config.json", "/kaggle/working/model_config.json", "/content/working/model_config.json"]:
            if os.path.exists(p):
                config_path = p
                break
        if config_path is None:
            config_path = "model_config.json"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, 0.0

    model = load_model(model_path)
    thresh = 0.5
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            thresh = cfg.get("threshold", 0.5)
        except Exception:
            pass

    prob   = model.predict(extract_features(file_path), verbose=0)[0][0]
    label  = "Deepfake (AI-Generated)" if prob > thresh else "Genuine (Human)"
    conf   = prob if prob > thresh else (1 - prob)
    
    print("\n" + "━" * 37)
    print(f"  PREDICTION  :  {label}")
    print(f"  CONFIDENCE  :  {conf*100:.1f}%")
    print(f"  Fake prob   :  {prob:.4f}  (threshold: {thresh:.2f})")
    print("━" * 37 + "\n")
    return label, float(conf)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/audio.wav [model_path]")
    else:
        m_path = sys.argv[2] if len(sys.argv) > 2 else None
        predict_audio(sys.argv[1], model_path=m_path)
