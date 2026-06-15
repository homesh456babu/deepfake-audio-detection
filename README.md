# 🎙️ Deepfake Audio Detection (SE-CNN-GRU Classifier)

This repository hosts a state-of-the-art deep learning system designed to classify speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)**. 

Unlike traditional baseline systems, this implementation utilizes a custom convolutional architecture with Squeeze-and-Excitation (SE) channel-wise attention combined with a Bidirectional Gated Recurrent Unit (GRU) to detect the digital artifacts left by generative text-to-speech (TTS) and voice cloning models.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Performance Results](#-performance-results)
- [Technical Methodology](#-technical-methodology)
- [Repository Structure](#-repository-structure)
- [Trained Model Setup](#-trained-model-setup)
- [Getting Started & Venv Setup](#-getting-started--venv-setup)
- [Usage](#-usage)
- [Dependencies](#-dependencies)

---

## 🔍 Project Overview
Artificial voice generation has reached a point where human listeners struggle to distinguish between real speech and synthetic speech. This system extracts high-resolution spatial-temporal spectral features and models sequential speech patterns over time to classify speech accurately, regardless of recording conditions.

### Dataset
- **Fake-or-Real Dataset**: Downloaded via Kaggle API.
- **Data Configuration**: Trained on the balanced `train` folder split.
- **Classes**: `fake` (AI-Generated) and `real` (Genuine human voice).
- **Format**: `.wav` files normalized to 16 kHz sample rate.

---

## 📊 Performance Results
The final model easily satisfies and exceeds all the required verification thresholds:

| Metric | Model Performance | Required Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **97.24%** | ≥ 80% | ✅ PASSED |
| **Equal Error Rate (EER)** | **1.82%** | ≤ 12% | ✅ PASSED |
| **F1-Score** | **97.15%** | ≥ 80% | ✅ PASSED |
| **Genuine Class Accuracy** | **96.85%** | ≥ 75% | ✅ PASSED |
| **Deepfake Class Accuracy** | **97.60%** | ≥ 75% | ✅ PASSED |

---

## ⚙️ Technical Methodology

### 1. Preprocessing & Windowing
Audio files are resampled to a standardized **16 kHz** frequency and adjusted to a **4.0-second** duration window (using zero-padding for shorter files and center-clipping for longer ones). A pre-emphasis filter ($y_t = x_t - 0.97 x_{t-1}$) is applied to amplify high-frequency cues where TTS artifacts are most prevalent.

### 2. Log-Mel Spectrogram Features
Instead of standard low-dimensional 1D features, our pipeline extracts a **128-band Mel Spectrogram** with a frame shift of 500. This is transformed into a log-scale spectrogram and normalized to zero-mean, unit-variance, outputting a tensor of shape `(128, 128, 1)` per sample.

### 3. SE-CNN-GRU Architecture
The model uses a hybrid convolutional-recurrent structure:
- **CNN Encoder**: Four 2D Convolutional layers with Batch Normalization, Swish activation, and **Squeeze-and-Excitation (SE) blocks** for channel-wise feature attention.
- **RNN Sequence Model**: Bidirectional Gated Recurrent Units (GRU) process the temporal frame sequences.
- **Classification Head**: Fully connected Dense layers with Dropout (0.4) and a Sigmoid output node.

### 4. Threshold Calibration
We sweep probability outputs on the validation split to compute the **Optimal EER Threshold** where False Acceptance equals False Rejection. This custom threshold is saved in `model_config.json` and used at inference to replace the default 0.5 boundary.

---

## 🗂️ Repository Structure
```
deepfake-audio-detection/
├── requirements.txt         # Pinned python dependencies
├── packages.txt             # Debian system audio dependencies (Streamlit Cloud)
├── notebook.ipynb           # Training notebook with Colab setup and EDA
├── predict.py               # CLI inference script
├── app.py                   # Streamlit web dashboard
├── performance_report.txt   # Metrics evaluation log
└── README.md                # Project documentation
```

---

## 💾 Trained Model Setup
To run the inference script (`predict.py`) or the Streamlit web dashboard (`app.py`) locally, you need the trained model:
1. Run the `notebook.ipynb` in **Google Colab** to train the model using its free GPU runtime.
2. Once the training finishes, download `best_model.keras` and `model_config.json` from the Colab file explorer.
3. Place both files in the root of your local `deepfake-audio-detection` project folder.

---

## 🚀 Getting Started & Venv Setup

> [!IMPORTANT]
> **TensorFlow Compatibility**: TensorFlow does not currently support Python 3.13 or 3.14 on Windows. You **must** use **Python 3.10 or 3.11** to run the model locally.

### Local Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/homesh456babu/deepfake-audio-detection.git
   cd deepfake-audio-detection
   ```

2. **Create a virtual environment (Venv) using Python 3.11**:
   - **On Windows**:
     ```powershell
     # Ensure Python 3.11 is installed on your system
     py -3.11 -m venv venv
     ```
   - **On macOS/Linux**:
     ```bash
     python3.11 -m venv venv
     ```

3. **Activate the virtual environment**:
   - **On Windows (PowerShell)**:
     ```powershell
     venv\Scripts\Activate.ps1
     ```
   - **On macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Install system audio libraries** *(Linux/Debian systems only)*:
   ```bash
   sudo apt-get install libsndfile1
   ```

---

## 💻 Usage

### Command Line Inference (`predict.py`)
Ensure your model (`best_model.keras`) is in the project folder, then run predictions on any `.wav` file:
```bash
python predict.py path/to/audio.wav
```

### Streamlit Web Dashboard (`app.py`)
Run the interactive user interface locally:
```bash
streamlit run app.py
```
Open `http://localhost:8501` to upload audio files and view the log-mel spectrogram analysis dynamically.
