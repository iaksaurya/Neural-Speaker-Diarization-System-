# Neural Speaker Diarization
### Speaker Change Detection using BiLSTM
This project implements a speaker change detection (SCD) pipeline that takes long conversational audio, runs VAD to find speech regions, and trains a BiLSTM model on MFCC-based features to detect speaker change points at frame level.​

## Project Structure
### Data Preparation
Using audiacity label speaker in audio.

### Preprocessing​
Audio pre‑processing (stereo → mono, 16 kHz) using pydub.AudioSegment.​
VAD utilities: read_wave, Frame, frame_generator, vad_collector, vad, fxn to output speech intervals as a NumPy array of [start, end] seconds.​
Voice Activity Detection (VAD) using WebRTC VAD to obtain speech-only time intervals.​
### Features Extraction
MFCC + delta + delta‑delta feature extraction using librosa (35‑dimensional feature vectors).
Uses MFCC, delta, delta‑delta, normalization, and sliding windows to form training sequences of shape (N, 137, 35).​
### Bi-LSTM Model
BiLSTM‑based sequence model with TimeDistributed layers for frame‑wise binary change‑point prediction.​
Sliding‑window segmentation (TIMESTEPS = 137 frames, step ≈ 0.8 s) for both training and inference.​
End‑to‑end training and inference script for a given .wav file.​

Training:

create_training_data(training_files_info) builds (X_train, y_train) from audio paths and change times.​

BiLSTM model defined in define_model() and trained via train_model(...), saving weights to my_trained_model.h5.
Model
Input shape: (TIMESTEPS=137, FEATURES=35).​

Architecture:

2 × Bidirectional LSTM layers (hidden size 128 per direction) with dropout.​

Several TimeDistributed Dense layers, final TimeDistributed Dense with 1 unit and sigmoid for binary prediction.​

Trained with binary_crossentropy loss and accuracy metric.​

​

### Inference:

multi_segmentation(file, trained_model) extracts features with the same configuration and runs model.predict over windows.​

Frame‑level scores are aggregated over overlapping windows to produce a change‑likelihood sequence.​


training_files_info = {audio_path: list(end_times)}.​


Call:

X_train, y_train = create_training_data(training_files_info)

model = define_model()

train_model(model, X_train, y_train, SAVED_MODEL_PATH) where SAVED_MODEL_PATH = 'my_trained_model.h5'.​

Example shapes after preparation:

X_train: (1278, 137, 35)

y_train: (1278, 137, 1).​


### Requirements
numpy==1.24.4
pandas==2.0.3
librosa==0.10.1
matplotlib==3.7.3
soundfile==0.12.1
pydub==0.25.1
webrtcvad==2.0.10
scikit-learn==1.3.2
tensorflow==2.13.0
torch==2.1.0
pyannote.core==5.0.0
pyannote.audio==2.1.1


# Thanks


