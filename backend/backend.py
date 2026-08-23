import os
import numpy as np
import librosa
import torchvision.models as models

#---- FastApi related imports ----
from fastapi import FastAPI, HTTPException, File

app = FastAPI(title = "Bird Recognition")

#---- Model related paths ----
model_directory = "model"
model_path = os.path.join(model_directory, "best_model.pth")
label_encoder_path = os.path.join(model_directory, "label_encoder_classes.npy")

#settings that must match the same used beforehand during the preprocessing
MEL_LINES = 150 #spectrogram height, measured in n_mels
SAMPLE_RATE = 32000 #set sample rate for each spectrogram
AUDIO_DURATION = 5.0 #seconds per segment

MAX_DARKNESS = 0.05 #intensity below .05 means the segment gets skipped(i.e its too quiet/empty)
MINIMAL_ACTIVE_THRESHOLD = 0.25 #the amount of "active" volume in the file

SEGMENT_SAMPLES = int(AUDIO_DURATION * SAMPLE_RATE)
HOP_LENGTH = int(0.020 * SAMPLE_RATE)


#---- filtering functions ----
#checks if the segment is active enough
#if it isnt above the threshold it gets skipped
def is_seg_active(spectrogram, threshold = 0.05, min_ratio = MINIMAL_ACTIVE_THRESHOLD):
    active_columns = np.any(spectrogram > threshold, axis = 0)
    ratio = np.sum(active_columns) / len(active_columns)

    return ratio >= min_ratio

#checks for the overall darkness of the spectrogram, if its too dark it gets skipped
def spectrogram_too_dark(spectrogram):
    return spectrogram.mean() < max_darkness


#---- Audio processing related functions ----
#preprocessing, which is done in the same way as preprocessing.py
def split_audio(file_path, sr = SAMPLE_RATE, duration = AUDIO_DURATION):
    y, sr = librosa.load(file_path,sr = sr, res_type ="soxr_hq")

    y, _ = librosa.effects.trim(y)
    segment_length = int(duration * sr)
    hop_samples = int(4.0 * sr) #allows a 1 second overlap between 2 consecutive segments
    segments = []

    for start in range(0, len(y), hop_samples):
        end = start + segment_length
        seg = y[start:end]
        if len(seg) < segment_length:
            seg = np.pad(seg,(0, segment_length - len(seg)))
        segments.append(seg)

    return segments, sr

#turns the 5s audio segments into a normalized mel spectrogram
def audio_to_spectrogram(y, sr):
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 2048, hop_length = hop_length, n_mels = mel_lines, fmin = 200, fmax = 16000, power = 2.0)
    spectrogram_db = librosa.power_to_db(spectrogram, ref = np.max, top_db=80)
    spectrogram_norm = (spectrogram_db + 80.0) / 80.0
    spectrogram_norm = np.clip( spectrogram_norm, 0.0, 1.0)
    return spectrogram_norm.astype(np.float32)


#---- Model loading and performance functions

def load_model():
    classes = np.load(label_encoder_path, allow_pickle = True)


def predict_file(file_path):
    segments, sr = split_audio(file_path)
    predictions = []


def main():
    

if __name__ == "__main__":
    main()