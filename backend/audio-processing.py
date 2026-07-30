#File that handles the processing of all the audio files
#before training the model on them

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
import cv2
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

#folders conainting the npy output of the spectrograms
spectogram_npy_output = "spectrograms_npy"

#creating (if missing) folders
os.makedirs(spectogram_npy_output, exist_ok = True)

#spectrogram specifications
mel_lines = 150 #spectrogram height, measured in n_mels
spectrogram_width = 128 #spectrogram width
sample_rate = 16000 #set sample rate for each spectrogram
audio_duration = 5.0 #seconds per segment
minimal_active_threshold = 0.5 #the amount of "active" volume in the file
max_darkness = 0.10 #intensity below 0.10 means the segment gets skipped, it's not active enough (i.e its too quiet/empty)

#storage dtype for the saved spectrograms
#a float16 format is chosen as it is a healthy middleground between uint8 and float36
STORAGE_DTYPE = "float16"

#i have chosen to use 4 of the available cores on my computer for the processing
#in case you want to test and run this locally, never go above the amount you actually have
#(i would even suggest your cores - 1 just in case)
NUM_WORKERS = min(4, cpu_count())

#used to precompute the lenght of the spectrogram, removing the need of resizing
segment_samples = int(audio_duration * sample_rate)
hop_length = segment_samples // spectrogram_width

# ----FUNCTIONS ----

#splits the audio files into numerous 5s segments for better structured learning 
#and transforming into a mel spectrogram
def split_audio(file_path, sr = sample_rate, duration = audio_duration):
    y, sr = librosa.load(file_path, sr = sr, res_type = "soxr_hq")
    y, _ = librosa.effects.trim(y)
    segment_length = int(duration * sr)
    segments = []

    #this segment of the function is used in case a segment is too short
    #the empty time is filled up with 0s, until the segment reaches a 5s length
    for start in range(0, len(y), segment_length):
        end = start + segment_length
        seg = y[start:end]
        if len(seg) < segment_length:
            seg = np.pad(seg, (0, segment_length - len(seg)))
        segments.append(seg)
    return segments, sr

#turns the 5s audio segments into a normalized mel spectrogram
def audio_to_spectrogram(y, sr):
    spectrogram = librosa.feature.melspectrogram( y = y, sr = sr, n_mels = mel_lines, hop_length = hop_length)
    spectrogram_db = librosa.power_to_db(spectrogram, ref = np.max)
    spectrogram_norm = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min() + 1e-8)

    #safety net if the hop_length is off by one or two frames
    if spectrogram_norm.shape[1] != spectrogram_width:
        spectrogram_norm = cv2.resize(spectrogram_norm, (spectrogram_width, mel_lines))

    return spectrogram_norm

#checks if the segment is active enough
#if it isnt above the threshold it gets skipped
def is_seg_active(spectrogram, threshold = 0.05, min_ratio = minimal_active_threshold):
    active_columns = np.any(spectrogram > threshold, axis = 0)
    ratio = np.sum(active_columns) / len(active_columns)

    return ratio >= min_ratio

def spectrogram_too_dark(spectrogram):
    return spectrogram.mean() < max_darkness

def process_file(file_path):
    try:
        segments, sr = split_audio(file_path)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return []

    results = []
    for seg in segments:
        spec = audio_to_spectrogram(spectrogram, sr)

        if not is_seg_active(spectrogram) or spectrogra_too_dark(spectrogram):
            continue

        if STORAGE_DTYPE == "unit8":
            spectrogram = (spectrogram * 255).astype(np.uint8)
        elif STORAGE_DTYPE == "float16":
            spectrogram = spectrogram.astype(np.float16)
        else: 
            spectrogram = spectrogram.astype(np.float32)

        results.append(spectrogram)
    return results = []

def process_class(bird_class):
    pass

def main():
    label_encoder = LabelEncoder()

if __name__ == "__main__":
        main()