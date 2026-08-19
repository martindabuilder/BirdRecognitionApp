#File that handles the processing of all the audio files
#before training the model on them

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
import cv2
from tqdm import tqdm
import glob

import warnings
warnings.filterwarnings("ignore")

#folder containing all the audio files that will be processed
data_directory = os.path.join("..", "dataset")

#folders conainting the npy output of the spectrograms
spectrogram_npy_output = "spectrograms_npy"

#creating the (potentially missing) folders
os.makedirs(spectrogram_npy_output, exist_ok = True)

#spectrogram specifications
mel_lines = 150 #spectrogram height, measured in n_mels
sample_rate = 32000 #set sample rate for each spectrogram
audio_duration = 5.0 #seconds per segment

minimal_active_threshold = 0.25 #the amount of "active" volume in the file
max_darkness = 0.05 #intensity below .05 means the segment gets skipped, it's not active enough (i.e its too quiet/empty)

#storage dtype for the saved spectrograms
#a float16 format is chosen as it is a healthy middleground between uint8 and float36
STORAGE_DTYPE = "float16"

#i have chosen to use 4 of the available cores on my computer for the processing
#in case you want to test and run this locally, never go above the amount you actually have
#(i would even suggest your cores - 1 just in case)
NUM_WORKERS = min(6, cpu_count())

#used to precompute the lenght of the spectrogram, removing the need of resizing
segment_samples = int(audio_duration * sample_rate)
hop_length = int(0.020 * sample_rate)


# ----FUNCTIONS ----

#checks if the segment is active enough
#if it isnt above the threshold it gets skipped
def is_seg_active(spectrogram, threshold = 0.05, min_ratio = minimal_active_threshold):
    active_columns = np.any(spectrogram > threshold, axis = 0)
    ratio = np.sum(active_columns) / len(active_columns)

    return ratio >= min_ratio


#checks for the overall darkness of the spectrogram
#if its too dark it gets skipped
def spectrogram_too_dark(spectrogram):
    return spectrogram.mean() < max_darkness


#splits the audio files into numerous 5s segments for better structured learning 
#and transforming into a mel spectrogram
def split_audio(file_path, sr = sample_rate, duration = audio_duration):
    y, sr = librosa.load(file_path,sr = sr,res_type = "soxr_hq")

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


def process_file(file_path):
    try:
        segments, sr = split_audio(file_path)
        
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return []

    file_id = os.path.splitext(os.path.basename(file_path))[0]
    results = []

    for segment_index, seg in enumerate(segments):
        spec = audio_to_spectrogram(seg, sr)

        if not is_seg_active(spec) or spectrogram_too_dark(spec):
            continue

        if STORAGE_DTYPE == "uint8":
            spec = (spec * 255).astype(np.uint8)
        elif STORAGE_DTYPE == "float16":
            spec = spec.astype(np.float16)
        else:
            spec = spec.astype(np.float32)

        results.append((spec, file_id, segment_index))

    return results


def process_class(bird_class):
    out_path = os.path.join(spectrogram_npy_output, f"{bird_class}.npy")
    sources_path = out_path.replace(".npy", "_sources.npy")

    #checks if a class has already been processed, if it has it gets skipped
    if os.path.exists(out_path):
        return bird_class, 0

    class_folder = os.path.join(data_directory, bird_class)
    audio_files = (
        glob.glob(os.path.join(class_folder, "*.wav"))
        + glob.glob(os.path.join(class_folder, "*.WAV"))
        + glob.glob(os.path.join(class_folder, "*.mp3"))
        + glob.glob(os.path.join(class_folder, "*.MP3"))
    )

    class_spectrograms = []
    for f in audio_files:
        class_spectrograms.extend(process_file(f))

    if not class_spectrograms:
        return bird_class, 0

    specs, file_ids, segment_indices = zip(*class_spectrograms)
    arr = np.stack(specs)
    np.save(out_path,arr)
    np.save( sources_path, np.array(file_ids))
    np.save(out_path.replace(".npy", "_segments.npy"),np.array(segment_indices, dtype=np.int32))

    return bird_class, len(class_spectrograms)


def main():
    bird_classes = sorted(
        d for d in os.listdir(data_directory)
        if os.path.isdir(os.path.join(data_directory, d))
    )

    print(f"Processing {len(bird_classes)}")

    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm( pool.imap_unordered(process_class, bird_classes),total = len(bird_classes)))

    total_segments = sum(count for _, count in results)
    
    #label encoding all the classes
    label_encoder = LabelEncoder()
    label_encoder.fit(bird_classes)
    np.save(os.path.join(spectrogram_npy_output, "label_encoder_classes.npy"), label_encoder.classes_)


if __name__ == "__main__":
    main()