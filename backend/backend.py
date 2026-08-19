import os
import torch

model_directory = "model"
model_path = os.path.join(model_directory, "best_model.pth")
label_encoder_path = os.path.join(model_directory, "label_encoder_classes.npy")

#specifications that must match the same used beforehand
#i.e in the training and preprocessing
mel_lines = 150 #spectrogram height, measured in n_mels
sample_rate = 32000 #set sample rate for each spectrogram
audio_duration = 5.0 #seconds per segment

max_darkness = 0.05 #intensity below .05 means the segment gets skipped, it's not active enough (i.e its too quiet/empty)
minimal_active_threshold = 0.25 #the amount of "active" volume in the file

segment_samples = int(audio_duration * sample_rate)
hop_length = int(0.020 * sample_rate)


# ---- FUNCTIONS ----

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


def split_audio(file_path, sr = sample_rate, duration = audio_duration):
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


def process_file(file_path):

def load_model():

def predict():

if __name__ == "__main__":
    main()