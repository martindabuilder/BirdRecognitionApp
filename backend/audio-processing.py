#File that handles the processing of all the audio files
#before training the model on them

import os
import librosa
import numpy as np
from matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEnconder

warnings.filterwarnings("ignore")

#folders conainting the npy output of the spectrograms and the images themself
spectro_img_out = "spectrogram_images"
spectogram_npy_out = "spectrograms_npy"

#creating (if missing) folders
os.makedirs(spectro_img_out, exist__ok = True)
os.makedirs(spectogram_npy_out, exist__ok = True)

#spectrogram specifications
mel_lines = 150
spectro_width = 128
sample_rate = 16000
audio_duration = 5.0
minimal_active_threshold = 0.5
max_darkness = 0.10

x = [] #spectrograms
y = [] #all seperate bird classes