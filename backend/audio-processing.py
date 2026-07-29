#File that handles the processing of all the audio files
#before training the model on them

import os
import librosa
import numpy as np
from matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEnconder

warnings.filterwarnings("ignore")

#folders conainting the npy output of the spectrograms
spectogram_npy_out = "spectrograms_npy"

#creating (if missing) folders
os.makedirs(spectogram_npy_out, exist__ok = True)

#spectrogram specifications
mel_lines = 150 #spectrogram height, measured in n_mels
spectro_width = 128 #spectrogram width
sample_rate = 16000 
audio_duration = 5.0 #seconds per segment
minimal_active_threshold = 0.5 #the amount of "active" volume in a single column
max_darkness = 0.10 #mean intensity below 0.10 means the segment gets skipped as it is not active enough

#storage dtype for the saved spectrograms
#a float16 format is chosen as it is a healthy middleground between uint8 and float36


