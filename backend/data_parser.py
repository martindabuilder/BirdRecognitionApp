import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# same values used when creating spectrograms
mel_lines = 150
spectrogram_width = 128
TARGET_SIZE = (160, 160) #target resize for the efficientnet