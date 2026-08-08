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

IMAGINET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGINET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class SpectrogramDataset(Dataset):

def build_label_encoder():