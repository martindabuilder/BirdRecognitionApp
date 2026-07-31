#File that focuses on training the model before pushing it to the front end

import os
import tensorflow as tf
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#directories
data_directory = "spectrograms_npy"
model_directory = "model"
os.makedirs(model_dir, exist_ok = True)