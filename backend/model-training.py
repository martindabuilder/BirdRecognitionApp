#File that focuses on training the model before pushing it to the front end

import os
import glob
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

import warnings
warnings.filterwarnings("ignore")

#directories
data_directory = "spectrograms_npy"
model_directory = "model"
os.makedirs(model_directory, exist_ok = True)

#sets the precision to be float16 globally
#done to reduce memory usage
tf.keras.mixed_precision.set_global_policy("mixed_float16")
