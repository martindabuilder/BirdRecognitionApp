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

from data_parser import dataset_build, build_label_encoder, TARGET_SIZE

#directories
data_directory = "spectrograms_npy"

train_set_dir = "train_set"
test_set_dir = "test_set"
val_set_dir = "val_set"

model_directory = "model"
os.makedirs(model_directory, exist_ok = True)

#sets the precision to be float16 globally
#done to reduce memory usage
tf.keras.mixed_precision.set_global_policy("mixed_float16")
BATCH_SIZE = 32

#label encoder is built from the file names in the train set directory
label_encoder = build_label_encoder(train_set_dir)
num_classes = len(label_encoder.classes_)
np.save(os.path.join(model_directory, "label_encoder_classes.npy"), label_encoder.classes_)

inputs = Input(shape = (TARGET_SIZE[0], TARGET_SIZE[1], 3))

base_model = EfficientNetB0(include_top = False, weights = "imagenet", input_tensor = inputs)