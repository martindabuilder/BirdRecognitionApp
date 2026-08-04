#File that focuses on training the model before pushing it to the front end


import os
import glob
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

#taking all the necessary functions from data_parser.py
from data_parser import dataset_build, build_label_encoder, TARGET_SIZE

#directories
data_directory = "spectrograms_npy"

train_set_dir = "train_set"
test_set_dir = "test_set"
val_set_dir = "val_set"

model_directory = "model"
os.makedirs(model_directory, exist_ok = True)

#limiting the gpu memory to grow if the process needs it, rather than using all of it instantly
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#sets the precision to be float16 globally
#done to reduce memory usage
tf.keras.mixed_precision.set_global_policy("mixed_float16")
BATCH_SIZE = 64

#label encoder is built from the file names in the train set directory
label_encoder = build_label_encoder(train_set_dir)
num_classes = len(label_encoder.classes_)
np.save(os.path.join(model_directory, "label_encoder_classes.npy"), label_encoder.classes_)

#caches the datasets to avoid reloading them every time the model is trained again
train_dataset = dataset_build(train_set_dir, label_encoder, batch_size = BATCH_SIZE, training = True)
val_dataset = dataset_build(val_set_dir, label_encoder, batch_size = BATCH_SIZE, training = False)
test_dataset = dataset_build(test_set_dir, label_encoder, batch_size = BATCH_SIZE, training = False)

#shape of data is 160x160, with 3 channels (RGB)
inputs = Input(shape = (TARGET_SIZE[0], TARGET_SIZE[1], 3))

#EfficientNetB0 used as a base model for the project
base_model = EfficientNetB0(include_top = False, weights = "imagenet", input_tensor = inputs)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation = "relu")(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation = "softmax", dtype = "float32")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)


#callbacks
callbacks = [
    EarlyStopping(monitor = "val_accuracy", patience = 8, restore_best_weights = True),
    ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 4, min_lr = 1e-6),
    ModelCheckpoint(os.path.join(model_directory, "best_model.keras"), monitor = "val_accuracy", save_best_only = True)
]

#training the model
history = model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = 50,
    callbacks = callbacks,
    verbose = 1
)


test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

model_path = os.path.join(model_directory, "bird_recognition_model_effnet.h5")
model.save(model_path)

print(f"Model saved to {model_path}")