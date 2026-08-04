#As this is a one man project, to make it easier on my laptop im adding a parser class
#Its job will be to read and pass the data to the model training class one at a time, 
#making it lighter on memory usage

import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

#matching the values used when creating the spectrograms
#and then the values that will be used in the model training
mel_lines = 150
spectrogram_width = 128
TARGET_SIZE = (160, 160)

#sends through a spectrogram and its label, one at a time
def spectrogram_generator(data_directory, label_encoder):
    for path in sorted(glob.glob(os.path.join(data_directory, "*.npy"))):
        fname = os.path.basename(path)
        if fname == "label_encoder_classes.npy":
            continue
        
        class_name = os.path.splitext(fname)[0]
        label = label_encoder.transform([class_name])[0]

        arr = np.load(path, mmap_mode = "r")

        for spec in arr:
            yield spec, label

#transforms the given spectrogram from grayscale to RGB as EfficientNet needs a 3 channel input
def transform_to_rgb(x, label):
    x = tf.cast(x, tf.float32)
    x = tf.expand_dims(x, axis = -1)

    x = tf.image.grayscale_to_rgb(x)
    x = tf.image.resize(x, TARGET_SIZE)
    x = x * 255.0

    return x, label


def dataset_build(
    data_directory,
    label_encoder,
    batch_size=32,
    training=False,
    shuffle_buffer=2000
):

    output_signature = (
        tf.TensorSpec(
            shape=(mel_lines, spectrogram_width),
            dtype=tf.float16
        ),
        tf.TensorSpec(
            shape=(),
            dtype=tf.int32
        )
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: spectrogram_generator(
            data_directory,
            label_encoder
        ),
        output_signature=output_signature
    )

    if training:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.map(
        transform_to_rgb,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=True
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset

#adds the label encoder to the dataset directory so it can be used later when predicting
def build_label_encoder(data_directory):
    class_names = sorted(
        os.path.splitext(os.path.basename(p))[0]
        
        for p in glob.glob(os.path.join(data_directory, "*.npy"))
        if os.path.basename(p) != "label_encoder_classes.npy"
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    return label_encoder