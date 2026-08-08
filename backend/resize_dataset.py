#File that gets ran once before the model training
#idea behind it is to resize the spectrograms in 160x160 before the training, rather than during it
#that way the training process is a bit faster and more organized

import osimport glob
import numpy as np
import cv2
from tqdm import tqdm

SOURCE_DIRS = ["train_set", "val_set", "test_set"]
OUTPUT_SUFFIX = "_resized"

TARGET_SIZE = (160, 160)

def resize_class_file(path, output_dir):
    fname = os.path.basename(path)
    out_path = os.path.join(output_dir, fname)

    if os.path.exists(out_path):
        return 0

    arr = np.load(path)
    n = arr.shape[0]