# Resizes the processed spectrograms to match the input size 
# required by the EfficientNet model.

# Libraries and imports

import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import shutil

# Folders and settings
ORIGINAL_TRAIN_DIR = "train_set"
AUG_TRAIN_DIR = "train_set_augmented"
SOURCE_DIRS = ["val_set", "test_set"]
TRAIN_OUTPUT_DIR = "train_set_resized"
OUTPUT_SUFFIX = "_resized"

TARGET_SIZE = (224, 224) # EfficientNetB0 target size.


# ---- FUNCTIONS ---


def resize_array(arr):
    # Every spectrogram is resized to the same dimensions required by the model.
    n = arr.shape[0]
    resized = np.empty((n, TARGET_SIZE[0], TARGET_SIZE[1]), dtype = np.float16)

    # Convers to float32 during interpolation.
    # Done for a more stable and reliable resizing.
    for i in range(n):
        resized[i] = cv2.resize(arr[i].astype(np.float32), (TARGET_SIZE[1], TARGET_SIZE[0]), 
        interpolation=cv2.INTER_LINEAR).astype(np.float16)

    return resized


# Copies the metadata files so the resized set remains with the same mapping and metadata.
def copy_metadata(path, output_dir):
    fname = os.path.basename(path)
    out_path = os.path.join(output_dir, fname)

    if not os.path.exists(out_path):
        shutil.copy2(path, out_path)


# Main resizing function for the validation/test subsplits.
def resize_class_file(path, output_dir):
    fname = os.path.basename(path)
    out_path = os.path.join(output_dir, fname)

    # Skips already resized classes
    if os.path.exists(out_path):
        return 0

    # Loads and resizes all spectrograms from the respective class.
    arr = np.load(path)
    resized = resize_array(arr)
    np.save(out_path, resized)

    return len(arr)

# Additional function that resizes the augmented and non-augmented train sets.
def resize_train_sets(path, output_dir):
    fname = os.path.basename(path)

    #Loading and resizing the non-augmented train set.
    normal = np.load(path)
    normal_resized = resize_array(normal)
    normal_sources_path = path.replace(".npy", "_sources.npy")
    normal_segments_path = path.replace(".npy", "_segments.npy")
    
    normal_sources = np.load(normal_sources_path, allow_pickle = True)
    normal_segments = np.load(normal_segments_path)

    # Loading the augmented train set.
    augmented_path = os.path.join(AUG_TRAIN_DIR, fname)

    if os.path.exists(augmented_path):
        augmented = np.load(augmented_path)
        augmented_resized = resize_array(augmented)

        # Combines the augmented and non-augmented subsets into one whole training set.
        combined = np.concatenate([normal_resized, augmented_resized], axis = 0)

        augmented_sources_path = augmented_path.replace(".npy", "_sources.npy")
        augmented_segments_path = augmented_path.replace(".npy", "_segments.npy")

        augmented_sources = np.load(augmented_sources_path, allow_pickle = True)
        augmented_segments = np.load(augmented_segments_path)

        # Combines the metadata for the training sets.
        combined_sources = np.concatenate([normal_sources, augmented_sources])
        combined_segments = np.concatenate([normal_segments, augmented_segments])

    else:
        # If no augmented data exists, only uses the non-augmented original train set.
        combined = normal_resized
        combined_segments = normal_segments
        combined_sources = normal_sources

    out_path = os.path.join(output_dir, fname)

    # Saves the training data and its metadata.
    np.save(out_path, combined)
    np.save(out_path.replace(".npy", "_sources.npy"), combined_sources)
    np.save(out_path.replace(".npy", "_segments.npy"), combined_segments)

    return len(combined)


def main():
    # Resizing and combining the augmented and non augmented training sets
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok = True)
    train_files = sorted(glob.glob(os.path.join(ORIGINAL_TRAIN_DIR, "*.npy")))
    train_files = [
        f for f in train_files
        if os.path.basename(f) != "label_encoder_classes.npy"
        and not f.endswith("_sources.npy")
        and not f.endswith("_segments.npy")
    ]
    
    total_train = 0

    for path in tqdm(train_files, desc = "train_set"):
        total_train += resize_train_sets(path, TRAIN_OUTPUT_DIR)

    label_encoder_path = os.path.join(ORIGINAL_TRAIN_DIR, "label_encoder_classes.npy")

    if os.path.exists(label_encoder_path):
        copy_metadata(label_encoder_path, TRAIN_OUTPUT_DIR)


    # Resizing the seperate val/test sets.
    for source_directory in SOURCE_DIRS:
        output_directory = source_directory + OUTPUT_SUFFIX
        os.makedirs(output_directory, exist_ok = True)

        class_files = sorted(glob.glob(os.path.join(source_directory, "*.npy")))

        class_files = [
            f for f in class_files
            if os.path.basename(f) != "label_encoder_classes.npy"
            and not f.endswith("_sources.npy")
            and not f.endswith("_segments.npy")
        ]

        total_samples = 0

        for path in tqdm(class_files, desc = source_directory):
            total_samples += resize_class_file(path, output_directory)

            sources_path = path.replace(".npy", "_sources.npy")
            segments_path = path.replace(".npy", "_segments.npy")

            copy_metadata(sources_path, output_directory)
            copy_metadata(segments_path, output_directory)


        label_encoder_path = os.path.join(source_directory, "label_encoder_classes.npy")


    # Deleting the original splits if the resize was sucessful.
    # Done in order to save space.
    if os.path.exists(ORIGINAL_TRAIN_DIR):
        shutil.rmtree(ORIGINAL_TRAIN_DIR)

    if os.path.exists(AUG_TRAIN_DIR):
        shutil.rmtree(AUG_TRAIN_DIR)

    for source_directory in SOURCE_DIRS:
        if os.path.exists(source_directory):
            shutil.rmtree(source_directory)

    print("\nTraining splits successfully resized.")

if __name__ == "__main__":
    main()