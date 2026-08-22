import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import shutil

original_train_dir = "train_set"
aug_train_dir = "train_set_augmented"
source_dirs = ["val_set", "test_set"]
train_output_dir = "train_set_resized"
output_suffix = "_resized"
TARGET_SIZE = (224, 224)


def resize_array(arr):
    n = arr.shape[0]
    resized = np.empty((n, TARGET_SIZE[0], TARGET_SIZE[1]), dtype = np.float16)

    for i in range(n):
        resized[i] = cv2.resize(arr[i].astype(np.float32), (TARGET_SIZE[1], TARGET_SIZE[0]), 
        interpolation=cv2.INTER_LINEAR).astype(np.float16)

    return resized


def copy_metadata(path, output_dir):
    fname = os.path.basename(path)
    out_path = os.path.join(output_dir, fname)

    if not os.path.exists(out_path):
        shutil.copy2(path, out_path)


def resize_class_file(path, output_dir):
    fname = os.path.basename(path)
    out_path = os.path.join(output_dir, fname)

    if os.path.exists(out_path):
        return 0

    arr = np.load(path)
    resized = resize_array(arr)
    np.save(out_path, resized)

    return len(arr)

def resize_train_sets(path, output_dir):
    fname = os.path.basename(path)

    #non-augmented train set
    normal = np.load(path)
    normal_resized = resize_array(normal)
    normal_sources_path = path.replace(".npy", "_sources.npy")
    normal_segments_path = path.replace(".npy", "_segments.npy")
    
    normal_sources = np.load(normal_sources_path, allow_pickle = True)
    normal_segments = np.load(normal_segments_path)

    #augmented train set
    augmented_path = os.path.join(aug_train_dir, fname)

    if os.path.exists(augmented_path):
        augmented = np.load(augmented_path)
        augmented_resized = resize_array(augmented)

        combined = np.concatenate([normal_resized, augmented_resized], axis = 0)

        augmented_sources_path = augmented_path.replace(".npy", "_sources.npy")
        augmented_segments_path = augmented_path.replace(".npy", "_segments.npy")

        augmented_sources = np.load(augmented_sources_path, allow_pickle = True)
        augmented_segments = np.load(augmented_segments_path)

        combined_sources = np.concatenate([normal_sources, augmented_sources])
        combined_segments = np.concatenate([normal_segments, augmented_segments])

    else:
        combined = normal_resized
        combined_segments = normal_segments
        combined_sources = normal_sources

    out_path = os.path.join(output_dir, fname)

    np.save(out_path, combined)
    np.save(out_path.replace(".npy", "_sources.npy"), combined_sources)
    np.save(out_path.replace(".npy", "_segments.npy"), combined_segments)

    return len(combined)


def main():
    #resizing the combined augmented and non augmented training sets
    os.makedirs(train_output_dir, exist_ok = True)
    train_files = sorted(glob.glob(os.path.join(original_train_dir, "*.npy")))
    train_files = [
        f for f in train_files
        if os.path.basename(f) != "label_encoder_classes.npy"
        and not f.endswith("_sources.npy")
        and not f.endswith("_segments.npy")
    ]
    
    total_train = 0

    for path in tqdm(train_files, desc = "train_set"):
        total_train += resize_train_sets(path, train_output_dir)

    label_encoder_path = os.path.join(original_train_dir, "label_encoder_classes.npy")

    if os.path.exists(label_encoder_path):
        copy_metadata(label_encoder_path, train_output_dir)


    #resizing val/test sets
    for source_directory in source_dirs:
        output_directory = source_directory + output_suffix
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


    #deleting the original splits if the resize was sucessful
    #done in order to save memory
    if os.path.exists(original_train_dir):
        shutil.rmtree(original_train_dir)

    if os.path.exists(aug_train_dir):
        shutil.rmtree(aug_train_dir)

    for source_directory in source_dirs:
        if os.path.exists(source_directory):
            shutil.rmtree(source_directory)

    print("\nTraining splits successfully resized.")

if __name__ == "__main__":
    main()