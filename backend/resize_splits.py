#File that gets ran once before the model training
#idea behind it is to resize the spectrograms in 160x160 before the training, rather than during it
#that way the training process is a bit faster and more organized

import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import shutil

SOURCE_DIRS = ["train_set", "val_set", "test_set"]
OUTPUT_SUFFIX = "_resized"

TARGET_SIZE = (224, 224)

def resize_class_file(path, output_dir):
    fname = os.path.basename(path)
    out_path = os.path.join(output_dir, fname)

    if os.path.exists(out_path):
        return 0

    arr = np.load(path)
    n = arr.shape[0]

    resized = np.empty((n, TARGET_SIZE[0], TARGET_SIZE[1]),dtype=np.float16)

    for i in range(n):
        resized[i] = cv2.resize(
            arr[i].astype(np.float32),
            (TARGET_SIZE[1], TARGET_SIZE[0]),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float16)
    np.save(out_path, resized)

    return n


def copy_metadata(path, output_dir):
    fname = os.path.basename(path)
    out_path = os.path.join(output_dir, fname)

    if not os.path.exists(out_path):
        shutil.copy2(path, out_path)


def main():
    for source_directory in SOURCE_DIRS:
        output_directory = source_directory + OUTPUT_SUFFIX
        os.makedirs(output_directory, exist_ok=True)

        class_files = sorted(
            glob.glob(
                os.path.join(
                    source_directory,
                    "*.npy"
                )
            )
        )

        class_files = [
            f for f in class_files
            if os.path.basename(f) != "label_encoder_classes.npy"
            and not f.endswith("_sources.npy")
            and not f.endswith("_segments.npy")
        ]

        total_samples = 0

        for path in tqdm(
            class_files,
            desc=source_directory
        ):
            total_samples += resize_class_file(
                path,
                output_directory
            )

            sources_path = path.replace(".npy", "_sources.npy")
            segments_path = path.replace(".npy", "_segments.npy")

            if os.path.exists(sources_path):
                copy_metadata(
                    sources_path,
                    output_directory
                )

            if os.path.exists(segments_path):
                copy_metadata(
                    segments_path,
                    output_directory
                )

        print(f"{source_directory}: {total_samples} samples")

    print("\nTraining splits successfully resized.")

if __name__ == "__main__":
    main()