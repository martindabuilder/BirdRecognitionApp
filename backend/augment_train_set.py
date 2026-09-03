# Applies SpecAugment to a portion of the training set.
# Validation and test subsets are left unchanged so the 
# model's performance can be measured to the original data.

# Libraries and imports

import os
import glob
import numpy as np
from tqdm import tqdm

# Folders and settings for the augmentation process
TRAIN_DIR = "train_set"
OUTPUT_DIR = "train_set_augmented"

AUGMENT_RATIO = 0.30 # % of the original training samples that will be augmented.
NUM_AUGMENTATIONS = 1 # Number of augmented versions of each sample.
TIME_MASK_MAX = 20 # Max width of the masked time region.
FREQ_MASK_MAX = 15 # Max width of the masked frequency region.
SEED = 42 # Fixed random seed.

# Creates output folder if it's missing.
os.makedirs(OUTPUT_DIR, exist_ok = True)

def spec_augment(spec, rng):
    augmented = spec.copy()
    n_mels, n_frames = augmented.shape
    freq_width = rng.integers(0, min(FREQ_MASK_MAX, n_mels) + 1)

    if freq_width > 0:
        freq_start = rng.integers(0, n_mels - freq_width + 1)
        augmented[freq_start:freq_start + freq_width, :] = 0.0

    time_width = rng.integers(0, min(TIME_MASK_MAX, n_frames) + 1)

    if time_width > 0:
        time_start = rng.integers(0, n_frames - time_width + 1)
        augmented[:, time_start:time_start + time_width] = 0.0

    return augmented

def augment_class(path, rng):
    class_name = os.path.splitext(os.path.basename(path))[0]

    if (class_name.endswith("_sources") or class_name.endswith("_segments")):
        return 0

    output_path = os.path.join(OUTPUT_DIR, f"{class_name}.npy")
    sources_output = os.path.join(OUTPUT_DIR, f"{class_name}_sources.npy")
    segments_output = os.path.join(OUTPUT_DIR,f"{class_name}_segments.npy")
    sources_path = path.replace(".npy", "_sources.npy")
    segments_path = path.replace(".npy", "_segments.npy")

    if not os.path.exists(sources_path):
        print(f"Missing sources for {class_name}")
        return 0

    if not os.path.exists(segments_path):
        print(f"Missing segments for {class_name}")
        return 0

    arr = np.load(path)
    file_ids = np.load(sources_path, allow_pickle = True)
    segment_indices = np.load(segments_path)

    if len(arr) == 0:
        return 0
        
    n_augmented = int(len(arr) * AUGMENT_RATIO)

    if n_augmented == 0:
        return 0

    selected_indices = rng.choice(len(arr),size = n_augmented, replace = False)
    augmented_samples = []
    augmented_sources = []
    augmented_segments = []

    for index in selected_indices:
        original = arr[index].astype(np.float32)
        for _ in range(NUM_AUGMENTATIONS):
            augmented = spec_augment(original, rng)
            augmented_samples.append(augmented.astype(np.float16))
            augmented_sources.append(file_ids[index])
            augmented_segments.append(segment_indices[index])

    if not augmented_samples:
        return 0

    augmented_samples = np.stack(augmented_samples)
    augmented_sources = np.asarray(augmented_sources)
    augmented_segments = np.asarray(augmented_segments, dtype = np.int32)

    np.save(output_path, augmented_samples)
    np.save(sources_output, augmented_sources)
    np.save(segments_output, augmented_segments)

    return len(augmented_samples)


def main():
    rng = np.random.default_rng(SEED)
    class_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.npy")))

    class_files = [
        f for f in class_files
        if not f.endswith("_sources.npy")
        and not f.endswith("_segments.npy")
    ]

    total = 0

    for path in tqdm(class_files, desc = "SpecAugment"):
        total += augment_class(path, rng)

    print(f"\nCreated {total} augmented samples.")

if __name__ == "__main__":
    main()