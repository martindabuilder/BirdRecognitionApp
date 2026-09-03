# Loads the processed spectrogram dataset for PyTorch training.
# Handles labeling, file loading, caching and the conversion 
# from a single to 3 channel spectrogram input.

# Libraries and imports

import os
import glob
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# Settings for the data parsing.
# MEAN and STD set to certain settings, to match the model's requirements.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MAX_CACHED_FILES = 32 # Maximum number of caches spectrograms open at once.


# ---- Functions ----

# PyTorch dataset that provides individual samples and labels.
class SpectrogramDataset(Dataset):
    def __init__(self, data_directory, label_encoder, teacher_probs=None):
        self.teacher_probs = teacher_probs

        # Verifies if the teacher probability matrix matches the number of classes in the dataset.
        if teacher_probs is not None:
            if teacher_probs.shape != (len(label_encoder.classes_), len(label_encoder.classes_)):
                raise ValueError(f"Teacher matrix shape {teacher_probs.shape} does not match {len(label_encoder.classes_)} classes.")

        # Finds all NumPy files containing processed spectrogram classes.
        files = sorted(glob.glob(os.path.join(data_directory, "*.npy")))

        # Stores information about the classes without having to load the entire dataset at once.
        file_paths = []
        path_indices = []
        segment_indices = []
        labels = []

        for path in files:

            # Metadata is being ignored, as it does not contain spectrogram data.
            fname = os.path.basename(path)

            if fname == "label_encoder_classes.npy":
                continue
            if fname.endswith("_sources.npy") or fname.endswith("_segments.npy"):
                continue

            # The filename represents the bird species/class.
            class_name = os.path.splitext(fname)[0]

            if class_name not in label_encoder.classes_:
                continue

            # Converts the species name into an integer class to be used by PyTorch.
            label = label_encoder.transform([class_name])[0]

            # Memory-map the file so it can be accesed from the disk 
            # without loading the entire array into the memory.
            arr = np.load(path, mmap_mode = "r")
            n = arr.shape[0]
            
            # Stores the indexing information used to get individual samples later.
            path_index = len(file_paths)
            file_paths.append(path)

            path_indices.append(np.full(n, path_index, dtype = np.int32))
            segment_indices.append(np.arange(n, dtype = np.int32))
            labels.append(np.full(n, label, dtype = np.int32))

        self.file_paths = file_paths
        self.path_indices = np.concatenate(path_indices) if path_indices else np.array([], dtype = np.int32)
        self.segment_indices = np.concatenate(segment_indices) if segment_indices else np.array([], dtype = np.int32)
        self.labels = np.concatenate(labels) if labels else np.array([], dtype = np.int32)

        # Recenty used files are being caches so they don't need to be reopened from the disk again.
        self._mmap_cache = OrderedDict()

        print(f"{data_directory}: {len(self.labels)} samples")


    def __len__(self):
        return len(self.labels)


    # Function focusing on efficient memory usage.
    def _get_array(self, path_index):
        path = self.file_paths[path_index]

        # if the file is already in the cache it get's reused.
        if path in self._mmap_cache:
            self._mmap_cache.move_to_end(path)
            return self._mmap_cache[path]

        # If it's not already cached, memory-map it to the cache.
        arr = np.load(path, mmap_mode = "r")
        self._mmap_cache[path] = arr

        # Remove the least recently used cached file if the cache memory is full.
        if len(self._mmap_cache) > MAX_CACHED_FILES:
            self._mmap_cache.popitem(last = False)

        return arr


    def __getitem__(self, index):
        # Loads the file, segments and class corresponding to the current sample.
        path_index = int(self.path_indices[index])
        segment_index = int(self.segment_indices[index])
        label = int(self.labels[index])

        # Loads only the required spectrograms instead of the entire dataset.
        arr = self._get_array(path_index)

        spec = torch.from_numpy(np.array(arr[segment_index], dtype = np.float32))

        # Converts the spectrogram using mean and standart deviation.
        # The MEAN and STD are set to match the model's requirements.
        spec = spec.unsqueeze(0).repeat(3, 1, 1)
        spec = (spec - IMAGENET_MEAN) / IMAGENET_STD

        label_tensor = torch.tensor(label, dtype = torch.long)

        if self.teacher_probs is not None:
            teacher_target = torch.tensor(self.teacher_probs[label], dtype = torch.float32)
            return spec, label_tensor, teacher_target

        return spec, label_tensor

    @property
    def samples(self):
        return list(zip((self.file_paths[i] for i in self.path_indices), self.segment_indices, self.labels))


# Builds the main label encoder.
def build_label_encoder(data_directory):
    class_names = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(data_directory, "*.npy"))
        if not os.path.basename(p).endswith("_sources.npy")
        and not os.path.basename(p).endswith("_segments.npy")
        and os.path.basename(p) != "label_encoder_classes.npy"
    )

    # Converts the class names into consistent labels used by the model.
    encoder = LabelEncoder()
    encoder.fit(class_names)

    return encoder