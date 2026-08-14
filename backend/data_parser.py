import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SpectrogramDataset(Dataset):
    def __init__(self, data_directory, label_encoder, teacher_probs=None):
        self.samples = []
        self.teacher_probs = teacher_probs
        files = sorted(glob.glob(os.path.join(data_directory, "*.npy")))

        for path in files:
            fname = os.path.basename(path)

            if fname == "label_encoder_classes.npy":
                continue

            if fname.endswith("_sources.npy") or fname.endswith("_segments.npy"):
                continue

            class_name = os.path.splitext(fname)[0]
            if class_name not in label_encoder.classes_:
                continue

            label = label_encoder.transform([class_name])[0]

            arr = np.load(path, mmap_mode="r")

            for i in range(arr.shape[0]):
                self.samples.append((path, i, label))

        print(f"{data_directory}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, idx, label = self.samples[index]
        arr = np.load(path, mmap_mode="r")
        spec = torch.from_numpy(np.array(arr[idx], dtype=np.float32))

        spec = spec.unsqueeze(0).repeat(3, 1, 1)
        spec = (spec - IMAGENET_MEAN) / IMAGENET_STD

        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.teacher_probs is not None:
            teacher_target = torch.tensor(self.teacher_probs[label],dtype=torch.float32)
            return spec, label_tensor, teacher_target

        return spec, label_tensor

def build_label_encoder(data_directory):
    class_names = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(data_directory, "*.npy"))
        if not os.path.basename(p).endswith("_sources.npy")
        and not os.path.basename(p).endswith("_segments.npy")
        and os.path.basename(p) != "label_encoder_classes.npy"
    )

    encoder = LabelEncoder()
    encoder.fit(class_names)

    return encoder