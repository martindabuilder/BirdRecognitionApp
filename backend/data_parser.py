import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# same values used when creating spectrograms
mel_lines = 150
spectrogram_width = 128
TARGET_SIZE = (160, 160) #target resize for the efficientnet

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class SpectrogramDataset(Dataset):
    def __init__(self, data_directory, label_encoder):
        self.samples = []
        files = sorted(glob.glob(os.path.join(data_directory, "*.npy")))

        for path in files:
            fname = os.path.basename(path)

            if fname == "label_encoder_classes.npy":
                continue

            class_name = os.path.splitext(fname)[0]
            label = label_encoder.transform([class_name])[0]

            arr = np.load(path, mmap_mode = "r")

            for i in range(arr.shape[0]):
                self.samples.append((path, i, label))

        print(f"{data_directory}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        path, idx, label = self.samples[index]
        arr = np.load(path, mmap_mode = "r")
        spec = arr[idx]

        spec = torch.tensor(np.array(spec), dtype = torch.float32)
        spec = spec.unsqueeze(0)
        spec = spec.repeat(3, 1, 1)

        spec = (spec - IMAGENET_MEAN) / IMAGENET_STD
        
        return(spec, torch.tensor(label, dtype = torch.long))

def build_label_encoder(data_directory):
    class_names = sorted(
        os.path.splitext(os.path.basename(p))[0]

        for p in glob.glob(
            os.path.join(data_directory, "*.npy")
        )

        if os.path.basename(p) != "label_encoder_classes.npy"
    )

    encoder = LabelEncoder()
    encoder.fit(class_names)

    return encoder