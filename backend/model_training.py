import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():

    #folders containing all the train sets and the model directory
    train_set_dir = "train_set"
    val_set_dir = "val_set"
    test_set_dir = "test_set"

    model_directory = "model"
    os.makedirs( model_directory, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    #settings for the model later on
    BATCH_SIZE = 32
    EPOCHS = 20

if __name__ == "__main__":
    main()