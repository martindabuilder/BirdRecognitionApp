import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

test_set_dir = "test_set"
val_set_dir = "val_set"
train_set_dir = "train_set"

os.makedirs(test_set_dir, exist_ok = True)
os.makedirs(val_set_dir, exist_ok = True)
os.makedirs(train_set_dir, exist_ok = True)