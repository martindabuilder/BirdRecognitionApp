#Seperate file for splitting the dataset into seperate train, validation and test sets
#Done seperately so that the training and go smoother and faster

import os
import glob
import numpy as np

dataset = "spectrograms_npy"
test_set_dir = "test_set"
val_set_dir = "val_set"
train_set_dir = "train_set"
os.makedirs(test_set_dir, val_set_dir, train_set_dir, exist_ok = True)

RANDOM_SEED = 42

#shuffles and splits a given single class into seperate train, validation and test sets
def split_each_class(arr, rng):
    n = len(arr)

    indices = rng.permutation(n)
    arr = arr[indices]

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_set = arr[:train_end]
    val_set = arr[train_end:val_end]
    test_set = arr[val_end:]

    return train_set, val_set, test_set

#def main that actually creates the splits and saves them
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    class_files = sorted(glob.glob(os.path.join(dataset, "*.npy")))
    class_files = [f for f in class_files if os.path.basename(f) != "label_encoder_classes.npy"]

    summary = []

    for path in class_files:
        class_name = os.path.splitext(os.path.basename(path))[0]
        arr = np.load(path)
        n = len(arr)

        train_output = os.path.join(train_set_dir, f"{class_name}.npy")
        val_output = os.path.join(val_set_dir, f"{class_name}.npy")
        test_output = os.path.join(test_set_dir, f"{class_name}.npy")

        train_set, val_set, test_set = split_each_class(arr, rng)

        np.save(train_output, train_set)
        if len(val_set) > 0:
            np.save(val_output, val_set)

        if len(test_set) > 0:
            np.save(test_output, test_set)

        summary.append((class_name, n, len(train_set), len(val_set), len(test_set)))

    #full listing of every class and its total sample count, sorted lowest first
    print(f"\nAll classes ({len(summary)}) and their sample counts:")
    for class_name, n, _, _, _ in sorted(summary, key = lambda s: s[1]):
        print(f"{class_name}: {n} samples")

    #final information after completing the splitting
    total_train_samples = sum([s[2] for s in summary])
    total_val_samples = sum([s[3] for s in summary])
    total_test_samples = sum([s[4] for s in summary])
    print(f"Total train: {total_train_samples}, val: {total_val_samples}, test: {total_test_samples}")


#main
if __name__ == "__main__":
    main()