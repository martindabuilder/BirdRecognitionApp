#Seperate file for splitting the dataset into seperate train, validation and test sets
#Done seperately so that the training and go smoother and faster

import os
import glob
import numpy as np

dataset = "spectrograms_npy"
test_set_dir = "test_set"
val_set_dir = "val_set"
train_set_dir = "train_set"
os.makedirs(test_set_dir, exist_ok = True)
os.makedirs(val_set_dir, exist_ok = True)
os.makedirs(train_set_dir, exist_ok = True)

def split_each_class(arr, file_ids):
    unique_files = sorted(set(file_ids))
    n_files = len(unique_files)

    train_end = int(n_files * 0.70)
    val_end = int(n_files * 0.85)

    train_files = set(unique_files[:train_end])
    val_files = set(unique_files[train_end:val_end])
    test_files = set(unique_files[val_end:])

    file_ids = np.array(file_ids)
    train_mask = np.isin(file_ids, list(train_files))
    val_mask = np.isin(file_ids, list(val_files))
    test_mask = np.isin(file_ids, list(test_files))

    train_set = arr[train_mask]
    val_set = arr[val_mask]
    test_set = arr[test_mask]

    return train_set, val_set, test_set

#def main that actually creates the splits and saves them
def main():
    class_files = sorted(glob.glob(os.path.join(dataset, "*.npy")))
    class_files = [
        f for f in class_files
        if os.path.basename(f) != "label_encoder_classes.npy"
        and not f.endswith("_sources.npy")
    ]

    summary = []

    for path in class_files:
        class_name = os.path.splitext(os.path.basename(path))[0]
        sources_path = path.replace(".npy", "_sources.npy")

        arr = np.load(path)
        file_ids = np.load(sources_path, allow_pickle=True)
        n = len(arr)

        train_output = os.path.join(train_set_dir, f"{class_name}.npy")
        val_output = os.path.join(val_set_dir, f"{class_name}.npy")
        test_output = os.path.join(test_set_dir, f"{class_name}.npy")

        train_set, val_set, test_set = split_each_class(arr, file_ids)

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