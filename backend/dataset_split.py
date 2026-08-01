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

TEST_SIZE = 0.15
VAL_SIZE = 0.15
TRAIN_SIZE = 0.7

MINIMUM_SAMPLES_PER_CLASS = 50
RANDOM_SEED = 42

#shuffles and splits a given single class into seperate train, validation and test sets
def split_each_class(arr, rng):
    n = len(arr)
    indices = rng.permutation(n)
    arr = arr[indices]

    train_end = int(n * TRAIN_SIZE)
    val_end = train_end + int(n * VAL_SIZE)

    train_set = arr[:train_end]
    val_set = arr[train_end:val_end]
    test_set = arr[val_end:]

    return train_set, val_set, test_set

#def main that actually creates the splits and saves them
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    class_files = sorted(glob.glob(os.path.join(dataset, "*.npy")))
    class_files = [f for f in class_files if os.path.basename(f) != "label_encoder_classes.npy"]

    too_few_samples_classes = 0
    too_few_samples_classes_list = []
    summary = []

    for path in class_files:
        class_name = os.path.splitext(os.path.basename(path))[0]
        arr = np.load(path)
        n = len(arr)

        train_output = os.path.join(train_set_dir, f"{class_name}.npy")
        val_output = os.path.join(val_set_dir, f"{class_name}.npy")
        test_output = os.path.join(test_set_dir, f"{class_name}.npy")

        #if they exist ^, they get skipped
        if os.path.exists(train_output) and os.path.exists(val_output) and os.path.exists(test_output):
            continue

        #if the class has too few samples, it gets skipped
        #and added directly to the train set
        if n < MINIMUM_SAMPLES_PER_CLASS:
            np.save(train_output, arr)
            too_few_samples_classes += 1
            too_few_samples_classes_list.append((class_name, n))
            summary.append((class_name, n, n, 0, 0))
            continue

        train_set, val_set, test_set = split_each_class(arr, rng)

        np.save(train_output, train_set)
        if len(val_set) > 0:
            np.save(val_output, val_set)
        if len(test_set) > 0:
            np.save(test_output, test_set)

        summary.append((class_name, n, len(train_set), len(val_set), len(test_set)))


    if too_few_samples_classes_list:
        print(f"Classes with too few samples: {too_few_samples_classes}")

        for class_name, n in too_few_samples_classes_list:
            print(f"{class_name}: {n} samples")

    #final information after completing the splitting
    total_train_samples = sum([s[2] for s in summary])
    total_val_samples = sum([s[3] for s in summary])
    total_test_samples = sum([s[4] for s in summary])
    print(f"Total train: {total_train_samples}, val: {total_val_samples}, test: {total_test_samples}")


#main
if __name__ == "__main__":
    main()