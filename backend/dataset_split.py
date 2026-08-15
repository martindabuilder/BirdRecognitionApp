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


def split_each_class(arr, file_ids, segment_indices):
    unique_files = sorted(set(file_ids))

    rng = np.random.default_rng(42)
    rng.shuffle(unique_files)

    n_files = len(unique_files)

    train_end = int(n_files * 0.70)
    val_end = int(n_files * 0.85)

    train_files = set(unique_files[:train_end])
    val_files = set(unique_files[train_end:val_end])
    test_files = set(unique_files[val_end:])
    file_ids = np.asarray(file_ids)
    segment_indices = np.asarray(segment_indices)

    train_mask = np.isin(file_ids, list(train_files))
    val_mask = np.isin(file_ids, list(val_files))
    test_mask = np.isin(file_ids, list(test_files))

    return (
        arr[train_mask],
        arr[val_mask],
        arr[test_mask],
        file_ids[train_mask],
        file_ids[val_mask],
        file_ids[test_mask],
        segment_indices[train_mask],
        segment_indices[val_mask],
        segment_indices[test_mask]
    )


def main():
    class_files = sorted(glob.glob(os.path.join(dataset, "*.npy")))

    class_files = [
        f for f in class_files
        if os.path.basename(f) != "label_encoder_classes.npy"
        and not f.endswith("_sources.npy")
        and not f.endswith("_segments.npy")
    ]

    summary = []

    for path in class_files:
        class_name = os.path.splitext(os.path.basename(path))[0]
        sources_path = path.replace(".npy","_sources.npy")
        segments_path = path.replace(".npy","_segments.npy")

        if not os.path.exists(sources_path):
            print(
                f"WARNING: Missing sources file for {class_name}"
            )
            continue

        if not os.path.exists(segments_path):
            print(
                f"WARNING: Missing segments file for {class_name}"
            )
            continue

        arr = np.load(path)
        file_ids = np.load(sources_path,allow_pickle=True)
        segment_indices = np.load(segments_path)

        if len(arr) != len(file_ids):
            raise ValueError(f"{class_name}: spectrogram count ({len(arr)}) does not match source count ({len(file_ids)}).")

        if len(arr) != len(segment_indices):
            raise ValueError(f"{class_name}: spectrogram count, ({len(arr)}) does not match segment count, ({len(segment_indices)}).")

        (
            train_set,
            val_set,
            test_set,
            train_sources,
            val_sources,
            test_sources,
            train_segments,
            val_segments,
            test_segments
        ) = split_each_class(
            arr,
            file_ids,
            segment_indices
        )

        train_output = os.path.join( train_set_dir,f"{class_name}.npy")
        val_output = os.path.join(val_set_dir, f"{class_name}.npy")
        test_output = os.path.join(test_set_dir,f"{class_name}.npy")

        np.save( train_output, train_set )
        np.save( train_output.replace( ".npy","_sources.npy" ),train_sources)
        np.save(train_output.replace(".npy","_segments.npy"),train_segments)

        if len(val_set) > 0:
            np.save(val_output,val_set)
            np.save(val_output.replace( ".npy","_sources.npy"),val_sources )
            np.save(val_output.replace(".npy","_segments.npy"),val_segments)

        if len(test_set) > 0:
            np.save(test_output,test_set)
            np.save(test_output.replace(".npy","_sources.npy"),test_sources)
            np.save(test_output.replace(".npy","_segments.npy"),test_segments)

        summary.append((class_name,len(arr),len(train_set),len(val_set),len(test_set)))

    print(f"\nAll classes ({len(summary)}) and their sample counts:")

    for class_name, total, train, val, test in sorted(summary,key=lambda x: x[1]):
        print(f"{class_name}: {total} samples | train={train}, val={val}, test={test}")

    total_train = sum(x[2] for x in summary)
    total_val = sum(x[3] for x in summary)
    total_test = sum(x[4] for x in summary)

    print(f"\nTotal train: {total_train} Total val: {total_val} Total test: {total_test}")

if __name__ == "__main__":
    main()