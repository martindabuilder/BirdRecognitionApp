# Разделят се обработените спектрограми в комплекти за обучение/валидиране/тестване преди обучението на модела.

# Библиотеки и импорти.
import os
import glob
import numpy as np

# Отделни папки за всеки комплект данни.
DATASET = "spectrograms_npy"

TEST_SET_DIR = "test_set"
VAL_SET_DIR = "val_set"
TRAIN_SET_DIR = "train_set"

# Създават се папки ако липсват.
os.makedirs(TEST_SET_DIR, exist_ok = True)
os.makedirs(VAL_SET_DIR, exist_ok = True)
os.makedirs(TRAIN_SET_DIR, exist_ok = True)


# ---- Функции ----

# Главна функция за разделяне.
def split_each_class(arr, file_ids, segment_indices):

    # Зарежда всички уникални аудиозаписи.
    # Раделянето се случва на запис, вместо индивидуален сегмент.
    unique_files = sorted(set(file_ids))

    # На случаен принцип се разместват спектрограмите.
    rng = np.random.default_rng(42)
    rng.shuffle(unique_files)

    n_files = len(unique_files)

    # Разделя спектрограмите на: 70% тренировъчен, 15% валидационен и 15% тестови комплект
    train_end = int(n_files * 0.70)
    val_end = int(n_files * 0.85)

    train_files = set(unique_files[:train_end])
    val_files = set(unique_files[train_end:val_end])
    test_files = set(unique_files[val_end:])

    file_ids = np.asarray(file_ids)
    segment_indices = np.asarray(segment_indices)

    # Усигурява че всички сегменти са от един и същи запис.
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
    # Зарежда всеки клас от комплекта данни.
    class_files = sorted(glob.glob(os.path.join(DATASET, "*.npy")))

    # Само масиви от спектрограми биват обработени (без файловете за метаданни)
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

        # Ако метаданни липсват класа няма да бъде обработен.
        if not os.path.exists(sources_path):
            print(f"WARNING: Missing sources file for {class_name}")
            continue

        # Ако метаданни липсват сегмента няма да бъде обработен.
        if not os.path.exists(segments_path):
            print(f"WARNING: Missing segments file for {class_name}")
            continue

        arr = np.load(path)
        file_ids = np.load(sources_path,allow_pickle=True)
        segment_indices = np.load(segments_path)

        # Проверява дали всички спектрограми имат съответстващ изходен файл. 
        if len(arr) != len(file_ids):
            raise ValueError(f"{class_name}: spectrogram count ({len(arr)}) does not match source count ({len(file_ids)}).")

        # Проверява дали всички спектрограми имат съответстващ индекс на сегмент. 
        if len(arr) != len(segment_indices):
            raise ValueError(f"{class_name}: spectrogram count, ({len(arr)}) does not match segment count, ({len(segment_indices)}).")

        # Разделя класа докато задържа всички сегменти от даден запис в същия комплект.
        (train_set, val_set, test_set, train_sources, val_sources, test_sources, train_segments, val_segments, test_segments) = split_each_class(arr, file_ids, segment_indices)

        # Създават се папки за запазване на комплектите от данни за дадения клас птица.
        train_output = os.path.join(TRAIN_SET_DIR, f"{class_name}.npy")
        val_output = os.path.join(VAL_SET_DIR, f"{class_name}.npy")
        test_output = os.path.join(TEST_SET_DIR,f"{class_name}.npy")

        # Запазват се изходните данни и техните метаданни
        np.save(train_output, train_set )
        np.save(train_output.replace(".npy","_sources.npy" ), train_sources)
        np.save(train_output.replace(".npy","_segments.npy"), train_segments)

        if len(val_set) > 0:
            np.save(val_output,val_set)
            np.save(val_output.replace( ".npy","_sources.npy"), val_sources )
            np.save(val_output.replace(".npy","_segments.npy"), val_segments)

        if len(test_set) > 0:
            np.save(test_output,test_set)
            np.save(test_output.replace(".npy","_sources.npy"), test_sources)
            np.save(test_output.replace(".npy","_segments.npy"), test_segments)
        
        # Запазва се общия брой проби за класа за да се покаже крайното разпределение.
        summary.append((class_name,len(arr),len(train_set),len(val_set),len(test_set)))

    # Изчислява общия брой спектрограми във всяка част на датасета.
    total_train = sum(x[2] for x in summary)
    total_val = sum(x[3] for x in summary)
    total_test = sum(x[4] for x in summary)

    # Извежда в конзолата дистрибуцията на спектрограмите между трите части.
    print(f"\nAll classes ({len(summary)}) and their sample counts:")
    for class_name, total, train, val, test in sorted(summary,key = lambda x: x[1]):
        print(f"{class_name}: {total} samples | train={train}, val={val}, test={test}")

    # Общ брой спектрограми за всеки от трите комплекта.
    print(f"\nTotal train: {total_train} Total val: {total_val} Total test: {total_test}")

if __name__ == "__main__":
    main()