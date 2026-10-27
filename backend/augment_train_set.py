# Прилага SpecAugment филтри върху част от тренировъчния комплект.
# Валидационния и тестовия комплект остават непроменени, за да може производителността на модела да бъде измерена спрямо непроменените комплекти.

# Библиотеки и импорти.
import os
import glob
import numpy as np
from tqdm import tqdm

# Папки и настройки за аугментирането на данните.
TRAIN_DIR = "train_set"
OUTPUT_DIR = "train_set_augmented"

AUGMENT_RATIO = 0.30 # 30% от оригиналните данни ще бъдат обработени.
NUM_AUGMENTATIONS = 1 # Брой аугментирани версии на всяка проба.
TIME_MASK_MAX = 20 # Максимална широчина на региона за времево маскиране.
FREQ_MASK_MAX = 15 # Максимална височина на региона за честотно маскиране.
SEED = 42 # Fixed random seed.

#Създава папка за изходните данни ако липсва.
os.makedirs(OUTPUT_DIR, exist_ok = True)


# ---- Функции ----

def spec_augment(spec, rng):
    # Работи се върху копие на спектрограмата, не оригинала.
    augmented = spec.copy()
    n_mels, n_frames = augmented.shape
    freq_width = rng.integers(0, min(FREQ_MASK_MAX, n_mels) + 1)

    # Случайно приложена височина за честотно маскиране.
    if freq_width > 0:
        freq_start = rng.integers(0, n_mels - freq_width + 1)
        augmented[freq_start:freq_start + freq_width, :] = 0.0

    time_width = rng.integers(0, min(TIME_MASK_MAX, n_frames) + 1)

    # Случайно приложена широчина за времево маскиране.
    if time_width > 0:
        time_start = rng.integers(0, n_frames - time_width + 1)
        augmented[:, time_start:time_start + time_width] = 0.0

    # Връща обработената спектрограма.
    return augmented


# Основна функция която извиква SpecAugment за целия клас.
def augment_class(path, rng):
    class_name = os.path.splitext(os.path.basename(path))[0]

    if (class_name.endswith("_sources") or class_name.endswith("_segments")):
        return 0

    # Изходни файлове за обработените спектрограми и техните метаданни.
    output_path = os.path.join(OUTPUT_DIR, f"{class_name}.npy")
    sources_output = os.path.join(OUTPUT_DIR, f"{class_name}_sources.npy")
    segments_output = os.path.join(OUTPUT_DIR,f"{class_name}_segments.npy")
    sources_path = path.replace(".npy", "_sources.npy")
    segments_path = path.replace(".npy", "_segments.npy")

    if not os.path.exists(sources_path):
        print(f"Missing sources for {class_name}")
        return 0

    if not os.path.exists(segments_path):
        print(f"Missing segments for {class_name}")
        return 0

    # Зарежда избраните спектрограми и техните метаданни/
    arr = np.load(path)
    file_ids = np.load(sources_path, allow_pickle = True)
    segment_indices = np.load(segments_path)

    # Празни класове биват пропуснати.
    if len(arr) == 0:
        return 0
    
    # Пресмята се колко от тренировъчните данни ще бъдат аугментирани
    n_augmented = int(len(arr) * AUGMENT_RATIO)

    # Ако класа е твърде малък се пропуска.
    if n_augmented == 0:
        return 0

    # Избира случайни проби да аугментира, като всяка проба се избира само веднъж
    selected_indices = rng.choice(len(arr), size = n_augmented, replace = False)
    augmented_samples = []
    augmented_sources = []
    augmented_segments = []

    # Създава аугментирани версии на избраните проби.
    for index in selected_indices:
        original = arr[index].astype(np.float32)
        for _ in range(NUM_AUGMENTATIONS):
            augmented = spec_augment(original, rng)
            augmented_samples.append(augmented.astype(np.float16))
            augmented_sources.append(file_ids[index])
            augmented_segments.append(segment_indices[index])

    if not augmented_samples:
        return 0

    # Преобразува пробите и техните метаданни в NumPy масиви.
    augmented_samples = np.stack(augmented_samples)
    augmented_sources = np.asarray(augmented_sources)
    augmented_segments = np.asarray(augmented_segments, dtype = np.int32)

    # Запазва аугментираните тренировъчни данни и техните метаданни
    np.save(output_path, augmented_samples)
    np.save(sources_output, augmented_sources)
    np.save(segments_output, augmented_segments)

    return len(augmented_samples)


def main():
    rng = np.random.default_rng(SEED)
    class_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.npy")))

    class_files = [
        f for f in class_files
        if not f.endswith("_sources.npy")
        and not f.endswith("_segments.npy")
    ]

    total = 0

    for path in tqdm(class_files, desc = "SpecAugment"):
        total += augment_class(path, rng)

    print(f"\nCreated {total} augmented samples.")

if __name__ == "__main__":
    main()