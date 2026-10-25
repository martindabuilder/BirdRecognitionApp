# Файл чрез който се преобразуват всички аудио файлове от датасета в спектрограми за обучение на модела.

# Настройки за средата във Visual Studio Code.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Библиотеки и импорти.
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
import cv2
from tqdm import tqdm
import glob

import warnings
warnings.filterwarnings("ignore")


# Конфигурация на дата сета.
# Папка която съдържа всички базови аудио файлове.
data_directory = os.path.join("..", "dataset")

# Папка в която ще се съхранят преобразуваните спектрограми.
spectrogram_npy_output = "spectrograms_npy"

# Създава се папка за спектрограмите ако липсва.
os.makedirs(spectrogram_npy_output, exist_ok = True)

# Настройки за спектрограмите.
mel_lines = 150 # размер на спектрограмите, измерени в n_mels единици.
sample_rate = 32000 # предварително зададена честота на дискретизация за всяка спектрограма.
audio_duration = 5.0 # секунди за един сегмент.

minimal_active_threshold = 0.30 # минимума активен шум в един сегмент.
max_darkness = 0.15 # при интензивност под 0.15 сегментът ще бъде пропуснат (твърде малко активно пространсто/твърде слабо ниво на шум).

# Избран dtype формат за запазване на спектрограмите.
# Избран е Float16 формат защото е с най-добър баланс между качество и размер в сравнение с uint8/float36 
STORAGE_DTYPE = "float16"

# Използват се максимум 6 от 8 налични ядра на лаптопа за превръщането в спектрограми
# В случай на тестване на различен хардуер може да се увеличи броя ядра.
NUM_WORKERS = min(6, cpu_count())

# Предварително изчисляване на дължината на спектрограмата.
segment_samples = int(audio_duration * sample_rate)
hop_length = int(0.020 * sample_rate)


# ----Функции ----

# -- Функции за филтриране на спектрограми --
# Проверява се дали даден сегмент е достатъчно активен, ако не е по-активен от дадения праг той ще бъде пропуснат
def is_seg_active(spectrogram, threshold = 0.05, min_ratio = minimal_active_threshold):
    active_columns = np.any(spectrogram > threshold, axis = 0)
    ratio = np.sum(active_columns) / len(active_columns)

    return ratio >= min_ratio


# Проверява средното ниво на сила за дадена спектрограма, ако е твърде тиха ще се пропусне.
# Спектрограми с много тъмнина са най-вероятно напълно тихи.
def spectrogram_too_dark(spectrogram):
    return spectrogram.mean() < max_darkness


# Зарежда аудио файл и го превръща на редица 5 секундни сегменти.
def split_audio(file_path, sr = sample_rate, duration = audio_duration):
    y, sr = librosa.load(file_path,sr = sr,res_type = "soxr_hq")

    y, _ = librosa.effects.trim(y)
    segment_length = int(duration * sr)
    hop_samples = int(4.0 * sr) # Позволява 1 секунда препокриване между 2 последователни сегмента.
    segments = []
    
    for start in range(0, len(y), hop_samples):
        end = start + segment_length
        seg = y[start:end]

        # Ако последния сегмент е по-кратък от 5 секунди, празното пространство се запълва с 0.
        if len(seg) < segment_length:
            seg = np.pad(seg,(0, segment_length - len(seg)))
        segments.append(seg)

    return segments, sr


# Превръща един 5 секунден сегмент в нормализирана Mel спектрограма
def audio_to_spectrogram(y, sr):
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 2048, 
    hop_length = hop_length, n_mels = mel_lines, fmin = 200, fmax = 16000, power = 2.0)
    spectrogram_db = librosa.power_to_db(spectrogram, ref = np.max, top_db=80)
    spectrogram_norm = (spectrogram_db + 80.0) / 80.0
    spectrogram_norm = np.clip( spectrogram_norm, 0.0, 1.0)

    return spectrogram_norm.astype(np.float32)


# Зарежда даден аудио файл.
def process_file(file_path):
    try:
        segments, sr = split_audio(file_path)
        
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return []

    file_id = os.path.splitext(os.path.basename(file_path))[0]
    results = []

    for segment_index, seg in enumerate(segments):
        spec = audio_to_spectrogram(seg, sr)

        if not is_seg_active(spec) or spectrogram_too_dark(spec):
            continue

        if STORAGE_DTYPE == "uint8":
            spec = (spec * 255).astype(np.uint8)
        elif STORAGE_DTYPE == "float16":
            spec = spec.astype(np.float16)
        else:
            spec = spec.astype(np.float32)

        results.append((spec, file_id, segment_index))

    return results


# Основен процес за всеки зареден аудио файл.
# Файлът:
# 1. Файла се разделя на 5 секундни сегменти;
# 2. Всеки 5 секунден сегмент се превръща в Mel спектрограма;
# 3. Всеки не-активен/празен сегмент бива премахнат;
# 4. Спектрограмата се запазва с избрания формат.
def process_class(bird_class):
    out_path = os.path.join(spectrogram_npy_output, f"{bird_class}.npy")
    sources_path = out_path.replace(".npy", "_sources.npy")

    # Проверява дали определен клас птица е вече обработен.
    # Ако е обработен, се пропуска
    if os.path.exists(out_path):
        return bird_class, 0

    # настройки които позволяват използване на .mp3 и .wav формат файлове в датасета.
    class_folder = os.path.join(data_directory, bird_class)
    audio_files = (
        glob.glob(os.path.join(class_folder, "*.wav"))
        + glob.glob(os.path.join(class_folder, "*.WAV"))
        + glob.glob(os.path.join(class_folder, "*.mp3"))
        + glob.glob(os.path.join(class_folder, "*.MP3"))
    )

    class_spectrograms = []
    for f in audio_files:
        class_spectrograms.extend(process_file(f))

    if not class_spectrograms:
        return bird_class, 0

    specs, file_ids, segment_indices = zip(*class_spectrograms)
    arr = np.stack(specs)
    np.save(out_path,arr)

    # Оригиналния ID за всеки оригинален сегмент се запазва за отговарящата спектрограма.
    np.save( sources_path, np.array(file_ids))
    np.save(out_path.replace(".npy", "_segments.npy"),np.array(segment_indices, dtype=np.int32))

    return bird_class, len(class_spectrograms)


def main():
    bird_classes = sorted(
        d for d in os.listdir(data_directory)
        if os.path.isdir(os.path.join(data_directory, d))
    )

    print(f"Processing {len(bird_classes)}")

    # Позволява обработката на няколко класа птици наведнъж, използвайки множество ядра на процесора.
    # Използа се за намаляне на времето за обработка.
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm( pool.imap_unordered(process_class, bird_classes),total = len(bird_classes)))

    # Общ брой валидни спектрограмни сегмента.
    total_segments = sum(count for _, count in results)
    
    # Запазват се имената на класове в същия ред в който са по начало.
    label_encoder = LabelEncoder()
    label_encoder.fit(bird_classes)
    np.save(os.path.join(spectrogram_npy_output, "label_encoder_classes.npy"), label_encoder.classes_)


if __name__ == "__main__":
    main()