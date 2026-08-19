import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
from pathlib import Path
import numpy as np
import pandas as pd
from birdnet import load

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIRECTORY = PROJECT_ROOT / "dataset"
TAXONOMY_PATH = DATA_DIRECTORY / "eBird_taxonomy_v2025-4.csv"
OUTPUT_DIR = BASE_DIR / "birdnet_teacher"
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

CHECKPOINT = 5
FILES_PER_CLASS = 25
RANDOM_SEED = 42
N_WORKERS = 4
PREDICT_BATCH_SIZE = 5
TOP_K = 20

CHECKPOINT_MATRIX = (OUTPUT_DIR / "teacher_matrix_checkpoint.npy")
CHECKPOINT_INDEX = (OUTPUT_DIR / "teacher_checkpoint_index.npy")

def normalise_name(name):
    if pd.isna(name):
        return None
    name = str(name).strip()
    if not name:
        return None
    return " ".join(name.split())


def birdnet_to_scientific_name(species_label):
    species_label = normalise_name(species_label)

    if species_label is None:
        return None

    species_label = species_label.replace("_", " ")
    parts = species_label.split()

    if len(parts) >= 2:
        return normalise_name(f"{parts[0]} {parts[1]}")
    return species_label


def build_code_to_scientific(taxonomy_df):
    return dict(
        zip(taxonomy_df["SPECIES_CODE"], taxonomy_df["SCI_NAME"]))


def get_birdnet_species_probs(model, file_paths):
    file_paths = [Path(path) for path in file_paths]

    def predict_single_file(file_path):
        try:
            result = model.predict([str(file_path)], top_k=TOP_K, n_workers=N_WORKERS, n_producers=1, batch_size=1, device="CPU", show_stats="minimal")

        except Exception as e:
            print(f"{file_path.name}: individual prediction failed: {type(e).__name__}: {e}")
            return {}

        species_list = np.asarray(result.species_list)
        species_probs = np.asarray(result.species_probs)
        species_ids = np.asarray(result.species_ids)
        species_masked = np.asarray(result.species_masked)

        while species_probs.ndim > 2:
            species_probs = species_probs[0]

        while species_ids.ndim > 2:
            species_ids = species_ids[0]

        while species_masked.ndim > 2:
            species_masked = species_masked[0]

        if species_probs.ndim == 1:
            species_probs = species_probs[np.newaxis, :]

        if species_ids.ndim == 1:
            species_ids = species_ids[np.newaxis, :]

        if species_masked.ndim == 1:
            species_masked = species_masked[np.newaxis, :]

        if species_probs.ndim != 2:
            print(f"{file_path.name}: unexpected probability shape {species_probs.shape}")
            return {}

        if species_ids.ndim != 2:
            print(f"{file_path.name}: unexpected ID shape {species_ids.shape}")
            return {}

        accumulated = {}
        counts = {}

        for segment_index in range(species_probs.shape[0]):
            for prediction_index in range(species_probs.shape[1]):

                if (segment_index >= species_ids.shape[0] or prediction_index >= species_ids.shape[1]):
                    continue

                # Ignore masked predictions.
                if (
                    species_masked.ndim == 2
                    and
                    segment_index < species_masked.shape[0]
                    and
                    prediction_index < species_masked.shape[1]
                ):

                    if species_masked[segment_index, prediction_index]:
                        continue

                species_id = int(species_ids[segment_index, prediction_index])

                if (species_id < 0 or species_id >= len(species_list)):
                    continue

                probability = float(species_probs[segment_index, prediction_index])

                if (not np.isfinite(probability) or probability <= 0):
                    continue

                species_label = str(species_list[species_id]).strip()

                if not species_label:
                    continue

                scientific_name = (birdnet_to_scientific_name(species_label))

                if scientific_name is None:
                    continue

                accumulated[scientific_name] = (accumulated.get(scientific_name,0.0) + probability)
                counts[scientific_name] = (counts.get(scientific_name, 0) + 1)

        if not accumulated:
            return {}

        averaged = {
            name:
                accumulated[name] / counts[name]
            for name in accumulated
        }

        total = sum(averaged.values())

        if total <= 0:
            return {}

        return {
            name:
                probability / total
            for name, probability in averaged.items()
        }

    try:
        result = model.predict([str(path) for path in file_paths], top_k=TOP_K, n_workers=N_WORKERS, n_producers=1, batch_size=PREDICT_BATCH_SIZE, device="CPU", show_stats="minimal")

    except Exception as e:
        print(f"\nBirdNET batch failed: {type(e).__name__}: {e}")
        print("Falling back to individual file prediction...")
        results = []

        for file_path in file_paths:
            print(f"Retrying individually: {file_path.name}")
            predictions = predict_single_file(file_path)
            results.append(predictions)

        return results

    species_list = np.asarray(result.species_list)
    species_probs = np.asarray(result.species_probs)
    species_ids = np.asarray(result.species_ids)
    species_masked = np.asarray(result.species_masked)

    if species_probs.ndim == 2:
        species_probs = species_probs[np.newaxis, ...]

    if species_ids.ndim == 2:
        species_ids = species_ids[np.newaxis, ...]

    if species_masked.ndim == 2:
        species_masked = species_masked[np.newaxis, ...]

    if species_probs.ndim != 3:

        print("Unexpected BirdNET probability shape:", species_probs.shape)

        return [
            predict_single_file(path)
            for path in file_paths
        ]

    if species_ids.ndim != 3:
        print("Unexpected BirdNET ID shape:", species_ids.shape)

        return [
            predict_single_file(path)
            for path in file_paths
        ]

    file_predictions = []
    number_of_files = min(len(file_paths), species_probs.shape[0])

    for file_index in range(number_of_files):
        accumulated = {}
        counts = {}

        for segment_index in range(species_probs.shape[1]):
            for prediction_index in range(species_probs.shape[2]):

                if (file_index >= species_ids.shape[0] or segment_index >= species_ids.shape[1] or prediction_index >= species_ids.shape[2]):
                    continue

                if (
                    species_masked.ndim == 3
                    and
                    file_index < species_masked.shape[0]
                    and
                    segment_index < species_masked.shape[1]
                    and
                    prediction_index < species_masked.shape[2]
                ):

                    if species_masked[file_index, segment_index, prediction_index]:
                        continue

                species_id = int(species_ids[file_index, segment_index, prediction_index])

                if (species_id < 0 or species_id >= len(species_list)):
                    continue

                probability = float(species_probs[file_index, segment_index, prediction_index])

                if (not np.isfinite(probability) or probability <= 0):
                    continue

                species_label = str(species_list[species_id]).strip()

                if not species_label:
                    continue

                scientific_name = (birdnet_to_scientific_name(species_label))

                if scientific_name is None:
                    continue

                accumulated[scientific_name] = (
                    accumulated.get(scientific_name, 0.0) + probability)

                counts[scientific_name] = (counts.get(scientific_name, 0) + 1)

        if not accumulated:
            file_predictions.append({})
            continue

        averaged = {
            name:
                accumulated[name] / counts[name]
            for name in accumulated
        }

        total = sum(averaged.values())

        if total <= 0:
            file_predictions.append({})
            continue

        normalised = {
            name:
                probability / total
            for name, probability in averaged.items()
        }

        file_predictions.append(normalised)

    while len(file_predictions) < len(file_paths):
        file_predictions.append({})

    return file_predictions

def print_top_predictions(predictions, expected_name=None, n=10):
    if not predictions:
        print("No predictions.")
        return

    predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    expected_normalised = (
        normalise_name(expected_name)
        if expected_name is not None
        else None
    )

    for rank, (name, probability) in enumerate(predictions[:n], start=1):
        marker = ""
        if (expected_normalised is not None and normalise_name(name) == expected_normalised):
            marker = " < expected"
        print(f"{rank:2d} {name:<45} {probability:8.3%} {marker}")

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    birdnet_model = load("acoustic", "2.4", "tf")
    taxonomy_df = pd.read_csv(TAXONOMY_PATH)
    code_to_sci = build_code_to_scientific(taxonomy_df)

    print("BirdNET 2.4 loaded.")

    bird_classes = sorted(
        d
        for d in os.listdir(DATA_DIRECTORY)
        if (os.path.isdir(DATA_DIRECTORY / d) and d != "birdnet_teacher")
    )

    num_classes = len(bird_classes)
    our_scientific_names = []

    for code in bird_classes:
        name = normalise_name(code_to_sci.get(code))

        our_scientific_names.append(name)
        print(f"{code} -> {name}")

  
    name_to_index = {
        normalise_name(name): i
        for i, name in enumerate(our_scientific_names)
        if name is not None
    }

    if (CHECKPOINT_MATRIX.exists() and CHECKPOINT_INDEX.exists()):
        teacher_matrix = np.load(CHECKPOINT_MATRIX)

        start_index = int(
            np.load(CHECKPOINT_INDEX))

        if teacher_matrix.shape != (num_classes, num_classes):
            raise ValueError("Teacher checkpoint shape does not match current dataset.")

        print(f"Resuming from class {start_index + 1} / {num_classes}.")

    else:
        teacher_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.float32)

        start_index = 0

    total_files = 0
    successful_files = 0
    expected_found_count = 0

    for i, code in enumerate( bird_classes[start_index:], start=start_index):
        class_folder = (DATA_DIRECTORY / code)

        audio_files = (
            list(class_folder.glob("*.mp3"))
            + list(class_folder.glob("*.MP3"))
            + list(class_folder.glob("*.wav"))
            + list(class_folder.glob("*.WAV"))
        )

        if not audio_files:
            print(f"{code}: no audio files found")
            teacher_matrix[i] = (1.0 / num_classes)
            continue


        sample_files = random.sample(
            audio_files, min(FILES_PER_CLASS, len(audio_files)))

        expected_name = (our_scientific_names[i])

        print(f"\nClass {i + 1}/{num_classes}: {code} -> {expected_name}")
        print(f"Predicting {len(sample_files)} files in batches of {PREDICT_BATCH_SIZE}...")

        accumulated = np.zeros(num_classes, dtype=np.float32)
        successful = 0

        for batch_start in range(0, len(sample_files), PREDICT_BATCH_SIZE):
            batch_files = sample_files[batch_start: batch_start + PREDICT_BATCH_SIZE]
            print(f"\nBatch {batch_start + 1}-{batch_start + len(batch_files)}/{len(sample_files)}")
            predictions_batch = (get_birdnet_species_probs(birdnet_model, batch_files))

            for file_path, predictions in zip(batch_files, predictions_batch):
                total_files += 1

                if not predictions:
                    print(f"{file_path.name}: No usable prediction.")
                    continue

                successful += 1
                successful_files += 1

                print(f"\n  {file_path.name}")
                print_top_predictions(predictions, expected_name)

                expected_normalised = (normalise_name(expected_name))

                prediction_names = {
                    normalise_name(name)
                    for name in predictions
                }

                if (expected_normalised in prediction_names):
                    expected_found_count += 1

                for name, probability in (predictions.items()):
                    index = name_to_index.get(normalise_name(name))
                    if index is not None:
                        accumulated[index] += (probability)

     
        if successful > 0:
            accumulated /= successful
            total = accumulated.sum()
            if total > 0:
                accumulated /= total
            else:
                accumulated[:] = (1.0 / num_classes)

        else:
            accumulated[:] = (1.0 / num_classes)

        teacher_matrix[i] = accumulated
        top_indices = np.argsort(accumulated)[::-1][:10]
        print("\nTop dataset classes:")

        expected_normalised = (normalise_name(expected_name)
            if expected_name is not None
            else None
        )

        for rank, index in enumerate(top_indices, start=1):
            marker = ""
            dataset_name = (our_scientific_names[index])

            if (dataset_name is not None and expected_normalised is not None and normalise_name(dataset_name) == expected_normalised):
                marker = " < expected"

            print(
                f"{rank:2d}. "
                f"{bird_classes[index]:<12} "
                f"{str(dataset_name):<40} "
                f"{accumulated[index]:.3%}"
                f"{marker}"
            )

        if ((i + 1) % CHECKPOINT == 0):
            np.save(CHECKPOINT_MATRIX, teacher_matrix)
            np.save(CHECKPOINT_INDEX, np.array(i + 1))
            print(f"\nCheckpoint saved at {i + 1}/{num_classes}")

    np.save(OUTPUT_DIR / "teacher_probs.npy", teacher_matrix)
    np.save(OUTPUT_DIR / "class_order.npy", np.array(bird_classes))
    max_probabilities = (teacher_matrix.max(axis=1))

    print(
        f"\nTeacher generation complete"
        f"\nClasses: {num_classes}, files: {successful_files}/{total_files}W"
        f"\nExpected: {expected_found_count}/{successful_files}"
        f"Matrix: {teacher_matrix.shape}"
        f"\nAvg max: {max_probabilities.mean():.3%} | Min max: {max_probabilities.min():.3%} | Max max: {max_probabilities.max():.3%}"
    )

    print(f"Saved: {OUTPUT_DIR / 'teacher_probs.npy'}, saved: {OUTPUT_DIR / 'class_order.npy'}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()