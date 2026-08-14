import os
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES_PER_CLASS = 5
RANDOM_SEED = 42
N_WORKERS = 1
BATCH_SIZE = 1
TOP_K = 10


def normalise_scientific_name(name):
    if pd.isna(name):
        return None

    name = str(name).strip()

    if not name:
        return None

    return " ".join(name.split())


def build_code_to_scientific(taxonomy_df):
    return dict( zip(taxonomy_df["SPECIES_CODE"],taxonomy_df["SCI_NAME"] ))


def get_birdnet_species_probs(model, file_path):
    try:
        result = model.predict(
            str(file_path),
            top_k=TOP_K,
            n_workers=N_WORKERS,
            n_producers=1,
            batch_size=BATCH_SIZE,
            device="CPU",
            show_stats="minimal"
        )
    except Exception as e:
        print(f"BirdNET failed on {file_path.name}: {type(e).__name__}: {e}")
        return {}

    species_list = np.asarray(result.species_list)
    species_probs = np.asarray(result.species_probs)
    species_ids = np.asarray(result.species_ids)
    species_masked = np.asarray(result.species_masked)

    if species_probs.ndim == 3:
        species_probs = species_probs[0]

    if species_ids.ndim == 3:
        species_ids = species_ids[0]

    if species_masked.ndim == 3:
        species_masked = species_masked[0]

    accumulated = {}
    counts = {}

    for segment_index in range(species_probs.shape[0]):
        for prediction_index in range(species_probs.shape[1]):
            if species_masked[segment_index, prediction_index]:
                continue

            species_id = int(species_ids[segment_index, prediction_index] )

            if species_id < 0 or species_id >= len(species_list):
                continue

            probability = float(species_probs[segment_index, prediction_index])

            if not np.isfinite(probability) or probability < 0:
                continue

            species_label = str(species_list[species_id])

            if "_" in species_label:
                scientific_name = species_label.split("_", 1)[0].strip()
            else:
                scientific_name = species_label.strip()

            if not scientific_name:
                continue

            accumulated[scientific_name] = (accumulated.get(scientific_name, 0.0) + probability)
            counts[scientific_name] = (counts.get(scientific_name, 0) + 1 )

    if not accumulated:
        return {}

    averaged = {
        name: accumulated[name] / counts[name]
        for name in accumulated
    }

    total = sum(averaged.values())

    if total <= 0:
        return {}

    return {
        name: probability / total
        for name, probability in averaged.items()
    }


def print_top_predictions(predictions, expected_name=None, n=10):
    if not predictions:
        print("No predictions.")
        return

    sorted_predictions = sorted(predictions.items(),key=lambda x: x[1], reverse=True)

    for rank, (name, probability) in enumerate( sorted_predictions[:n], start=1):
        marker = ""

        if expected_name is not None and name == expected_name:
            marker = " <-- EXPECTED"

        print(f"{rank:2d}. {name:<45} {probability:8.3%}{marker}" )

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Loading BirdNET 2.4...")

    birdnet_model = load( "acoustic", "2.4", "tf")

    print("BirdNET loaded.")

    taxonomy_df = pd.read_csv(TAXONOMY_PATH)
    code_to_sci = build_code_to_scientific(taxonomy_df)

    bird_classes = sorted(
        d for d in os.listdir(DATA_DIRECTORY)
        if os.path.isdir(DATA_DIRECTORY / d)
        and d != "birdnet_teacher"
    )

    num_classes = len(bird_classes)

    our_scientific_names = []

    for code in bird_classes:
        scientific_name = normalise_scientific_name(code_to_sci.get(code))
        our_scientific_names.append(scientific_name)
        print(f"{code} -> {scientific_name}")

    teacher_matrix = np.zeros((num_classes, num_classes),dtype=np.float32)

    total_files = 0
    successful_files = 0
    expected_found_count = 0

    for i, code in enumerate(bird_classes):
        class_folder = DATA_DIRECTORY / code

        audio_files = (
            list(class_folder.glob("*.mp3")) +
            list(class_folder.glob("*.MP3")) +
            list(class_folder.glob("*.wav")) +
            list(class_folder.glob("*.WAV"))
        )

        if not audio_files:
            print(f"{code}: no audio files found")
            teacher_matrix[i] = 1.0 / num_classes
            continue

        sample_files = random.sample(audio_files min(FILES_PER_CLASS, len(audio_files)))

        true_scientific_name = our_scientific_names[i]

        print( f"\nClass {i + 1}/{num_classes}: {code} -> {true_scientific_name}")

        accumulated = np.zeros(
            num_classes,
            dtype=np.float32
        )

        successful = 0

        for file_path in sample_files:
            total_files += 1

            print(f"  Processing: {file_path.name}")

            predictions = get_birdnet_species_probs(birdnet_model,file_path)

            if not predictions:
                print("No usable prediction.")
                continue

            successful += 1
            successful_files += 1
            print_top_predictions(predictions, expected_name=true_scientific_name)

            if true_scientific_name in predictions:
                expected_found_count += 1

            for j, scientific_name in enumerate( our_scientific_names):
                if scientific_name is not None:
                    accumulated[j] += predictions.get(scientific_name, 0.0 )

        if successful > 0:
            accumulated /= successful

            total = accumulated.sum()

            if total > 0:
                accumulated /= total
            else:
                accumulated[:] = 1.0 / num_classes
        else:
            accumulated[:] = 1.0 / num_classes

        teacher_matrix[i] = accumulated

        top_indices = np.argsort(accumulated)[::-1][:10]

        print("Top dataset classes:")

        for rank, index in enumerate(top_indices, start=1):
            print(f"{rank:2d}. {bird_classes[index]:<12} {our_scientific_names[index]:<40} {accumulated[index]:.3%}" )

    np.save(OUTPUT_DIR / "teacher_probs.npy",teacher_matrix )
    np.save(OUTPUT_DIR / "class_order.npy",np.array(bird_classes))
    max_probabilities = teacher_matrix.max(axis=1)

    print(
    f"\nTeacher generation complete | "
    f"Classes: {num_classes} | "
    f"Files: {successful_files}/{total_files} | "
    f"Expected: {expected_found_count}/{successful_files} | "
    f"Matrix: {teacher_matrix.shape} | "
    f"Avg max: {max_probabilities.mean():.3%} | "
    f"Min max: {max_probabilities.min():.3%} | "
    f"Max max: {max_probabilities.max():.3%}")
    
    print(f"Saved: {OUTPUT_DIR / 'teacher_probs.npy'}")
    print(f"Saved: {OUTPUT_DIR / 'class_order.npy'}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()