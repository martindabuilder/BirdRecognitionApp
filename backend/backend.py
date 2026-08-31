import numpy as np
import librosa
import torchvision.models as models
import torch.nn as nn
import torch
from pathlib import Path
import cv2
import base64
import matplotlib.pyplot as plt
from io import BytesIO
import csv

#---- FastApi related imports ----
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


#---- Model related paths and data ----
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "best_model.pth"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder_classes.npy"
BIRD_CSV_PATH = BASE_DIR.parent / "dataset" / "eBird_taxonomy_v2025-4.csv"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

#settings that must match the same used beforehand during the preprocessing
MEL_LINES = 150 #spectrogram height, measured in n_mels
SAMPLE_RATE = 32000 #set sample rate for each spectrogram
AUDIO_DURATION = 5.0 #seconds per segment

MAX_DARKNESS = 0.15 #intensity below .05 means the segment gets skipped(i.e its too quiet/empty)
MINIMAL_ACTIVE_THRESHOLD = 0.30 #the amount of "active" volume in the file

SEGMENT_SAMPLES = int(AUDIO_DURATION * SAMPLE_RATE)
HOP_LENGTH = int(0.020 * SAMPLE_RATE)

TARGET_SIZE = (224, 224)

DEVICE = torch.device("cpu")


#---- Functions related to visual presentation in the frontend ----
#Saves and pushes the spectrogram segments for the predicted file
def spectrogram_to_base64(spectrogram):
    buffer = BytesIO()
    plt.figure(figsize = (10, 4))
    plt.imshow(spectrogram, aspect="auto", origin="lower")
    plt.axis("off")
    plt.savefig(buffer, format = "png", bbox_inches = "tight", pad_inches = 0)
    plt.close()
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")

#Loads the actual bird name against its scientific code
def load_bird_info():
    bird_info = {}

    with open(BIRD_CSV_PATH, newline = "", encoding = "utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            code = row["SPECIES_CODE"]
            bird_info[code] = {"common_name" : row["PRIMARY_COM_NAME"], "scientific_name" : row["SCI_NAME"]}

    return bird_info

BIRD_INFO = load_bird_info()


#---- Filtering functions ----
#checks if the segment is active enough
#if it isnt above the threshold it gets skipped
def is_seg_active(spectrogram, threshold = 0.05, min_ratio = MINIMAL_ACTIVE_THRESHOLD):
    active_columns = np.any(spectrogram > threshold, axis = 0)
    ratio = np.sum(active_columns) / len(active_columns)
    return ratio >= min_ratio


#checks for the overall darkness of the spectrogram, if its too dark it gets skipped
def spectrogram_too_dark(spectrogram):
    return spectrogram.mean() < MAX_DARKNESS


def prepare_spectrogram(spectrogram):
    spectrogram = cv2.resize(spectrogram.astype(np.float32), (TARGET_SIZE[1], TARGET_SIZE[0]),interpolation=cv2.INTER_LINEAR)
    spectrogram = torch.from_numpy(spectrogram)
    spectrogram = spectrogram.unsqueeze(0).repeat(3, 1, 1)
    spectrogram = (spectrogram - IMAGENET_MEAN) / IMAGENET_STD

    return spectrogram


#---- Audio processing related functions ----
#preprocessing, which is done in the same way as preprocessing.py
def split_audio(file_path, sr = SAMPLE_RATE, duration = AUDIO_DURATION):
    y, sr = librosa.load(file_path, sr = sr, res_type = "soxr_hq")

    y, _ = librosa.effects.trim(y)
    segment_length = int(duration * sr)
    hop_samples = int(4.0 * sr) #allows a 1 second overlap between 2 consecutive segments
    segments = []

    for start in range(0, len(y), hop_samples):
        end = start + segment_length
        seg = y[start:end]
        if len(seg) < segment_length:
            seg = np.pad(seg,(0, segment_length - len(seg)))
        segments.append(seg)

    return segments, sr


#turns the 5s audio segments into a normalized mel spectrogram
def audio_to_spectrogram(y, sr):
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 2048, 
    hop_length = HOP_LENGTH, n_mels = MEL_LINES, fmin = 200, fmax = 16000, power = 2.0)

    spectrogram_db = librosa.power_to_db(spectrogram, ref = np.max, top_db=80)
    spectrogram_norm = (spectrogram_db + 80.0) / 80.0
    spectrogram_norm = np.clip( spectrogram_norm, 0.0, 1.0)
    return spectrogram_norm.astype(np.float32)



#---- Model loading and performance functions
#Loads the pre-trained model with our custom weights
def load_model():
    classes = np.load(LABEL_ENCODER_PATH, allow_pickle = True)
    num_classes = len(classes)

    model = models.efficientnet_b0(weights = None)
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE, weights_only = True))
    model.to(DEVICE)
    model.eval()

    return model, classes


#Once the backend starts the model gets automatically loaded
model, classes = load_model()


#Used to predict the file uploaded by the user, regardless of if its from device or recording
def predict_uploaded_file(file_path):
    segments, sr = split_audio(file_path)
    inputs = []
    spectrograms = []

    for segment in segments:
        spectrogram = audio_to_spectrogram(segment, sr)

        if spectrogram_too_dark(spectrogram):
            continue

        if not is_seg_active(spectrogram):
            continue

        inputs.append(prepare_spectrogram(spectrogram))
        spectrograms.append(spectrogram)

    if not inputs:
        raise ValueError("No usable audio segments were generated.")

    batch = torch.stack(inputs).to(DEVICE)

    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.softmax(outputs, dim = 1)

    mean_probs = probabilities.mean(dim = 0)

    top_picks = min(5, len(classes))
    values, indices = torch.topk(mean_probs, top_picks)

    predictions = []

    for value, index in zip(values, indices):
        code = str(classes[index.item()])
        bird = BIRD_INFO.get(code)

        predictions.append({
            "species": bird["common_name"] if bird else code,
            "scientificName": bird["scientific_name"] if bird else None,
            "label": code,
            "probabilities": float(value.item())
        })

    return predictions, spectrograms



#---- FastAPI portion of the backend ----
app = FastAPI(title = "Bird Recognition")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#Main portion of the site
@app.get("/")
def home():
    return {"message" : "Bird Recognition FastAPI endpoint is running"}


#Receives an audio file and runs it through the model
@app.post("/predict")
async def results(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    temp_file = BASE_DIR / f"temp_{file.filename}"

    try:
        contents = await file.read()
        with open(temp_file, "wb") as f:
            f.write(contents)

        predictions, spectrograms = predict_uploaded_file(temp_file)
        spectrogram_images = []

        for spectrogram in spectrograms:
            image = spectrogram_to_base64(spectrogram)

            spectrogram_images.append(image)

        audio_base64 = base64.b64encode(contents).decode("utf-8")

        return {
            "filename": file.filename,
            "predictions": predictions,
            "spectrograms": spectrogram_images,
            "audio": audio_base64,
            "audioType": file.content_type
        }

    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Prediction failed: {str(e)}")

    finally:
        if temp_file.exists():
            temp_file.unlink()

# List of all the available birds in the project
@app.get("/birds")
def get_birds():
    birds = []

    for bird_class in classes:
        code = str(bird_class)
        bird = BIRD_INFO.get(code)

        if bird:
            birds.append({
                "label": code,
                "commonName": bird["common_name"],
                "scientificName": bird["scientific_name"]
            })

    return birds