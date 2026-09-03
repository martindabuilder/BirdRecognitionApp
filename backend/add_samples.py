# Additional file, used to download more audio files to a specific class.

# Libraries and imports.

import os
import sys
import time
import glob
import re
import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TAXONOMY_PATH = os.path.join(DATASET_DIR, "eBird_taxonomy_v2025-4.csv")

# Only download recordings with these quality ratings.
MIN_QUALITY = {"A", "B"}

# Max new recordings to download per species each time the script is run.
MAX_NEW_RECORDINGS = 45

# Maximum number of pages searched for a species.
MAX_PAGES = 20

# Delay between requests
REQUEST_DELAY_SECONDS = 1.0

# API key and xeno canto url endpoint
XC_API_URL = ("https://xeno-canto.org/api/3/recordings")
XC_API_KEY = " --- CUSTOM API KEY HERE --- "


# Maps the dataset's species codes to their scientific names when searching.
def lookup_scientific_name(code,taxonomy_df):
    row = taxonomy_df[taxonomy_df["SPECIES_CODE"] == code]
    if row.empty:
        return None
    return row.iloc[0]["SCI_NAME"]


# Finds any already downloaded recordings so the script doesn't duplicate any recordings.
def get_existing_xc_ids(species_dir):
    existing_ids = set()
    
    if not os.path.isdir(species_dir):
        return existing_ids

    for path in glob.glob(os.path.join(species_dir, "*")):
        filename = os.path.basename(path)
        match = re.search(r"XC(\d+)",filename)

        if match:
            existing_ids.add(match.group(1))
    return existing_ids


# Searches Xeno-canto for recordings of the requested species.
# Keeps only the newly found recordings.
def query_species(scientific_name, existing_ids, target_count):
    genus, _, species = (scientific_name.partition(" "))
    query = (f"gen:{genus} sp:{species}")
    results = []
    page = 1

    while True:
        print(f"Querying Xeno-canto page {page}...")
        response = requests.get(XC_API_URL, params = {"query": query, "page": page, "key": XC_API_KEY}, timeout = 30)
        response.raise_for_status()
        data = response.json()
        
        page_recordings = data.get("recordings", [])
        new_qualifying = [
            recording
            for recording in page_recordings
            if recording.get("q") in MIN_QUALITY
            and recording.get("id") not in existing_ids
        ]
        results.extend(new_qualifying)

        num_pages = int(data.get("numPages", 1))

        if len(results) >= target_count:
            print(f"Found {len(results)} new qualifying recordings, stopping early.")
            break

        if page >= num_pages:
            break

        if page >= MAX_PAGES:
            print(f"Reached MAX_PAGES ({MAX_PAGES}), stopping early.")
            break

        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    return results[:target_count]


# Downloads individual recordings and saves them using Xeno-canto ID.
def download_recording(recording, species_dir):
    file_id = recording["id"]
    file_url = recording["file"]
    if file_url.startswith("//"):
        file_url = ("https:" + file_url)

    extension = (
        os.path.splitext(file_url)[1]
        or ".mp3"
    )

    output_path = os.path.join(species_dir, f"XC{file_id}{extension}")
    response = requests.get(file_url, stream = True, timeout = 60)
    response.raise_for_status()

    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size = 8192):
            if chunk:
                file.write(chunk)
    return output_path


def process_code(code, taxonomy_df):
    print(f"\nProcessing: {code}")
    scientific_name = lookup_scientific_name(code,taxonomy_df )

    if scientific_name is None:
        print(f"Code '{code}' was not found in the taxonomy.")
        return

    common_name_row = (taxonomy_df[taxonomy_df["SPECIES_CODE"] == code].iloc[0])

    common_name = (common_name_row["PRIMARY_COM_NAME"])
    print(f"Species: {common_name}, Scientific name: {scientific_name}")

    species_dir = os.path.join( DATASET_DIR, code )
    os.makedirs( species_dir,  exist_ok=True)

    print(f"Destination: {species_dir}")
    existing_ids = get_existing_xc_ids(species_dir)
    print(f"Already have {len(existing_ids)} recording/s.")

    try:
        recordings = query_species(scientific_name, existing_ids,MAX_NEW_RECORDINGS)

    except requests.RequestException as error:
        print(f"Failed to query Xeno-canto: {error}")
        return

    print(f"Found {len(recordings)} recordings.")

    if not recordings:
        print("No new suitable recordings found.")
        return

    print(f"Downloading {len(recordings)} new recordings...")

    downloaded = 0
    for recording in recordings:
        recording_id = recording["id"]
        try:
            output_path = download_recording(recording, species_dir)
            downloaded += 1
            print(f"Downloaded XC{recording_id}")

        except requests.RequestException as error:
            print(f"Failed to download XC{recording_id}: {error}")
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n{downloaded} new recordings added, location: {species_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("py download_more_audio.py <code1> [code2] [code3] ...")
        return

    if not os.path.isfile(TAXONOMY_PATH):
        print("ERROR: Taxonomy CSV was not found:")
        print(TAXONOMY_PATH)
        return

    print(f"Dataset directory:")
    print(DATASET_DIR)
    print()

    # Reads the taxonomy once, and uses it for all the requested species.
    taxonomy_df = pd.read_csv(TAXONOMY_PATH)
    codes = sys.argv[1:]

    for code in codes:
        process_code(code, taxonomy_df)

    print(f"\nAll downloads complete.")
  
if __name__ == "__main__":
    main()