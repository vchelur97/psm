import os
import subprocess
import zipfile


NUM_TRAIN_SPECTRA = 10000
NUM_TEST_SPECTRA = 1000

def download_url(dir, url):
    try:
        subprocess.run(["aria2c", "-x", "16", "-c", "-d", f"data/{dir}", url], check=True)
    except subprocess.CalledProcessError as e:
        print(f"aria2c failed with error code {e.returncode}")
    except FileNotFoundError:
        print("aria2c is not installed or not in the system PATH")


def unzip_file(dir: str, file_name: str) -> None:
    # If the extraction has already been done, return
    if os.path.exists(f"data/{dir}/{file_name.split('.')[0]}"):
        return

    # Open the zip file in read mode
    with zipfile.ZipFile(f"data/{dir}/{file_name}", "r") as zip_ref:
        # Extract all files to the "data" directory
        zip_ref.extractall(f"data/{dir}/{file_name.split('.')[0]}")
