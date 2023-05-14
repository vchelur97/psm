import os
import subprocess
import zipfile


NUM_SPECTRA = {"train": 100000, "test": 10000}


def download_url(dir, url):
    try:
        subprocess.run(["aria2c", "-x", "16", "-c", "-d", f"{dir}", url], check=True)
    except subprocess.CalledProcessError as e:
        print(f"aria2c failed with error code {e.returncode}")
    except FileNotFoundError:
        print("aria2c is not installed or not in the system PATH")


def unzip_file(zip_file: str, unzipped_file: str) -> None:
    # If the extraction has already been done, return
    if os.path.exists(f"{unzipped_file}"):
        return

    # Open the zip file in read mode
    with zipfile.ZipFile(f"{zip_file}", "r") as zip_ref:
        # Extract all files to the "data" directory
        zip_ref.extractall(f"{unzipped_file}")
