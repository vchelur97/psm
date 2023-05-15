import os
import subprocess
import zipfile


NUM_SPECTRA = {"train": 500000, "test": 500000}
MAX_MASS = 5000
MASS_SHIFT = ["+0.984", "+42.011", "+15.995", "-17.027", "+43.006", "+57.021"]
MASS_SHIFT_DICT = {k: v for k, v in zip(MASS_SHIFT, range(len(MASS_SHIFT)))}
MASS_SHIFT_CMD = [f"-c{val}={key}" for key, val in MASS_SHIFT_DICT.items()]


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
