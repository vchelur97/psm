import os
import subprocess
import numpy as np
from collections import defaultdict
import zipfile
import pickle


NUM_SPECTRA = {"train": 500000, "test": 500000}
MAX_MASS = 5000
MASS_SHIFT = ["+0.984", "+42.011", "+15.995", "-17.027", "+43.006", "+57.021"]
MASS_SHIFT_DICT = {k: v for k, v in zip(MASS_SHIFT, range(len(MASS_SHIFT)))}
MASS_SHIFT_CMD = [f"-c{val}={key}" for key, val in MASS_SHIFT_DICT.items()]


def create_discretized_spectrum(mz_array, intensity_arr):
    discretized_peaks = defaultdict(float)
    for mz, intensity in zip(mz_array, intensity_arr):
        discretized_peaks[round(mz * 0.995)] += intensity
    max_intensity = max(discretized_peaks.values())
    discretized = [0.0] * MAX_MASS
    for mass, intensity in discretized_peaks.items():
        if mass < MAX_MASS:
            discretized[mass] = intensity / max_intensity
    return np.array(discretized)


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


def load_pickle(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
