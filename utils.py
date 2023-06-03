import os
import subprocess
import numpy as np
from collections import defaultdict
import zipfile
import pickle


# NUM_SPECTRA = {"train": 500000, "test": 500000}
MAX_MASS = 2000
MASS_SHIFT = ["+0.984", "+42.011", "+15.995", "-17.027", "+43.006", "+57.021"]
MASS_SHIFT_DICT = {k: v for k, v in zip(MASS_SHIFT, range(len(MASS_SHIFT)))}
MASS_SHIFT_CMD = [f"-c{val}={key}" for key, val in MASS_SHIFT_DICT.items()]
DATASET_URLS = {
    "train": "https://www.dropbox.com/s/tplshpf10pxoed0/train.tsv?dl=1",
    "test": "https://www.dropbox.com/s/enjuvpe3enuz7hk/test.tsv?dl=1",
}


def create_discretized_spectrum(mz_array, intensity_arr, annotations=None):
    discretized_peaks = defaultdict(float)
    if annotations is None:
        annotations = [0] * len(mz_array)
    for mz, intensity, annotation in zip(mz_array, intensity_arr, annotations):
        if annotation == 0:
            discretized_peaks[round(mz * 0.995)] += intensity
        else:
            if annotation == "?":
                discretized_peaks[round(mz * 0.995)] += (-intensity)
            else:
                discretized_peaks[round(mz * 0.995)] += intensity
    max_intensity = max(discretized_peaks.values())
    max_intensity = max(abs(min(discretized_peaks.values())), max_intensity)
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


def download_extract(url, raw_dir, file_name):
    download_url(raw_dir, url)
    if url.endswith(".zip"):
        unzip_file(os.path.join(raw_dir, url.split("/")[-1]), file_name)


def load_pickle(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell" or shell == "Shell":
            return True  # Jupyter notebook or qtconsole or colab
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
