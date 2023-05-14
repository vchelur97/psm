import os
import pickle
from collections import defaultdict
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyteomics import mzml
from tqdm import tqdm
from utils import NUM_SPECTRA, MAX_MASS, MASS_SHIFT_DICT, download_url, unzip_file


def download_extract_dataset(url, raw_dir, file_name):
    if hparams.on_sample:
        return
    download_url(raw_dir, url)
    # unzip_file(os.path.join(raw_dir, url.split("/")[-1]), file_name)


def download_mzml_file(mzml_dir, mzml_location):
    url = f"ftp://massive.ucsd.edu/{mzml_location}"
    download_url(mzml_dir, url)


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

def extract_spectrums_for_mzml(mzml_name: str, scan_nums: list[int]):
    spectrums = {}
    with mzml.PreIndexedMzML(
        mzml_name,
        read_schema=False,
        iterative=True,
        use_index=True,
        dtype=None,
        huge_tree=True,
        decode_binary=True,
    ) as reader:
        for scan_num in scan_nums:
            spectrum = reader.get_by_id(
                f"controllerType=0 controllerNumber=1 scan={scan_num}", element_type="spectrum"
            )
            spectrums[scan_num] = create_discretized_spectrum(spectrum["m/z array"], spectrum["intensity array"])
    return spectrums


def preprocess_mzmls(mzml_dir, spectrum_info_path, tsv_file, nrows):
    df = pd.read_csv(tsv_file, sep="\t", nrows=nrows)
    for mzml_file_location in df["OriginalFilepath"].unique():
        download_mzml_file(mzml_dir, mzml_file_location)

    spectrum_info = defaultdict(dict)
    for mzml_file_location, scan_nums in (
        df.groupby("OriginalFilepath")["ScanNum"].unique().apply(list).to_dict().items()
    ):
        mzml_file_name = mzml_file_location.split("/")[-1]
        scan_nums = [int(a) for a in scan_nums]
        spectrums = extract_spectrums_for_mzml(os.path.join(mzml_dir, mzml_file_name), scan_nums)
        spectrum_info[mzml_file_name] = spectrums

    with open(spectrum_info_path, "wb") as f:
        pickle.dump(spectrum_info, f)


def extract_meta_info(meta_info_path, tsv_file, nrows):
    df = pd.read_csv(tsv_file, sep="\t", nrows=nrows)
    df = df[df["Protein"].str.startswith("XXX_sp") | df["Protein"].str.startswith("sp")]
    meta_info = defaultdict(dict)
    for mzml_file_location, scan_nums in (
        df.groupby("OriginalFilepath")["ScanNum"].unique().apply(list).to_dict().items()
    ):
        mzml_file_name = mzml_file_location.split("/")[-1]
        meta_info[mzml_file_name] = {}
        for scan_num in scan_nums:
            scan_df = df[df["ScanNum"] == scan_num]
            scan_df = scan_df[
                [
                    "Charge",
                    "Peptide",
                    "Protein",
                    "DeNovoScore",
                    "MSGFScore",
                    "SpecEValue",
                    "EValue",
                ]
            ]
            scan_df["label"] = False
            scan_df["label"].iloc[0] = True
            scan_df = pd.concat([scan_df.iloc[0], scan_df.iloc[5:14]])
            meta_info[mzml_file_name][scan_num] = scan_df

    with open(meta_info_path, "wb") as f:
        pickle.dump(meta_info, f)


def generate_theospec_for_peptide(peptide):
    for key, val in MASS_SHIFT_DICT.items():
        peptide = peptide.replace(key, str(val))
    # Gen theospec


def generate_theospec(theospec_path, meta_info_path):
    meta_info = pickle.load(open(meta_info_path, "rb"))
    theospec = {}
    for mzml_file_name, scan_info in meta_info.items():
        for scan_num, scan_df in scan_info.items():
            for peptide in scan_df["Peptide"]:
                if peptide in theospec:
                    continue
                theospec[peptide] = generate_theospec_for_peptide(peptide)

    with open(theospec_path, "wb") as f:
        pickle.dump(theospec, f)

def preprocess_data(type: str):
    if type == "train":
        data_dir = hparams.train_dir
        # dataset_url = "http://proteomics.ucsd.edu/data/cse291_2022/lung_top20_dcf82dfcd2b8456b800d07e682d494b4.zip"
        dataset_url = "https://www.dropbox.com/s/tplshpf10pxoed0/train.tsv?dl=1"
    else:
        data_dir = hparams.test_dir
        # dataset_url = "http://proteomics.ucsd.edu/data/cse291_2022/colon_top20_87bb3840244542918c777f63352a6115.zip"
        dataset_url = "https://www.dropbox.com/s/enjuvpe3enuz7hk/test.tsv?dl=1"
    raw_dir = os.path.join(data_dir, "raw")
    mzml_dir = os.path.join(raw_dir, "mzml")
    nrows = NUM_SPECTRA[type]
    if hparams.on_sample:
        tsv_file = os.path.join(hparams.sample_dir, "abc.tsv")
    else:
        tsv_file = os.path.join(raw_dir, f"{type}.tsv")

    spectrum_info_path = os.path.join(data_dir, "spectrum.pkl")
    if not os.path.exists(spectrum_info_path):
        download_extract_dataset(dataset_url, raw_dir, tsv_file)
        preprocess_mzmls(mzml_dir, spectrum_info_path, tsv_file, nrows)

    meta_info_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_info_path):
        extract_meta_info(meta_info_path, tsv_file, nrows)

    theospec_path = os.path.join(data_dir, "theospec.pkl")
    if not os.path.exists(theospec_path):
        generate_theospec(theospec_path, meta_info_path)


parser = ArgumentParser(description="Peptide Spectrum Matching Preprocessing", add_help=True)
parser.add_argument(
    "--data-dir",
    default="./data/",
    type=str,
    help="Location of data directory. Default: %(default)s",
)
parser.add_argument(
    "--sample",
    dest="on_sample",
    action="store_true",
    help="Run preprocessing on a sample subset. Default: %(default)s",
)
parser.set_defaults(on_sample=False)
hparams = parser.parse_args()
hparams.data_dir = os.path.abspath(os.path.expanduser(hparams.data_dir))
hparams.train_dir = os.path.join(hparams.data_dir, "train")
hparams.test_dir = os.path.join(hparams.data_dir, "test")
hparams.sample_dir = os.path.join(hparams.data_dir, "sample")

preprocess_data("train")
preprocess_data("test")
