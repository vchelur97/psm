import os
import pickle
import subprocess
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyteomics import mzml
from tqdm import tqdm
from utils import (
    MASS_SHIFT,
    MASS_SHIFT_DICT,
    MASS_SHIFT_CMD,
    NUM_SPECTRA,
    download_url,
    unzip_file,
)


def download_extract_dataset(url, raw_dir, file_name):
    if hparams.on_sample:
        return
    download_url(raw_dir, url)
    # unzip_file(os.path.join(raw_dir, url.split("/")[-1]), file_name)


def download_mzml_file(mzml_dir, mzml_location):
    url = f"ftp://massive.ucsd.edu/{mzml_location}"
    download_url(mzml_dir, url)


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
        print("Extracting spectrums for", mzml_name)
        for scan_num in tqdm(scan_nums):
            spectrum = reader.get_by_id(
                f"controllerType=0 controllerNumber=1 scan={scan_num}", element_type="spectrum"
            )
            spectrums[scan_num] = {
                "mz_arr": spectrum["m/z array"],
                "intensity_arr": spectrum["intensity array"],
            }
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
    print("Saved spectrum info to", spectrum_info_path)


def extract_meta_info(meta_info_path, tsv_file, nrows):
    df = pd.read_csv(tsv_file, sep="\t", nrows=nrows)
    # Let's not do this considering we aren't using gensim embeddings
    # df = df[df["Protein"].str.startswith("XXX_sp") | df["Protein"].str.startswith("sp")]
    meta_info = defaultdict(dict)
    for mzml_file_location, scan_nums in (
        df.groupby("OriginalFilepath")["ScanNum"].unique().apply(list).to_dict().items()
    ):
        mzml_file_name = mzml_file_location.split("/")[-1]
        meta_info[mzml_file_name] = {}
        print("Extracting meta info for", mzml_file_name)
        for scan_num in tqdm(scan_nums):
            # Let's drop duplicate peptides in a scan
            scan_df = df[df["ScanNum"] == scan_num].drop_duplicates(subset="Peptide")
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
            scan_df["Label"] = False
            scan_df.iat[0, -1] = True
            scan_df = scan_df.iloc[[0] + list(range(5, min(len(scan_df), 14)))]
            meta_info[mzml_file_name][scan_num] = scan_df

    with open(meta_info_path, "wb") as f:
        pickle.dump(meta_info, f)
    print("Saved meta info to", meta_info_path)


def run_theospec_command(charge, sequence, n_charge=0.0):
    if n_charge != 0.0:
        n_charge_cmd = [f"-N{n_charge}"]
    else:
        n_charge_cmd = []
    cmd = ["./theospec", "-iabcxyz", f"-z{charge}"] + n_charge_cmd + MASS_SHIFT_CMD + [sequence]
    # print(" ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
        )
        output = result.stdout.decode("utf-8").split("\n")
        output = [el.strip().split(";") for el in output[19:-2]]
        return output
    except subprocess.CalledProcessError as e:
        print(f"theospec failed with error code {e.returncode}")
        exit(1)
    except FileNotFoundError:
        print("theospec is not installed or not in the system PATH")
        exit(1)


def generate_theospec_for_peptide(charge, peptide):
    if charge > 5:
        charge = 5
    for key, val in MASS_SHIFT_DICT.items():
        peptide = peptide.replace(key, str(val))
    i = 2
    n_charge = 0.0
    while peptide[i].isdigit():
        n_charge += float(MASS_SHIFT[int(peptide[i])])
        i += 1
    if n_charge != 0:
        peptide = peptide[:2] + peptide[i:]
    # If the peptide starts with "X", we need to change it for theospec. Replacing with most occurring amino acid
    if peptide.startswith("X"):
        peptide = "K" + peptide[1:]
    if peptide.endswith("X"):
        peptide = peptide[:-1] + "L"
    output = run_theospec_command(charge, peptide, n_charge)
    mz_arr = [float(el[0].split("(")[0]) for el in output]
    charge_arr = [int(el[1]) for el in output]
    ion_name_arr = [el[2] + el[4] for el in output]
    intensity_arr = [1] * len(mz_arr)
    return {
        "mz_arr": mz_arr,
        "charge_arr": charge_arr,
        "ion_name_arr": ion_name_arr,
        "intensity_arr": intensity_arr,
    }


def generate_theospec(theospec_path, meta_info_path):
    meta_info = pickle.load(open(meta_info_path, "rb"))
    theospec = {}
    for mzml_file_name, scan_info in meta_info.items():
        print("Generating theospec for", mzml_file_name)
        for scan_num, scan_df in tqdm(scan_info.items()):
            for charge, peptide in scan_df[["Charge", "Peptide"]].values:
                if peptide in theospec:
                    continue
                raw = generate_theospec_for_peptide(charge, peptide)
                theospec[peptide] = raw

    with open(theospec_path, "wb") as f:
        pickle.dump(theospec, f)
    print("Saved theospec to", theospec_path)


def preprocess_data(type: str):
    if type == "train":
        data_dir = hparams.train_dir
        # dataset_url = "http://proteomics.ucsd.edu/data/cse291_2022/lung_top20_dcf82dfcd2b8456b800d07e682d494b4.zip"
        # dataset_url = "https://www.dropbox.com/s/tplshpf10pxoed0/train.tsv?dl=1"
        dataset_url = "train.tsv"
    else:
        data_dir = hparams.test_dir
        # dataset_url = "http://proteomics.ucsd.edu/data/cse291_2022/colon_top20_87bb3840244542918c777f63352a6115.zip"
        # dataset_url = "https://www.dropbox.com/s/enjuvpe3enuz7hk/test.tsv?dl=1"
        dataset_url = "test.tsv"
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
