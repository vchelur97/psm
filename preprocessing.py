import os
import pickle
from collections import defaultdict
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyteomics import mzml
from tqdm import tqdm
from utils import NUM_SPECTRA, download_url, unzip_file


def download_extract_dataset(url, raw_dir, file_name):
    if hparams.on_sample:
        return
    download_url(raw_dir, url)
    # unzip_file(os.path.join(raw_dir, url.split("/")[-1]), file_name)


def download_mzml_file(mzml_dir, mzml_name):
    url = f"ftp://massive.ucsd.edu/{mzml_name}"
    download_url(mzml_dir, url)


def extract_spectrums(mzml_name: str, scan_nums: list[int]):
    specs = {}
    lengths = []
    X = {}
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
            discretized_peaks = defaultdict(float)
            lengths.append(len(spectrum["m/z array"]))
            for mz, intensity in zip(spectrum["m/z array"], spectrum["intensity array"]):
                discretized_peaks[round(mz * 0.995)] += intensity
                break
            specs[scan_num] = list(map(list, zip(*(discretized_peaks.items()))))
        for idx in specs.keys():
            mz_arr, intensity_arr = specs[idx]
            vec = [0] * 2000
            for i in range(len(mz_arr)):
                if mz_arr[i] < 2000:
                    vec[mz_arr[i]] = intensity_arr[i]
            max_intensity = max(intensity_arr)
            vec = [x / max_intensity for x in vec]
            X[idx] = np.array(vec)
    return X

    # print(specs)


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
    if hparams.on_sample:
        tsv_file = os.path.join(hparams.sample_dir, "abc.tsv")
    else:
        tsv_file = os.path.join(raw_dir, f"{type}.tsv")

    download_extract_dataset(dataset_url, raw_dir, tsv_file)
    df = pd.read_csv(tsv_file, sep="\t", nrows=NUM_SPECTRA[type])
    for mzml_file_name in tqdm(df["OriginalFilepath"].unique()):
        download_mzml_file(mzml_dir, mzml_file_name)

    data_dict = defaultdict(dict)
    for mzml_file_name, scan_nums in tqdm(
        df.groupby("OriginalFilepath")["ScanNum"].unique().apply(list).to_dict()
    ):
        spectrums = extract_spectrums(os.path.join(mzml_dir, mzml_file_name), scan_nums)
        final_indices = []
        lengths = []
        final_df = pd.DataFrame()
        for scan_num in scan_nums:
            temp_df = df[df["ScanNum"] == scan_num]
            peptides = temp_df.loc[:, "Peptide"].tolist()
            proteins = temp_df.loc[:, "Protein"].tolist()
            peptide_dict = {k: v for k, v in enumerate(peptides)}
            # Considering the first match as the True Positive always
            protein_dict = {
                k: v for k, v in enumerate(proteins) if v[:5] != "XXX_t" and v[0] != "t"
            }
            peptide_dict = {k: peptide_dict[k] for k in protein_dict.keys()}
            temp_peptide_dict = {v: k for k, v in peptide_dict.items()}
            final_peptide_dict = {v: k for k, v in temp_peptide_dict.items()}
            filtered_indices = list(final_peptide_dict.keys())
            filtered_indices = filtered_indices[0:1] + filtered_indices[5:14]
            filtered_indices = [a + sum(lengths) for a in filtered_indices]
            lengths.append(len(peptide_dict))
            final_indices.extend(filtered_indices)
            final_df = df.loc[final_indices, :]
            final_df = final_df[
                [
                    "ScanNum",
                    "Charge",
                    "Peptide",
                    "Protein",
                    "DeNovoScore",
                    "MSGFScore",
                    "SpecEValue",
                    "EValue",
                    "OriginalFilepath",
                ]
            ]

        scan_nums = final_df["ScanNum"]
        scan2vec = {}
        for scan_num in scan_nums:
            scan2vec[scan_num] = spectrums[scan_num]
        data_dict[mzml_file_name] = scan2vec

    spectrum_file_path = os.path.join(data_dir, "spectrum.pkl")
    if not os.path.exists(os.path.dirname(spectrum_file_path)):
        os.makedirs(os.path.dirname(spectrum_file_path))
    with open(spectrum_file_path, "wb") as f:
        pickle.dump(data_dict, f)


parser = ArgumentParser(description="Peptide Spectrum Matching Preprocessing", add_help=True)
parser.add_argument(
    "--data-dir",
    default="data/",
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
