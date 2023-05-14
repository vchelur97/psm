from utils import download_url, unzip_file, NUM_TRAIN_SPECTRA
from pyteomics import mzml, mzxml
import numpy as np


def download_extract_dataset():
    url = "http://proteomics.ucsd.edu/data/cse291_2022/lung_top20_dcf82dfcd2b8456b800d07e682d494b4.zip"
    download_url("raw", url)
    unzip_file("raw", url.split("/")[-1])


def download_mzml_file(mzml_name):
    url = f"ftp://massive.ucsd.edu/{mzml_name}"
    download_url("raw/mzml", url)

    file_path = "./data/raw/mzml/" + mzml_name
    reader = mzml.read(file_path)

    # Iterate over the spectra in the file
    for spectrum in reader:
        # Get the m/z and intensity arrays
        mz_array = np.array(spectrum["m/z array"])
        intensity_array = np.array(spectrum["intensity array"])

        # Process the data as needed
        # For example, you can print the m/z and intensity values
        # print("m/z values:", mz_array)
        # print("Intensity values:", intensity_array)


def create_gensim_embeddings_for_peptides():
    # Get NUM_TRAIN_SPECTRA from dataset
    # Get all the peptides in the dataset and corresponding protein IDs
    # Download protein sequences from Uniref
    # Train gensim embeddings on the above data
    # Store the peptide embeddings in a file
    pass


def preprocess_data(dataset):
    # Preprocessing for training and testing datasets
    # Ensure that the data is in the format expected by the model
    pass


if __name__ == "__main__":
    # download_extract_dataset()
    # create_gensim_embeddings_for_peptides()
    # preprocess_data("train")
    # preprocess_data("test")
    download_mzml_file(
        "MSV000083508/ccms_peak/lung/Trypsin_HCD_QExactiveplus/01088_A05_P010740_S00_N33_R1.mzML"
    )
