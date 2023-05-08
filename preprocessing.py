from utils import download_url, unzip_file, NUM_TRAIN_SPECTRA


def download_extract_dataset():
    url = "http://proteomics.ucsd.edu/data/cse291_2022/lung_top20_dcf82dfcd2b8456b800d07e682d494b4.zip"
    download_url("raw", url)
    unzip_file("raw", url.split("/")[-1])


def download_mzml_file(mzml_name: str):
    url = f"ftp://massive.ucsd.edu/{mzml_name}"
    download_url("raw/mzml", url)


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
    download_extract_dataset()
    create_gensim_embeddings_for_peptides()
    preprocess_data("train")
    preprocess_data("test")
