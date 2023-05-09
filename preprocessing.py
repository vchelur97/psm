from utils import download_url, unzip_file, NUM_TRAIN_SPECTRA
import pandas as pd


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


def preprocess_data(dataset: str) -> pd.DataFrame:
    # Preprocessing for training and testing datasets
    # Ensure that the data is in the format expected by the model
    df = pd.read_csv(dataset,sep='\t')
    unique_scannums = df['ScanNum '].unique().tolist()
    final_indices = []
    lengths = []
    final_df = pd.DataFrame()
    for scan_nums in unique_scannums:
        temp_df = df[df['ScanNum ']==scan_nums]
        peptides = temp_df.loc[:,'Peptide                                                                     '].tolist()
        proteins = temp_df.loc[:,'Protein                            '].tolist()
        peptide_dict = {k:v for k,v in enumerate(peptides)}
        ### Considered the first match as the True Positive always
        protein_dict = {k:v for k,v in enumerate(proteins) if v[:5]!='XXX_t' and v[0]!='t'}
        peptide_dict = {k:peptide_dict[k] for k in protein_dict.keys()}
        temp_peptide_dict = {v:k for k,v in peptide_dict.items()}
        final_peptide_dict = {v:k for k,v in temp_peptide_dict.items()}

        filtered_indices = list(final_peptide_dict.keys())[:5]
        filtered_indices = [a+sum(lengths) for a in filtered_indices]
        lengths.append(len(peptide_dict))
        final_indices.extend(filtered_indices)
    final_df = df.loc[final_indices,:]
    final_df[['ScanNum ', 'Charge ', 'Peptide                                                                     ', 'Protein                            ', 'DeNovoScore ', 'MSGFScore ', 'SpecEValue    ', 'EValue        ', 'OriginalFilepath']]

    return final_df


if __name__ == "__main__":
    download_extract_dataset()
    create_gensim_embeddings_for_peptides()
    preprocess_data("train")
    preprocess_data("test")
