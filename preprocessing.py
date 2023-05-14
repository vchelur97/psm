from utils import download_url, unzip_file, NUM_TRAIN_SPECTRA
from pyteomics import mzml
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

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

def read_mzml_file(mzml_name: str):
    specs = {}
    with mzml.read(mzml_name) as reader:
        for i,spectrum in tqdm(enumerate(reader)):
            scan_number = int(spectrum['id'].split('scan=')[1])
            discretized_peaks = defaultdict(float)
            print(len(spectrum['m/z array']))
            for mz,intensity in zip(spectrum['m/z array'],spectrum['intensity array']):
                discretized_peaks[round(mz*0.995)] += intensity
            specs[scan_number] = list(map(list,zip(*(discretized_peaks.items()))))
            if(i==10):
                break
    # print(specs)







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
    # download_extract_dataset()
    read_mzml_file('/Users/ganeshanmalhotra/Desktop/Quarter3/cse 291/psm/data/raw/mzml/01088_A05_P010740_S00_N33_R1.mzML')
    # download_mzml_file('MSV000083508/ccms_peak/lung/Trypsin_HCD_QExactiveplus/01088_A05_P010740_S00_N33_R1.mzML')
    # create_gensim_embeddings_for_peptides()
    # preprocess_data("train")
    # preprocess_data("test")
