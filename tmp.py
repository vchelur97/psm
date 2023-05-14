from pyteomics import mzml

reader = mzml.PreIndexedMzML("/home/tyrion/programs/ucsd/sp23/cse291d/psm/data/train/raw/mzml/01088_A05_P010740_S00_N33_R1.mzML", use_index=True)
reader._build_index()