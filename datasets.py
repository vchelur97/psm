# Assuming that preprocessing has been completed from preprocessing folder
# We define a dataset based on the preprocessed data
import os
from collections import defaultdict
from utils import load_pickle, create_discretized_spectrum

from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset


class PSMDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.num_cpus = hparams.num_cpus
        self.batch_size = hparams.batch_size
        self.fold = hparams.fold
        self.input_size = None
        if hparams.load_train_ds:
            self.train_ds = MSV000083508(hparams)
            self.input_size = self.train_ds.input_size
        if hparams.run_tests:
            self.test_ds = MSV000083508(hparams, test=True)
            self.input_size = self.test_ds.input_size
        if not self.input_size:
            self.input_size = hparams.input_size

    def train_dataloader(self):
        return DataLoader(
            Subset(self.train_ds, self.train_ds.train_indices[self.fold]),
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.train_ds, self.train_ds.valid_indices[self.fold]),
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
            pin_memory=True,
            shuffle=False,
        )

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--num-cpus",
            default=8,
            type=int,
            help="Number of CPUs for dataloader. Default: %(default)f",
        )
        parser.add_argument(
            "--no-train-ds",
            dest="load_train_ds",
            action="store_false",
            help="Use this during evaluation mode to not load the train dataset. Default: %(default)s",
        )
        parser.set_defaults(load_train_ds=True)
        return parser


class MSV000083508(Dataset):
    def __init__(self, hparams, test=False):
        super().__init__()
        self.hparams = hparams
        self.test = test
        if test:
            self.dataset_dir = os.path.join(hparams.data_dir, "test")
        else:
            self.dataset_dir = os.path.join(hparams.data_dir, "train")
            self.fold = str(hparams.fold)

        self.annotations_info = load_pickle(os.path.join(self.dataset_dir, "annotations.pkl"))
        # self.meta_info = load_pickle(os.path.join(self.dataset_dir, "meta.pkl"))
        # self.spectrum_info = load_pickle(os.path.join(self.dataset_dir, "spectrum.pkl"))
        # self.theospec_info = load_pickle(os.path.join(self.dataset_dir, "theospec.pkl"))
        self.dataset = [
            [mzml, scan_num, idx, True] if idx == 0 else [mzml, scan_num, idx, False]
            for mzml in self.annotations_info
            for scan_num in self.annotations_info[mzml]
            for idx in range(len(self.annotations_info[mzml][scan_num]["peptides"]))
        ]

        # Folds for cross-validation
        if not test:
            self.train_indices = []
            self.valid_indices = []
            for i, j in KFold(n_splits=5).split(self.dataset):  # type: ignore
                self.train_indices.append(i)
                self.valid_indices.append(j)
            # indices = list(range(len(self.dataset)))
            # # Put 80% of the data in the training set and 20% in the validation set
            # self.train_indices, self.valid_indices = train_test_split(
            #     indices, test_size=0.2, shuffle=False
            # )
            # self.train_indices = [self.train_indices]
            # self.valid_indices = [self.valid_indices]

        self.input_size = self[0]["feature"].shape[0]

    def __getitem__(self, index):
        mzml, scan_num, idx, label = self.dataset[index]
        # meta_info = self.meta_info[mzml][scan_num].iloc[idx]
        discretized_spec = create_discretized_spectrum(
            self.annotations_info[mzml][scan_num]["mz_arr"],
            self.annotations_info[mzml][scan_num]["intensity_arr"],
            self.annotations_info[mzml][scan_num]["annotations_arr"][idx]
        )
        data = {
            "feature": np.array(discretized_spec).astype(np.float32),
            "label": label,
        }
        return data

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--data-dir",
            default="data/",
            type=str,
            help="Location of data directory. Default: %(default)s",
        )
        parser.add_argument(
            "--fold",
            metavar="NUMBER",
            type=int,
            default=0,
            help="Cross Validation fold number to train on. Default: %(default)d",
        )
        return parser
