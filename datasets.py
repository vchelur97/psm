# Assuming that preprocessing has been completed from preprocessing folder
# We define a dataset based on the preprocessed data
import os
from collections import defaultdict
from glob import glob

import numpy as np
import pytorch_lightning as pl
from torch import from_numpy
from torch.utils.data import DataLoader, Dataset, Subset

AMINO_ACIDS = "XACDEFGHIKLMNPQRSTVWY"
AA_DICT = defaultdict(lambda: 0, {aa: idx for idx, aa in enumerate(AMINO_ACIDS)})


class PSMDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.num_cpus = hparams.num_cpus
        self.batch_size = hparams.batch_size
        if hparams.gpus != 0:
            self.pin_memory = True
        else:
            self.pin_memory = False
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
            Subset(self.train_ds, self.train_ds.train_indices),
            batch_size=self.batch_size,
            collate_fn=MSV000083508.collate_fn,
            num_workers=self.num_cpus,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.train_ds, self.train_ds.valid_indices),
            batch_size=self.batch_size,
            collate_fn=MSV000083508.collate_fn,
            num_workers=self.num_cpus,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=MSV000083508.collate_fn,
            num_workers=self.num_cpus,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--num-cpus",
            default=10,
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
        self.raw_dir = os.path.join(self.dataset_dir, "raw")

        # TODO: Load features and labels here

        # ------------MAPPINGS------------
        # TODO: If there are any mappings to be done, do them here

        # Folds for cross-validation
        self.train_fold, self.valid_fold = self.get_fold()

        # TODO: Get input size properly based on the data
        self.input_size = self[0][0]["feature"].shape[0]

    def get_npy(self, name, flag=True):
        # TODO: Might have to change this function based on how we store the data
        if not flag:
            return None
        mapping = {}
        if self.test:
            print("Loading", name, "of test set")
        else:
            print("Loading", name, "of train set")
        tmp = glob(os.path.join(self.preprocessed_dir, "*", name + "_?.npy"))
        if tmp == []:
            tmp = glob(os.path.join(self.msa_dir, "*", name + "_?.npy"))
        if tmp == []:
            tmp = glob(os.path.join(self.raw_dir, "*", name + "_?.npy"))
        tmp = sorted(tmp)
        for file in tmp:
            pis, chain = file.split("/")[-2:]
            chain = chain[-5:-4]
            mapping[pis + "/" + chain] = np.load(file).astype(np.float32)
        return mapping

    def get_fold(self):
        # TODO: Create cross-validation folds here
        return [], []

    def __getitem__(self, index):
        # TODO: Return a single sample data from the dataset
        # In this case, keep it as 2 dictionaries, first one has data with 2 keys,
        # feature and label, second dict has meta information which may be used
        return {}, {}

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    # A collate function to merge samples into a minibatch, will be used by DataLoader
    def collate_fn(samples):
        # TODO
        pass

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--data-dir",
            default="../data/",
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
