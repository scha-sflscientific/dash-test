# -*- coding: utf-8 -*-

import os
import sys
import copy
import logging

import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from functools import lru_cache
from datetime import datetime, timezone

import glob
from glob import iglob
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# from fsspec.utils import infer_storage_options

import warnings

warnings.filterwarnings("ignore")

from src.classes.errors import DataSetIOError

# ------------------------------------------------------------------------------#
#                                 LOGGER                                   #
# ------------------------------------------------------------------------------#

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Datasets(object):
    """
    An abstract class representing a machine learning dataset.

    Attributes:
    ---------
        path(str):
            Input data path
        dataset(df/generator,etc,etc):
            Input dataset
        training(boolen):
            Training mode if True
        feat_list(list):
            Feature White List
        target_col(str/list):
            For training only: Training target column

    Methods:
    --------
        from_path:
            Load dataset from path
        from_dataframe:
            Load dataset from dataframe
        from_folder:
            Load dataset from folder
        split:
            Split dataset into train/val/test sets
        sanity_check:
            Sanity check for input data
        consistency_check:
            Consistency check for input data
        feature_check:
            fill/drop feature column based on feature white list
    """

    def __init__(self, dataset, path="", training=False, target_col_list=[]):
        self._dataset_path = path
        self._dataset = dataset
        self._feat_list = []

        self.training = training
        self.target_col_list = target_col_list

        self.sanity_check()

    @property
    def dataset(self):
        """Return dataset"""
        return self._dataset

    @property
    def train_set(self):
        """Return train set"""
        pass

    @property
    def validation_set(self):
        """Return validation set"""
        pass

    @property
    def test_set(self):
        """Return test set"""
        pass

    @property
    def feat_list(self):
        """Return feature white list"""
        return self._feat_list

    @abstractmethod
    def add_feat_list(self, feat_list):
        """Manually add feature list"""
        raise NotImplementedError

    # @abstractmethod
    # def get_n(self):
    #     """Return number of elements in the dataset == len(self)."""
    #     raise NotImplementedError

    @abstractmethod
    def from_path(self, path):
        """Load Dataset from paths"""
        raise NotImplementedError

    @abstractmethod
    def from_folder(self, path):
        """Load Dataset from paths"""
        raise NotImplementedError

    @abstractmethod
    def from_dataframe(self, path):
        """Load Dataset from paths"""
        raise NotImplementedError

    @abstractmethod
    def sanity_check(self):
        """Sanity Check"""
        raise NotImplementedError

    @abstractmethod
    def consistency_check(self):
        """Consistency Check"""
        raise NotImplementedError

    @abstractmethod
    def split(self):
        """Consistency Check"""
        raise NotImplementedError

    @staticmethod
    def feature_check(df, feature_list):
        """Check and fill missing df columns with feature list"""

        for col in feature_list:
            if col not in df.columns:
                logger.warning("{0} column is not found in dataframe".format(col))
                df[col] = np.nan
        return df

    @staticmethod
    def get_protocol_and_path(filepath):
        """Parses filepath on protocol and path.
        Args:
            filepath: raw filepath e.g.: `gcs://bucket/test.json`.
            version: instance of ``kedro.io.core.Version`` or None.
        Returns:
            Protocol and path.
        Raises:
            DataSetError: when protocol is http(s) and version is not None.
            Note: HTTP(s) dataset doesn't support versioning.
        """
        options_dict = infer_storage_options(filepath)
        path = options_dict["path"]
        protocol = options_dict["protocol"]

        if protocol in HTTP_PROTOCOLS:

            raise DataSetError(
                "HTTP(s) DataSet doesn't support versioning. "
                "Please remove version flag from the dataset configuration."
            )
            path = path.split(PROTOCOL_DELIMITER, 1)[-1]

        return protocol, path


class PartitionedDatasets(Datasets):

    """
    [TODO]
    IO Module for partitioned dataset. The module loads all partitioned files from input path.

    Key Features:
        It can recursively load all or specific files from a given location.
        Is platform agnostic and can work with any filesystem implementation supported by fsspec including local, S3, GCS, and many more.
        Implements a lazy loading approach and does not attempt to load any partition data until a processing node explicitly requests it.

    """


class CSVDataSets(Datasets):

    """
    [TODO]
    CSVDataSet loads/saves data from/to a CSV file using an underlying
    filesystem (e.g.: local, S3, GCS). It uses pandas to handle the CSV file.

    >>> import pandas as pd
    >>>
    >>> data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5],
    >>>                      'col3': [5, 6]})
    >>>
    >>> # data_set = CSVDataSet(filepath="gcs://bucket/test.csv")
    >>> data_set = CSVDataSet(filepath="test.csv")
    >>> data_set.save(data)
    >>> reloaded = data_set.load()
    >>> assert data.equals(reloaded)

    """

    def __len__(self):
        return self._dataset.shape[0]

    @property
    def train_set(self, cls="train"):
        """Return train set"""
        return self.get_dataset(cls, training=True)

    @property
    def validation_set(self, cls="validate"):
        """Return validation set"""
        return self.get_dataset(cls, training=False)

    @property
    def test_set(self, cls="test"):
        """Return test set"""
        return self.get_dataset(cls, training=False)

    def get_dataframe(self):
        """Return private _dataset(pd.DataFrame) attribute"""
        return self._dataset

    def set_target_col_list(self, target_col_list):
        """Set up ground truth column list"""

        self.target_col_list = target_col_list

    def get_target_col_list(self):
        """return ground truth column list"""

        return self.target_col_list

    @classmethod
    def from_path(
        cls,
        path,
        index_col=None,
        training=False,
        target_col_list=[],
        credentials={},
        load_args={},
    ):
        """
        Load Data from data path

        Arguments:
        ---------
            path(str):
                Direcotry of DICOM images
            Training(boolen):
                If False, all rows are in test set

        returns:
        --------
            dataset(PETImageDataset):PET Image Datasets
                PETImageDataset class

        """

        # [TODO]Need to solve fsspec installation
        # protocol, path =cls. get_protocol_and_path(filepath)
        # _fs = fsspec.filesystem(protocol, **_credentials, **_fs_args)

        import glob

        if path[-4:] != ".csv":
            csv_list = glob.glob(path + "/*.csv")
        else:
            csv_list = glob.glob(path)

        if len(csv_list) == 0:
            raise ValueError("[PREPROCESS]Input PERSI FIles not found.")
            raise DataSetIOError("[CSVDataSets IO]: No input")

        else:
            logger.info(
                "[CSVDataSets IO]: {} csv files found in path:{}".format(
                    len(csv_list), path
                )
            )

        dfs_list = []

        for _csv in csv_list:
            dfs_list.append(pd.read_csv(_csv, **load_args))

        try:
            df = pd.concat(dfs_list)
        except Exception as e:
            raise DataSetIOError("[CSVDataSets IO]: {}".format(Exception))

        return cls(df, path=path, training=training, target_col_list=target_col_list)

    @classmethod
    def from_dataframe(cls, df, training=True, target_col_list=[]):
        """
        Load Data from dataframe

        Arguments:
        ---------
            df(pd.DataFrame):
                Input dataframe
            Training(boolen):
                If False, all rows are in test set
            target_col(str/list):
                target column name/list

        returns:
        --------
            dataset(CSVDataset):
                CSVDataset class

        """

        return cls(df, training=training, target_col_list=target_col_list)

    def merge_df(self, df, key="id"):
        """Merge dataframe into self._dataset"""
        self._dataset = pd.merge(self._dataset, df, on=key)

    def get_dataset(self, cls=None, training=True):
        """
        Return train/val/test dataset
        """
        if cls != None:
            _df = self._dataset.loc[self._dataset["split"] == cls]
        else:
            _df = self._dataset

        # Drop split column
        if "split" in _df.columns:
            _df.drop(["split"], axis=1, inplace=True)

        return self.from_dataframe(
            _df, training=training, target_col_list=self.target_col_list
        )

    @staticmethod
    def save_df(df, save_dir, save_args={}):
        """
        Save df into save_dir

        Arguments:
        ---------
            df(pd.DataFrame):
                Target dataframe object
            save_dir(float):
                Save path: should be ended with .csv suffix
            save_args(dict):
                Arguments for pd.DataFrame.to_csv function.

        """

        _save_dir = os.path.dirname(save_dir)

        if not os.path.exists(_save_dir):
            os.makedirs(_save_dir)

        df.to_csv(save_dir, **save_args)
