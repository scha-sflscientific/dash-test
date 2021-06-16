# -*- coding: utf-8 -*-

"""Example code for the Dataset Module. This code is meant
just for illustrating basic SFL Template features.

Update this when you start working on your own SFL project.
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from src.classes.datasets import Datasets, CSVDataSets


# ------------------------------------------------------------------------------#
#                                 LOGGER                                   #
# ------------------------------------------------------------------------------#

from src import logger

# ------------------------------------------------------------------------------#
#                                 CONFIG                                   #
# ------------------------------------------------------------------------------#

from src import config

# ------------------------------------------------------------------------------#
#                                 PARAMETERS                                   #
# ------------------------------------------------------------------------------#

RAW_DIR = config.get("DATA", "RAW_DIR")


class ExampleDataset(CSVDataSets):

    """

    The class represents an Example Dataset Module.
    The Dataset is inherited from CSVDataSets

    Arguments:
    ---------
        path(str):
            Directory of DICOM images
        _dataset(pd.DataFrame):
            pd.Dataframe includes raw image path, processed image path, ETL results.
            For training, ground truth is saved in "MUBADA" column as well.

    Methods:
    --------
        get_dataframe:
            Return _dataset attribute as pd.DataFrame.
        get_dataset:
            return train/val/test set as a new PETImageDataset object
        merge_df:
            Merge another pd.DataFrame into self._dataset
        split:
            Splitting a dataframe df into train, val, and test sets using optional grouping for bleeding prevention
        from_path:
            Load image data from path
        from_dataframe:
            Load image data from pd.DataFrame
        sanity_check:
            input data sanity check
        consistency_check:
            validate ETL results check with pre-defined output
        feature_check:
            fill/drop data based on feature white list


    Note: Split/Sanity Check/Consistency Check/Feature Check functions can be user-defined

    """

    def sanity_check(self):
        """
        Example Sanity Check Function

        """

        ###########################################################################
        # Here you can find an example Sanity Check function.
        #
        # Delete this when you start working on your own SFL project

        # -------------------------------------------------------------------------
        if not isinstance(self._dataset, pd.DataFrame):
            logger.critical("Input data must be a pandas dataframe")
            sys.exit()
        elif self._dataset.index.nunique() < 20 and self.training:
            logger.critical(
                "[IOThere are less than 20 IDs, ML training is not possible"
            )
            sys.exit()
        elif len(self._dataset) == 0 and self.training == False:
            logger.error("[IO]No input data found")
            sys.exit()

        self._sanity_check = True
