# -*- coding: utf-8 -*-
"""Example code for the processor. This code is meant
just for illustrating basic SFL Template features.

Update this when you start working on your own SFL project.
"""


import os
import time
import tqdm
import logging
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------#
#                                 LOGGER                                   #
# ------------------------------------------------------------------------------#

from src import logger, config

# ------------------------------------------------------------------------------#
#                                 Trainer                                   #
# ------------------------------------------------------------------------------#


class Trainer(object):
    """
    An abstract class to represent model training process.

    """

    @classmethod
    def train(cls, model, train_dataset, validate_dataset, **kwargs):

        """
        Model Training Example Pipeline

        Arguments:
        ---------

            model(ExampleModel):
                Machine Learning Model
            train_dataset(IrisDataset):
                A Dataset class to represent training data.
            validate_dataset(IrisDataset):
                A Dataset class to represent validation data.

        Returns
        -------
            Model(DummyXGBModel):
                Trained Dummy XGB model.

        """

        train_df = train_dataset.get_dataframe()
        val_df = validate_dataset.get_dataframe()

        target_col_list = train_dataset.get_target_col_list()
        feat_col_list = [col for col in train_df.columns if col not in target_col_list]

        train_X = train_df[feat_col_list]
        train_Y = train_df[target_col_list]

        val_X = val_df[feat_col_list]
        val_Y = val_df[target_col_list]

        # Start training
        model.fit(
            train_X,
            train_Y,
            eval_set=[(val_X, val_Y)],
        )

        # Save trained model.
        if "save_dir" in kwargs:

            save_dir = kwargs["save_dir"]
            model.save(save_dir)
