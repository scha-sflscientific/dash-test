# -*- coding: utf-8 -*-

"""Example code for the Builder Module. This code is meant
just for illustrating basic SFL Template features.

Update this when you start working on your own SFL project.
"""

import re
import os
import ast
import sys
import glob
import logging

import warnings

warnings.filterwarnings("ignore")

from src.models.model import ExampleModel

# ------------------------------------------------------------------------------#
#                                 LOGGER                                   #
# ------------------------------------------------------------------------------#


from src import logger

# ------------------------------------------------------------------------------#
#                                MODEL BUILDER                                  #
# ------------------------------------------------------------------------------#


class ExampleBuilder(object):

    """
    Class represents that a builder to build dummy classification model
    """

    @staticmethod
    def build(config, **kwargs):
        """
        Build a dummy Model.

        Arguments:
        ---------

            config(dict):
                A python dictionary including following hyperparameters, the hyperparameters are defined in config/config.ini

        Returns
        -------
            model(torch.nn.module):
                Pytorch model

        """

        EXAMPLE_NUM_TRAIN_ITER = config.get("EXAMPLE_NUM_TRAIN_ITER")
        EXAMPLE_LEARNING_RATE = config.get("EXAMPLE_LEARNING_RATE")

        xgb_estimator = XGBModel(
            max_depth=3,
            learning_rate=EXAMPLE_LEARNING_RATE,
            n_estimators=EXAMPLE_NUM_TRAIN_ITER,
            silent=True,
            verbosity=4,
            booster="gbtree",
            n_jobs=1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=0.5,
            random_state=42,
            seed=42,
        )

        return xgb_estimator
