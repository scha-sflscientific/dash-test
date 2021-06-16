"""Example code for the processor. This code is meant
just for illustrating basic SFL Template features.

Update this when you start working on your own SFL project.
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.classes.process import Processes

# ------------------------------------------------------------------------------#
#                                  LOGGING                                     #
# ------------------------------------------------------------------------------#

from src import logger


# ------------------------------------------------------------------------------#
#                                 MAIN                                   #
# ------------------------------------------------------------------------------#


class ExampleProcessor(Processes):
    @staticmethod
    def process(dataset, tfms):
        """
        Preprocess Example dataset

        Arguments:
        ---------
            tfms(Transforms):
                Transforms function used to preprocess

        Returns
        -------
            dataset(ExampleDataset):
                preprocessed dataset
        """

        return tfms(dataset)
