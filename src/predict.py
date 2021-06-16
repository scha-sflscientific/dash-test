# -*- coding: utf-8 -*-

"""Example code for the inference pipeline. This code is meant
just for illustrating basic SFL Template features.

Update this when you start working on your own SFL project.
"""

import os
import sys

import warnings
import pandas as pd
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------#
#                                 LOGGER                                   #
# ------------------------------------------------------------------------------#

from src import logger, config

# ------------------------------------------------------------------------------#
#                                 MODULE                                   #
# ------------------------------------------------------------------------------#

from src.classes.transform import Transforms
from src.datasets.dataset import ExampleDataset
from src.processes.processor import ExampleProcessor
from src.processes.transforms import (
    ExampleTransform,
)

from src.models.xgb_model import ExampleModel

# ------------------------------------------------------------------------------#
#                                 PARAMETERS                                   #
# ------------------------------------------------------------------------------#

###########################################################################
# Here you can find an example configuration, made of two modular config files(config/config.toml
# and config/credentials.toml).
#
# Delete this when you start working on your own SFL project as
# well as config/*.toml
# -------------------------------------------------------------------------

RAW_DIR = config.get("DATA").get("RAW_DIR")
MODEL_DIR = config.get("INFERENCE").get("MODEL_DIR")
OUTPUT_DIR = config.get("INFERENCE").get("OUTPUT_DIR")

# ------------------------------------------------------------------------------#
#                                 Data Pipeline                                #
# ------------------------------------------------------------------------------#

###########################################################################
# Here you can find an example Transforms pipeline, made of one modular Transform.
#
# Delete this when you start working on your own SFL project as
# well as src/processes/processor.py and src/processes/transforms.py
# -------------------------------------------------------------------------

PREPROCESS_TRANSFORMS = Transforms(
    tfms=[
        ExampleTransform(),
    ]
)

# ------------------------------------------------------------------------------#
#                                 MAIN                                   #
# ------------------------------------------------------------------------------#


def predict():
    """
    Create the project's inference pipeline.
    """

    ## DS Initialization Placeholder
    example_ds = ExampleDataset.from_path(
        path=RAW_DIR,
        training=True,
    )
    logger.info("[Inference]Iris Dataset Initlization Completed.")

    ## Data Pipeline Process Placeholder
    example_ds = ExampleProcessor.process(example_ds, tfms=PREPROCESS_TRANSFORMS)
    logger.info("[Inference]Iris Dataset Processing Completed.")

    ## Load classification model
    example_model = ExampleModel.load(save_dir=MODEL_DIR)
    logger.info("[Inference]Model weight loading Completed.")

    # Inference Process placeholder
    example_df = example_ds.get_dataframe()

    output_df = pd.DataFrame()
    output_df["predicted"] = dummy_model.predict(example_df)
    output_df["index"] = iris_df.index

    # Save inference output
    IrisDataset.save_df(output_df, save_dir=OUTPUT_DIR)


if __name__ == "__main__":
    predict()
