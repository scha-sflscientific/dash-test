# -*- coding: utf-8 -*-

"""Example code for the training pipeline. This code is meant
just for illustrating basic SFL Template features.

Update this when you start working on your own SFL project.
"""

import os
import sys

import warnings

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

from src.models.builder import ExampleBuilder
from src.models.trainer import ExampleTrainer

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
SAVE_DIR = config.get("TRAIN").get("SAVE_DIR")
MODEL_CONFIG = config.get("MODEL")


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


def train():
    """
    Create the project's training pipeline.
    """

    ## DS Initialization Placeholder
    example_ds = ExampleDataset.from_path(
        path=RAW_DIR,
        training=True,
    )
    logger.info("[Training]Dataset Initlization Completed.")

    ## Data Pipeline Process Placeholder
    example_ds = ExampleProcessor.process(example_ds, tfms=PREPROCESS_TRANSFORMS)
    logger.info("[Training]Dataset Processing Completed.")

    ## Model Initialization Placeholder
    example_model = ExamplelBuilder.build(config=MODEL_CONFIG)
    logger.info("[Training]Model Initliazation Completed.")

    ## Training Placeholder
    logger.info("[Training]Start Training.....")
    train_ds = example_ds.train_set
    validate_ds = example_ds.validation_set

    Trainer.train(example_model, train_ds, validate_ds, save_dir=SAVE_DIR)
    logger.info("[Training]Training Completed.")


if __name__ == "__main__":
    train()
