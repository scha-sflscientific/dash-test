# -*- coding: utf-8 -*-
import os
import sys

import toml
import glob
import yaml
import logging
import logging.config
import collections
import configparser
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from os import path as os_path
from sys import path as sys_path

# ------------------------------------------------------------------------------#
#                                 LOGGER                                   #
# ------------------------------------------------------------------------------#


def setup_log_config(log_path, log_file_name, log_config_file):
    """Set up logger configuration"""

    os.makedirs(log_path, exist_ok=True)
    with open(log_config_file, "r") as f:
        log_cfg = yaml.safe_load(f.read())
    log_cfg["handlers"]["file_handler"]["filename"] = os_path.join(
        log_path, log_file_name
    )
    logging.config.dictConfig(log_cfg)


logger = setup_log_config("./logs/", "main.log", "./config/log_config.yaml")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------#
#                                 PARAMETERS                                   #
# ------------------------------------------------------------------------------#

CONFIG_LIST = glob.glob("./config/*.toml")

config = toml.load(CONFIG_LIST)

# ------------------------------------------------------------------------------#
#                                Config Parser                                 #
# ------------------------------------------------------------------------------#

import ast


def config_parser(input):
    """
    config parser to parse non-string type input. Return original input if input type is string.

    Examples:

        Input:
            "Hello World"
        Output:
            "Hello World"

        Input:
            "[1,2,3,4,5]"
        Output:
            [1,2,3,4,5]

    """

    try:
        return ast.literal_eval(input)
    except ValueError:
        return input


# ------------------------------------------------------------------------------#
#                                Config Section Parser                         #
# ------------------------------------------------------------------------------#


def config_section_parser(config):
    """
    config sectionparser to parse config section. By default, Python Configparser store section key in lowercase.
    However, uppercase by default is used in this template.


    Examples:

        Input config:
            {"my_lowercase_key":123}
        Output config:
            {"MY_LOWERCASE_KEY":123}

    """

    output = {}

    for k, v in config.items():
        output[k.upper()] = v

    return output
