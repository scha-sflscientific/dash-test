# -*- coding: utf-8 -*-

import os
import sys
import logging

from abc import ABCMeta, abstractmethod

import warnings

warnings.filterwarnings("ignore")


class Processes(object):
    """An abstract class representing an features."""

    @abstractmethod
    def preprocess(self, transform_functions):
        """preprocess"""
        raise NotImplementedError
