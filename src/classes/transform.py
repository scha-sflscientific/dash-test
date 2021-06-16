# -*- coding: utf-8 -*-

import os
import sys
import logging

from enum import IntEnum
from itertools import groupby
from operator import itemgetter
from abc import ABCMeta, abstractmethod

import warnings

warnings.filterwarnings("ignore")


class TfmType(IntEnum):
    """Type of transformation.
    Parameters
        IntEnum: predefined types of transformations
            NO:    the default, y does not get transformed when x is transformed.
            COORD: y are numeric labels
            CLASS: y are class labels
    """

    NO = 1
    COORD = 2
    CLASS = 3


class Transform(object):
    """A class that represents a transform.

    All other transforms should subclass it. All subclasses should override
    do_transform.

    Arguments
    ---------
        tfm_y : TfmType
            type of transform: See Class TfmType
    """

    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y = tfm_y

    def set_state(self):
        pass

    def __call__(self, x, y):
        self.set_state()
        x, y = (
            (self.transform(x), y)
            if self.tfm_y == TfmType.NO
            else self.transform_coord(x, y)
        )
        return x, y

    def transform_coord(self, x, y):
        return self.transform(x, y=y)

    def transform(self, x, y=None):
        if y is None:
            return self.do_transform(x)
        else:
            return self.do_transform(x, y)

    @abstractmethod
    def do_transform(self, x, y=None):
        raise NotImplementedError


class Transforms:
    """
    Generate a standard set of dataset transformations

    Arguments
    ---------

     tfms :
         iterable collection of transformation functions

    Returns
    -------
     type : ``Transforms``
         transformer for specified operations.
    """

    def __init__(self, tfms):
        self.tfms = tfms

    def __repr__(self):
        return str(self.tfms)

    def __call__(self, dataset):
        dataset._dataset = compose(x=dataset._dataset, fns=self.tfms)

        return dataset


def compose(x, y=None, fns=[], **kwargs):
    """ Apply a collection of transformation functions :fns: to dataset """

    for fn in fns:
        x, y = fn(x, y)

    if y is None:
        return x
    else:
        return x, y
