#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   ___ _      _   ___ ___ ___ ___ ___ ___   _   ___ _____ ___  _  _ 
  / __| |   /_\ / __/ __|_ _| __|_ _/ __| /_\ |_ _|_   _/ _ \| \| |
 | (__| |__ / _ \\__ \__ \| || _| | | (__ / _ \ | |  | || (_) | .` |
  \___|____/_/ \_\___/___/___|_| |___\___/_/ \_\___| |_| \___/|_|\_|
                                                                    
																	
	SFL Regression Validation module
	
		- mainly for regression validation
		
	
	SFL Scientific 19.AUG.18
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)

from .regression_contour_scatter_plot import actual_v_predictions_plot

import logging

logFormatter = "%(asctime)s - %(levelname)s - %(message)s"
MSG_LEVEL_NUM = 45
logging.addLevelName(MSG_LEVEL_NUM, "MSG")


def msg(self, message, *args, **kws):
    if self.isEnabledFor(MSG_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(MSG_LEVEL_NUM, message, args, **kws)


logging.Logger.msg = msg
logging.basicConfig(format=logFormatter, level=logging.ERROR)
logger = logging.getLogger(__name__)


def scatter_plot(
    df,
    truth_col,
    pred_col,
    class_col=None,
    show_residual=True,
    xlim=None,
    ylim=None,
    s=50,
):
    """
    scatter plot of truth and predicted
    Inputs:
        df (pd.DF): data frame to work on
        truth_col (str): truth column name
        pred_col (str): predicted column name
        show_residual (bool): show a residual plot [not yet implemented] [added]
        xlim (pair float): xmin, xmax (defaults to range of dataset)
        ylim (pair float): ymin, ymax (defaults to range of dataset)
        s (float): marker size
    """
    f1 = plt.figure(1)

    # set the limits manually
    if xlim is None:
        xmin = np.min(
            [
                df[truth_col].min(),
                df[pred_col].min(),
            ]
        )
        xmax = np.max(
            [
                df[truth_col].max(),
                df[pred_col].max(),
            ]
        )
    if ylim is None:
        ymin = np.min(
            [
                df[truth_col].min(),
                df[pred_col].min(),
            ]
        )
        ymax = np.max(
            [
                df[truth_col].max(),
                df[pred_col].max(),
            ]
        )

    if ymin < xmin:
        xmin = ymin
    elif ymin > xmin:
        ymin = xmin

    if class_col is not None:
        colormap = plt.cm.nipy_spectral  # gist_ncar #nipy_spectral, Set1,Paired
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(df[class_col].unique()))]

        for i, (g, df_g) in enumerate(df.groupby(class_col)):
            plt.scatter(df_g[truth_col], df_g[pred_col], label=g, c=colors[i], s=s)

        plt.legend()
    else:
        plt.scatter(df[truth_col], df[pred_col], s=s)

    plt.plot(
        [0, df[truth_col].max()],
        [0, df[truth_col].max()],
        linestyle="--",
        lw=2,
        color="r",
    )
    plt.xlabel(truth_col)
    plt.ylabel(pred_col)
    plt.xlim((xmin * 0.9, xmax * 1.2))
    plt.ylim((ymin * 0.9, ymax * 1.2))
    f1.show()

    if show_residual:
        f2 = plt.figure(2)
        residuals = df[pred_col] - df[truth_col]
        plt.scatter(df[pred_col], residuals, s=s)
        plt.hlines(y=0, xmin=xmin, xmax=xmax, linestyle="dashed")
        plt.xlabel("predicted value")
        plt.ylabel("residuals")
        f2.show()


def regression_scatter_plot(truth, prediction):
    """ Display a scatter plot of actual vs predicted values, along with mae/r2/rmse scores """
    regression_result = pd.DataFrame({"Truth": truth, "Prediction": prediction})
    scatter_plot(
        regression_result,
        truth_col="Truth",
        pred_col="Prediction",
        class_col=None,
        show_residual=True,
        xlim=None,
        ylim=None,
        s=50,
    )

    actual_v_predictions_plot(
        truth,
        prediction,
        "regression result",
        fig_size=(7, 7),
        ci=95,
        color="orange",
        labels=None,
        label_dummies=None,
        save=False,
        vmin=-10,
        vmax=110,
        scatter=True,
        contour=True,
    )
    mape = mean_absolute_percentage_error(truth, prediction)
    r2 = r2_score(truth, prediction)
    rmse = np.sqrt(mean_squared_error(truth, prediction))

    logger.msg("TEST SET MAPE = %.2f,\n R2 = %.2f \n RMSE = %s" % (mape, r2, rmse))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
