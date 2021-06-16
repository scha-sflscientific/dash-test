#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    SFL module

    Splits data set into a train/val/test for time series data.

    SFL Scientific 17.JAN.19
"""

import pandas as pd
import numpy as np

"""
    IMPORTANT

    the imput dataframe should have a period_id column (float) that indicates which BASE PERIOD it belongs to.
    For example, a dataframe contains 80 weeks (4 months) worth of data. If week is decided to be the base period, there
    should be a period_id column with values range from 1 to 80, where entries belong to the same week have the same number.
    Similarly, if month is decided to be the base period, the period_id column should now contain only numbers 1 to 4. Entries belong to 
    the same month will have the same number.
    The period_id should start at 1 and the dataframe should be sorted so that the period_id column is in ascending order

    It is better to do this as a pre-processing step so that there will be more freedom in manipulating the data.
"""


def _generate_splits(df, pps_train, pps_test, period_id, win_type, col_name):
    """
    Internal method to append columns to df which signifies the splits.

    pps_train: number of base periods of each train set
    pps_test: number of base periods of each test set (pps_test is generally no larger than pps_train)
    period_id: a column of float values that designates the base period for each entry
    win_type: sliding (same length of train) and expanding
    col_name: column name for the added column with 'train', 'test' labels.

    NOTE: if only splittig to one train and one test set, set the pps_train to be more than half of the total number of periods, and pps_test to None.
    The function will automatically use all data not included in the train set for test.
    """

    train_indice = []

    total_number_periods = int(np.max(period_id))
    if pps_test is not None:
        # to avoid the last training set has no matching test data, keep the test data first
        total_number_periods = total_number_periods - pps_test

    n_splits = total_number_periods // pps_train
    print(
        "Splitting the dataset to %d training and %d testing sets."
        % (n_splits, n_splits)
    )
    period_id = np.array(period_id)

    # split the index to get training label
    if n_splits > 1:
        train_ends = list(range(1, total_number_periods, pps_train))
        if len(train_ends) == n_splits:
            train_ends.append(total_number_periods + 1)
        for i in range(1, n_splits + 1):
            if win_type == "sliding":
                train_indice = []
            to_add_train = list(
                np.where(
                    np.array(period_id >= train_ends[i - 1])
                    & np.array(period_id < train_ends[i])
                )[0]
            )
            train_indice.extend(to_add_train)
            test_end = train_ends[i] + pps_test
            test_indice = list(
                np.where(
                    np.array(period_id >= train_ends[i])
                    & np.array(period_id < test_end)
                )[0]
            )

            new_col = np.array([None] * len(df.index.values))
            new_col[train_indice] = "train"
            new_col[test_indice] = "test"
            df[col_name + "_" + str(i)] = new_col
    if n_splits == 1 and pps_test is None:
        train_indice = list(np.where(np.array(period_id <= pps_train))[0])
        test_indice = list(
            np.where(
                np.array(period_id > pps_train)
                & np.array(period_id <= np.max(period_id))
            )[0]
        )
    return df


def _check_bleed(df, group):
    pass


def _validate_inputs(func):
    """
    Decorator for main function.  If there are any errors, it will return
    them all at once.
    """

    def wrapper(
        df,
        period,
        idx=None,
        n_splits=5,
        split_size=None,
        window_type="expanding",
        step_frac=1.0,
        group=[],
        col_name="split",
    ):

        errors = {}
        if type(df) is not pd.DataFrame:
            errors["df"] = "The df is not of type pd.DataFrame"

        if period_id not in df.columns.values.tolist():
            errors["period_id"] = (
                "The specified column does not exist in %s."
                % df.columns.values.tolist()
            )

        if window_type != "expanding" and window_type != "sliding":
            errors["window_type"] = (
                "Type %s " % window_type
                + 'is not supported. Only "expanding" and '
                + '"sliding" are supported.'
            )

        try:
            str(col_name)
        except:
            errors["col_name"] = "The passed value cannot be made a str."

        if len(errors) > 0:
            for key in errors:
                print("\nERROR: There was a mismatch for the following...")
                print("%s : %s" % (key, errors[key]))
            raise Exception(
                "ERROR: Some inputs to time_series_split() "
                + "did not meet specifications."
            )

        ret = func(df, period_id=period_id, window_type=window_type, col_name=col_name)
        return ret

    return func


@_validate_inputs
def forecasting_split(
    df,
    period_id=None,
    window_type="expanding",
    pps_train=12,
    pps_test=4,
    col_name="split",
):
    """
    Takes data in df and splits it into train, and test sets by
    adding 'split_#' columns.  Can be a sliding or expanding window.

    Args:
        df (pd.DataFrame):  Original df of entire dataset. Assumed sorted by
                            row (row n occurs at the same time or before row
                            n+1, but not after).

        period_id (str):    df must have a period_id column.

        window_type (str):  Default 'sliding', can be set to 'expanding'.

        pps_train (int):  number of base periods in trainning set.

        pps_test (int): number of base periods in testing set.

        col_name (str):     Default 'split', used to name the new column.

    Returns:
        df (pd.DataFrame): Original df with addition column col_name.

    """

    # Calculate the number of units by given period.
    if period_id is None:
        sys.exit("period_id is required")
    else:
        period_id = df[period_id].values.tolist()
        try:
            period_id = [float(x) for x in period_id]
        except:
            raise Exception(
                "Elements in passed index column contains "
                + "values that cannot be cast to float."
            )

    total_periods = np.max(period_id)
    print("The dataset contains %s periods." % total_periods)

    df = _generate_splits(
        df,
        pps_train=pps_train,
        pps_test=pps_test,
        period_id=period_id,
        win_type=window_type,
        col_name=col_name,
    )

    # df = _check_bleed(df, group)

    return df


if __name__ == "__main__":
    print("TESTING time_series_split.py ...")

    df = pd.read_csv("arem.csv")
    df = forecasting_split(
        df, period_id="ts_id", pps_train=12, pps_test=4, window_type="sliding"
    )
    df.to_csv("test_sliding.csv")

    df = pd.read_csv("arem.csv")
    df = forecasting_split(
        df, period_id="ts_id", pps_train=12, pps_test=4, window_type="expanding"
    )
    df.to_csv("test_expanding.csv")
