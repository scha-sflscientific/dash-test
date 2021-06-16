#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    SFL module

    Splits data set into a train/val/test without data bleed.

    SFL Scientific 24.JAN.19
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def _get_index(keys, mapping):
    """
    Convert a list of keys into a list of index values using the mapping.
    """
    output = []
    for key in keys:
        output += mapping[key]

    return output


def _validate_inputs(func):
    """
    Decorator for main function.  If there are any errors, it will return
    them all at once.
    """

    def wrapper(df, train=0.7, val=0.1, test=0.2, group=[], seed=41):

        errors = {}

        if type(df) is not pd.DataFrame:
            errors["df"] = "The DataFrame passed is not a DataFrame."

        if type(train) is not float:
            errors["train"] = "train must be of type float."
        if train <= 0.0 or train > 1.0:
            errors["train"] = "train must be 0.0 < train <= 1.0."
        if type(val) is not float:
            errors["val"] = "val must be of type float."
        if val <= 0.0 or val > 0.99:
            errors["val"] = "val must be 0.0 < val <= 0.99."
        if type(test) is not float:
            errors["test"] = "test must be of type float."
        if test <= 0.0 or test > 0.99:
            errors["test"] = "test must be 0.0 < test <= 0.99."

        if type(group) is not list:
            errors["group"] = "group must be a list"
        for elem in group:
            if type(elem) is not str:
                errors["group"] = "Not all elems in group are strings."
                break

        if type(seed) is not int:
            errors["seed"] = "The seed must be an int."

        if len(errors) > 0:
            for error in errors:
                print("%s : %s" % (error, errors[error]))
            return (errors, False, False)

        ret = func(df=df, train=train, val=val, test=test, group=group, seed=seed)
        return ret

    return wrapper


@_validate_inputs
def bleedless_split(df, train=0.7, val=0.1, test=0.2, group=[], seed=41):
    """
    Split the dataset df into a train/va/test set. If group is not empty,
    the columns in the groups will be used to uniquly id 'subjects', who
    will exist only in one set (no data bleed).

    Inputs:
        df (pd.DataFrame):  Dataframe with the data. Each row should be
                            a single point.

        train (float):      Approx fraction to go into training set.

        val (float):        Approx fraction to go into validaiton set.

        test (float):       Approx fraction to go into test set.

        group (list):       Str list of columns used to prevent entries
                            in the same cluster from appearing in different
                            sets.
        seed (int):         Make the splits random, but reproducible.

    Outputs:
        tuple (pd.DataFrame): dfs train, val, and test. In that order.

    """
    # Outputs
    train_df = pd.DataFrame(columns=df.columns.values)
    val_df = pd.DataFrame(columns=df.columns.values)
    test_df = pd.DataFrame(columns=df.columns.values)

    # Split the df into groupings which can be split by sklearn
    if len(group) == 0:
        idx = df.index.values
    else:
        # Get the indeces for each group, for faster indexing later
        temp = df.groupby(group).first().reset_index().astype(str)
        idx = list(pd.unique(temp[group].apply(lambda x: "_".join(x), axis=1)))

    # split = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=0)    <== stratified k-fold for classification when class imbalances
    # split.get_n_splits(idx, yclasses)

    X_tr, X_ts, _, _ = train_test_split(
        idx, [None] * len(idx), test_size=test, random_state=seed
    )

    val /= train  # Scale val to keep splits the same size
    X_tr, X_val, _, _ = train_test_split(
        X_tr, [None] * len(X_tr), test_size=val, random_state=seed
    )

    # Convert str keys back into index values
    temp = df[group].astype(str)
    temp = temp[group].apply(lambda x: "_".join(x), axis=1).to_frame("idx")

    if len(group) != 0:
        X_tr = temp.loc[temp["idx"].isin(X_tr)].index
        X_val = temp.loc[temp["idx"].isin(X_val)].index
        X_ts = temp.loc[temp["idx"].isin(X_ts)].index

    # Convert the index values back into the df as a column
    train_df = df.loc[X_tr]
    val_df = df.loc[X_val]
    test_df = df.loc[X_ts]
    # return (train_df, val_df, test_df)

    # DW: return df with "split" column
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    df = pd.concat([train_df, val_df, test_df])
    return df


if __name__ == "__main__":
    print("\nTESTING bleedless_split.py ...\n")
    df = pd.read_csv("arem.csv")
    X_tr, X_val, X_ts = bleedless_split(df, group=["ts_id"])

    train_ids = list(set(X_tr["ts_id"].values.tolist()))
    val_ids = list(set(X_val["ts_id"].values.tolist()))
    test_ids = list(set(X_ts["ts_id"].values.tolist()))

    for idx in train_ids:
        if idx in val_ids:
            print(idx)
            assert False
        if idx in test_ids:
            print(idx)
            assert False
    for idx in val_ids:
        if idx in test_ids:
            print(idx)
            assert False
