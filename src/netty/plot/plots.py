#  ___ _    ___ _____ ___
# | _ \ |  / _ \_   _/ __|
# |  _/ |_| (_) || | \__ \
# |_| |____\___/ |_| |___/
#
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../utils")
import metrics


def learning_curve(
    estimator, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
):

    plt.figure()

    from sklearn.model_selection import learning_curve

    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("training examples")
    plt.ylabel("score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="cross-validation score"
    )

    plt.legend()
    return plt


def acc_vs_threshold(df, thresholds, preds_col, target_col, thresh_col):
    """
    df (pd.DataFrame): dataframe should contain at least 3 columns:
        preds_col: predictions
        target_col: true values
        thresh_col: feature column containing the threshold values
    thresholds (list): list of thresholds
    """
    mape = []
    scatter_size = []
    for val in thresholds:
        mape.append(
            metrics.mean_absolute_percentage_error(
                df[df[thresh_col] <= val][target_col],
                df[df[thresh_col] <= val][preds_col],
            )
        )

        scatter_size.append(df[df[thresh_col] <= val].shape[0])

    plt.figure(figsize=(16, 8))
    size = np.array(scatter_size)
    plt.scatter(thresholds, mape, color="r", s=size)
    plt.xlabel(target_col)
    plt.ylabel("MAPE")
    plt.title("Cumulative MAPE")
    return plt
