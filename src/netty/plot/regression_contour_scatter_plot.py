import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

plt.style.use("ggplot")


def abline(slope, intercept):
    """
    DESC: Plot a line from slope and intercept
    INPUT: slope(float), intercept(float)
    -----
    OUTPUT: matplotlib plot with plotted line of desired slope and intercept
    """
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--", c="b", label="Perfect Predictions")


def line_of_best_fit(x, y, ci, label, c, scatter=True):
    """
    DESC: Plot a line of best fit from scatter plot
    INPUT: x-coordinates(list/array), y-coordinates(list/array), confidence-interval(float), label(str)
    -----
    OUTPUT: seaborn plot with plotted line of best fit with confidence interval and equation for line of best fit
    """
    sns.regplot(x, y, fit_reg=True, scatter=scatter, label=label, ci=ci, color=c)
    return np.polynomial.polynomial.polyfit(x, y, 1)


def scatter_plot2d(
    df,
    col1,
    col2,
    by=False,
    figsize=(8, 6),
    label=["Canola", "Durum", "Lentil", "Hard Wheat"],
    vmin=0,
    vmax=60,
    xlabel=None,
    ylabel=None,
    title=None,
    save=False,
):
    """
    DESC:
            Plot 2d histogram colored by group column
    INPUT:
            df(pd.DataFrame):           Target dataframe
            co11(str):                  First target column
            col2(str):                  Second target column
            by(str):                    group column
            label(list):                legend labels
            vmin(int):                  min value for xlim/ylim
            vmax(int):                  max value for xlim/ymin
    -----
    OUTPUT: matplotlib 2d scatter plot with perfect matching line
    """
    if by:
        num_unique = df[by].nunique()
        unique_value = sorted(df[by].unique())
        cmap = plt.cm.get_cmap("hsv", num_unique + 1)
        colors = []
        for i in range(num_unique):
            colors.append(cmap(i))
        axes = []
        values = []
        k = 0
        for value, c in zip(unique_value, colors):
            #            print (c,value)
            ax = plt.scatter(
                df.loc[df[by] == value][col1].values,
                df.loc[df[by] == value][col2].values,
                c=c,
                alpha=0.7,
            )
            axes.append(ax)
            values.append(value)
            k += 1
        legend1 = plt.legend(
            tuple(axes), tuple(values), scatterpoints=1, loc="best", prop={"size": 8}
        )
        plt.xlim(vmin, vmax)
        plt.ylim(vmin, vmax)
        # plt.xticks(np.arange(1000,6000,1000),fontsize=12)
        # plt.yticks(np.arange(1000,6000,1000),fontsize=12)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if save:
            plt.savefig(save)
    return legend1


def density_estimation(m1, m2, vmin, vmax):
    X, Y = np.mgrid[vmin:vmax:100j, vmin:vmax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


def actual_v_predictions_plot(
    actual,
    preds,
    title,
    fig_size=(7, 7),
    ci=80,
    color="orange",
    labels=None,
    label_dummies=None,
    save=False,
    vmin=None,
    vmax=None,
    scatter=True,
    contour=True,
):
    """
    DESC: Creates and acutal v.s. predictions plot to evaluate regressions
    INPUT: actual(list/array), preds(list/array), title(str), ci(float), pos1(tuple), pos2(tuple), save(bool)
    -----
    OUTPUT: matplotlib plot with prefect fit, line of best fit equation and plot, scatter plot of actual vs predicted values and MAPE
    """
    if vmin is None:
        vmin = np.min(np.min(actual), np.min(preds))
    if vmax is None:
        vmax = np.max(np.max(actual), np.max(preds))

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(fig_size))
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    best_fit_eq = line_of_best_fit(
        actual,
        preds,
        ci=ci,
        label="Line of Best Fit with {}% CI".format(ci),
        c=color,
        scatter=scatter,
    )
    if scatter:
        if isinstance(label_dummies, pd.DataFrame):
            labels = label_dummies.idxmax(axis=1)
            df = pd.DataFrame(
                {
                    "Actual": actual.tolist(),
                    "Predictions": list(preds),
                    "Labels": labels,
                }
            )
            legend1 = scatter_plot2d(df, "Actual", "Predictions", by="Labels")
        if isinstance(labels, np.ndarray):
            df = pd.DataFrame(
                {
                    "Actual": actual.tolist(),
                    "Predictions": list(preds),
                    "Labels": labels,
                }
            )
            legend1 = scatter_plot2d(
                df, "Actual", "Predictions", by="Labels", vmin=vmin, vmax=vmax
            )
    abline(1, 0)
    # MAE, RMSE, MAPE = regression_evaluation(df['Actual'].values, df['Predictions'].values)
    if contour:
        X, Y, Z = density_estimation(actual, preds, vmin, vmax)
        # Show density
        ax.imshow(
            np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[vmin, vmax, vmin, vmax]
        )
        # Add contour lines
        ax.contour(X, Y, Z, 10)
        #         ax.plot(actual, preds, 'k.', markersize=2)
        ax.set_xlim([vmin, vmax])
        ax.set_ylim([vmin, vmax])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Prediction")
    ax.set_title(title)
    try:
        plt.gca().add_artist(legend1)
    except:
        pass
    if save:
        #         plt.subplots_adjust(top=0.88, left=0.1, right=0.9)
        plt.savefig(str(save) + ".eps", format="eps", dpi=1000, bbox_inches="tight")

    plt.show()
    print("Line of Best Fit: \t\t y = {}x + {}".format(best_fit_eq[1], best_fit_eq[0]))
    return
