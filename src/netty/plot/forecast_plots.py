import warnings
import matplotlib.pyplot as plt
import plotly

plotly.offline.init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")


def prophet_score_scatter_plot(scores_dict):

    """
    prophet score scatter plot

    Arguments:
    ---------
        scores_dict(dict):
            prophet score dictionary: {ID:score}

    Returns
    -------
        model(XGBClassifier, XGBRegressor):
    """

    mape = [
        scores_dict[key]["mape"]
        .agg(
            {
                "mean": "mean",
                "min": "min",
                "max": "max",
                "first": lambda x: x.iloc[0],
                "last": lambda x: x.iloc[-1],
            }
        )
        .values
        for key in scores_dict.keys()
        if key != "fail"
    ]

    rmse = [
        scores_dict[key]["rmse"]
        .agg(
            {
                "mean": "mean",
                "min": "min",
                "max": "max",
                "first": lambda x: x.iloc[0],
                "last": lambda x: x.iloc[-1],
            }
        )
        .values
        for key in scores_dict.keys()
        if key != "fail"
    ]

    traces = []
    for i in range(len(mape)):
        key = list(scores_dict.keys())[i]
        traces.append(
            plotly.graph_objs.Scatter(
                x=mape[i],
                y=rmse[i],
                text=[
                    "mean %s" % key,
                    "min %s" % key,
                    "max %s" % key,
                    "first %s" % key,
                    "last %s" % key,
                ],
                hoverinfo="text",
                mode="markers",
                name=str(key),
            )
        )
        layout = plotly.graph_objs.Layout(
            title="cross validation scores",
            xaxis=dict(title="mape"),
            yaxis=dict(title="rmse"),
        )
    fig = dict(data=traces, layout=layout)
    plotly.offline.plot(fig)
    return


def prophet_ts_plots(train_df, forecast_df):

    plt.figure(figsize=(16, 5))
    # forecast = self.predictions[id_].forecast
    # cutoff_date = self.predictions[id_].cutoff_date

    plt.plot(train_df["ds"], train_df["y"], "k")

    plt.plot(forecast_df["ds"], forecast_df["yhat"], "-r")

    forecast_df.set_index("ds", inplace=True)
    plt.fill_between(
        forecast_df.index,
        forecast_df["yhat_lower"],
        forecast_df["yhat_upper"],
        facecolor="r",
        alpha=0.2,
    )
    plt.xticks(rotation=90)
    plt.title("Forecast")
    plt.show(block=False)
    return
