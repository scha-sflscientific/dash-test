import matplotlib.pyplot as plt


def sfl_defaults():
    plt.style.use("classic")
    plt.rcParams["figure.figsize"] = [8.0, 5.0]
    plt.rcParams["figure.facecolor"] = "w"

    # text size
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.fontsize"] = 12

    # grids
    plt.rcParams["grid.color"] = "k"
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5

    #
    print("SFL style loaded...")


sfl_defaults()
