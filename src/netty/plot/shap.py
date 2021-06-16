import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import shap

shap.initjs()


def shap_feature_importance(model, train_X):
    """ Get summary plot of feature importance """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_X)

    shap.summary_plot(shap_values, train_X, plot_type="bar")
    plt.show(block=False)
