#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  ___ _    ___ _____ ___ 
 | _ \ |  / _ \_   _/ __|
 |  _/ |_| (_) || | \__ \
 |_| |____\___/ |_| |___/
                         
																	
	SFL validation module
	
		- generic validation module
		
	SFL Scientific 14.Jan.18
"""

def feat_importance(gbm, feature_list, number=15, flag=True):
    """
    Plot all feature importance and top importance feature in hist
    Args:
        number(int): how many top importance
        folder_name: model folder name
        flag: True for xgboost and False for random forest or other models
    Returns:
        feature importance
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    bst = gbm.booster()
    importance = bst.get_fscore()

    newdict = dict(
        (feature_list[int(key.replace("f", ""))], value)
        for (key, value) in importance.items()
    )
    df_feat = pd.DataFrame(
        {"Importance": list(newdict.values()), "Feature": list(newdict.keys())}
    )

    df_feat = df_feat.sort_values(by="Importance", ascending=[False])

    fig = plt.figure(1, [6, 5])

    temp = df_feat[0:number]
    names = temp.Feature
    x = np.arange(len(names))
    plt.bar(x, temp.Importance.ravel())
    plt.xlabel("Feature", size=16)
    plt.ylabel("Normalised Importance", size=16)
    plt.title("Feature Importance Ranking", size=16)
    _ = plt.xticks(x + 0.5, names, rotation=90)
    plt.ylim([0, np.max(temp.Importance.ravel()) + 1.4])
    plt.xlim([-1, len(names) + 1])
    plt.show()

    x = list(range(0, len(df_feat["Importance"])))
    y = df_feat.Importance.ravel()

    plt.plot(
        x,
        y,
        linestyle="-",
        linewidth=1.0,
    )
    plt.xlabel("Number of Features")
    plt.ylabel("Normalized Feature Importance")
    plt.title("Feature Importance of all Features")
    plt.ylim([0, np.max(y) + 1.4])
    plt.show()

