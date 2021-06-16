#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   ___ _      _   ___ ___ ___ ___ ___ ___   _   ___ _____ ___  _  _
  / __| |   /_\ / __/ __|_ _| __|_ _/ __| /_\ |_ _|_   _/ _ \| \| |
 | (__| |__ / _ \\__ \__ \| || _| | | (__ / _ \ | |  | || (_) | .` |
  \___|____/_/ \_\___/___/___|_| |___\___/_/ \_\___| |_| \___/|_|\_|


	SFL validation module

		- mainly for classification validation


	SFL Scientific 21.FEB.18
"""


"""
   ___ ___ _  _ ___ ___ ___ ___
  / __| __| \| | __| _ \_ _/ __|
 | (_ | _|| .` | _||   /| | (__
  \___|___|_|\_|___|_|_\___\___|


"""
import numpy as np
import matplotlib.pyplot as plt


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


def bagging_histogram(truths, preds, cut_off=-1):
    acc = []
    pre = []
    rec = []
    f1s = []
    import sklearn as sk

    truth = truths
    for i in range(len(preds)):
        if cut_off < 1 and cut_off > 0:
            pred = preds[i][:, 1]
            prediction = pred > cut_off
        else:
            prediction = preds[i]
        acc.append(sk.metrics.accuracy_score(prediction, truth[i]))
        pre.append(sk.metrics.precision_score(prediction, truth[i]))
        rec.append(sk.metrics.recall_score(prediction, truth[i]))
        f1s.append(sk.metrics.f1_score(prediction, truth[i]))

    plt.figure(figsize=(8, 5))
    plt.hist(
        acc,
        bins=[0.0 + x * 0.025 for x in range(40)],
        alpha=0.6,
        label="Accuracy",
        color="b",
    )
    plt.axvline(
        linewidth=4,
        color="r",
        x=np.mean(acc),
        label="Mean(Acc) = " + str(round(np.mean(acc) * 1000) / 10),
        c="b",
    )

    plt.hist(
        pre,
        bins=[0.0 + x * 0.025 for x in range(40)],
        alpha=0.6,
        label="Precision",
        color="y",
    )
    plt.axvline(
        linewidth=4,
        x=np.mean(pre),
        label="Mean(Pre) = " + str(round(np.mean(pre) * 1000) / 10),
        c="y",
    )

    plt.hist(
        rec,
        bins=[0.0 + x * 0.025 for x in range(40)],
        alpha=0.6,
        label="Recall",
        color="g",
    )
    plt.axvline(
        linewidth=4,
        x=np.mean(rec),
        label="Mean(Rec) = " + str(round(np.mean(rec) * 1000) / 10),
        c="g",
    )

    plt.hist(
        f1s, bins=[0.0 + x * 0.025 for x in range(40)], alpha=0.6, label="F1", color="r"
    )
    plt.axvline(
        linewidth=4,
        x=np.mean(f1s),
        label="Mean(F1) = " + str(round(np.mean(f1s) * 1000) / 10),
        c="r",
    )

    plt.xlabel("Metric", size=18)
    plt.ylabel("Count", size=18)
    plt.legend(fontsize=13, loc="upper left")
    plt.title("Full Features", size=20)


"""
  _____ _  _ ___ ___ ___ _  _  ___  _   ___ ___ _  _  ___
 |_   _| || | _ \ __/ __| || |/ _ \| |  |   \_ _| \| |/ __|
   | | | __ |   / _|\__ \ __ | (_) | |__| |) | || .` | (_ |
   |_| |_||_|_|_\___|___/_||_|\___/|____|___/___|_|\_|\___|


"""


def probability_distribution(
    df, truth_col, class_column_dict, xlim=(0, 1), ylim=(0, 10), normed=False, fill=True
):

    """
    Probability histogram split by class

    Inputs:
        df (pd.DF): dataframe to work on
        truth_col (str): column name for the truth
        class_column_dict (dict): key = class in truth col, value = probably column associated to class
        xlim (float pair): (xmin, xmax)
        ylim (float pair): (ymin, ymax)
        normed (bool): normlise the histogram
        fill (bool): fill the histogram
    """
    plt.figure(figsize=(8, 6))

    n_bins = 20
    for class_name in class_column_dict:
        prob_pos = df.loc[df[truth_col] == class_name][
            class_column_dict[class_name]
        ].values
        if fill:
            plt.hist(
                prob_pos,
                range=(0, 1),
                bins=n_bins,
                label="Truth_score_ " + str(class_name),
                # histtype="step",
                lw=2,
                normed=normed,
                alpha=0.4,
            )
        else:
            plt.hist(
                prob_pos,
                range=(0, 1),
                bins=n_bins,
                label="Truth_score_ " + str(class_name),
                histtype="step",
                lw=2,
                normed=normed,
                alpha=0.4,
            )

    plt.legend(loc="upper center", ncol=2)
    plt.xlabel("Prediction Probability", fontsize=16)
    plt.ylabel("Count", fontsize=16)

    plt.xticks(fontsize=16)
    # plt.yticks(np.arange(0,100,2),fontsize=16)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.title("Prediction Probability Distribution", fontsize=16)


def logit_sp(truth, predicted_probability, default_class, xlim=(0, 1)):
    """
    logit plot for thresholding - sensitivity, specificity and f1

    Inputs:
        truth (str list): class labels in truth
        predicted_probability (float list): probability of class
        default_class (str): value which should be considered 0
        xlim (float pair): xmin, xmax
    """
    import numpy as np
    import copy
    from sklearn.metrics import f1_score

    sens = []
    spec = []
    clas = []
    f1 = []

    # convert to int and generate labels if string
    if isinstance(truth[0], str):
        # class_dict, labels = _generate_class_dicts(set(truth))
        truth = [0 if x == default_class else 1 for x in truth]

    def _make_logit(cut, mod, y):
        # print cut
        yhat = mod
        yhat[yhat < cut] = 0
        yhat[yhat >= cut] = 1
        w1 = np.where(y == 1)
        w0 = np.where(y == 0)

        sensitivity = np.mean(yhat[w1] == 1)
        specificity = np.mean(yhat[w0] == 0)
        c_rate = np.mean(y == yhat)
        f1_s = f1_score(y, yhat)

        # print sensitivity,specificity,c_rate,f1_s
        sens.append(sensitivity)
        spec.append(specificity)
        clas.append(c_rate)
        f1.append(f1_s)

    bins = [
        ((xlim[1] - xlim[0]) * x) / 100 for x in range(xlim[0], int(xlim[1] * 100) + 1)
    ]
    # make_logit(0.2,preds,labels)
    for i in bins:
        labels = copy.deepcopy(truth)
        preds = copy.deepcopy(predicted_probability)
        _make_logit(i, preds, labels)

    ## Sensitivity: true positive rate -> propotion of positives that are correctly identified as such (percentage of sick people who are correctly identified as sick)
    ## Specificity: true negative rate -> propotion of negatives that are correctly identified as such (percentage of healthy people who are correctly identified as healthy)
    ## A perfect classifier would have 100% sensitivity and specificity
    fig = plt.figure(1, figsize=(5, 5))
    plt.plot(bins, sens, label="Sensitivity")
    plt.plot(bins, spec, label="Specificity")
    plt.plot(bins, clas, label="Classification Rate")
    plt.plot(bins, f1, label="F1")
    plt.gca().xaxis.grid(True)
    # plt.xticks(list(plt.xticks()[0]) + [0.45])
    plt.xlabel("Logit Cutoff", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    # plt.title('')
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.6, 0.7))
    plt.show()


def logit(truth, predicted_probability, default_class, xlim=(0, 1)):
    """
    logit plot for thresholding - precision,recall,accuracy and f1

    Inputs:
        truth (str list): class labels in truth
        predicted_probability (float list): probability of class
        default_class (str): value which should be considered 0
        xlim (float pair): xmin, xrange
    """
    import copy
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    # from sklearn.metrics import f1_score

    # convert to int and generate labels if string
    if isinstance(truth[0], str):
        # class_dict, labels = _generate_class_dicts(set(truth))
        truth = [0 if x == default_class else 1 for x in truth]

    # sens = []
    # spec = []

    recall_s = []
    precision_s = []
    acc_s = []
    clas = []
    f1 = []

    def _make_logit(cut, mod, y):
        # print cut
        yhat = mod
        yhat[yhat < cut] = 0
        yhat[yhat >= cut] = 1

        w1 = np.where(y == 1)
        w0 = np.where(y == 0)

        c_rate = np.mean(y == yhat)
        f1_s = f1_score(y, yhat)

        r_s = recall_score(y, yhat)
        p_s = precision_score(y, yhat)
        acc = accuracy_score(y, yhat)
        # print sensitivity,specificity,c_rate,f1_s
        # sens.append(sensitivity)
        # spec.append(specificity)

        recall_s.append(r_s)
        precision_s.append(p_s)
        acc_s.append(acc)
        clas.append(c_rate)
        f1.append(f1_s)

    bins = [
        ((xlim[1] - xlim[0]) * x) / 100 for x in range(xlim[0], int(xlim[1] * 100) + 1)
    ]

    truth = np.array(truth)
    predicted_probability = np.array(predicted_probability)
    for i in bins:
        labels = copy.deepcopy(truth)
        preds = copy.deepcopy(predicted_probability)
        _make_logit(i, preds, labels)

    fig = plt.figure(1, figsize=(5, 5))
    plt.plot(bins, recall_s, label="Recall")
    plt.plot(bins, precision_s, label="Precision")
    plt.plot(bins, acc_s, label="Accuracy")

    plt.plot(bins, f1, label="F1")
    plt.gca().xaxis.grid(True)
    # plt.xticks(list(plt.xticks()[0]) + [0.45])
    plt.xlabel("Logit Cutoff", fontsize=20)
    plt.ylabel("Value", fontsize=20)

    # plt.legend()
    plt.legend(bbox_to_anchor=(1.4, 0.7))

    plt.xlim([0, 0.99])
    plt.show()


def accuracy_cut(
    df, truth_col, class_column_dict={}, xlim=(-0.01, 1.05), ylim=(-0.05, 1.05)
):
    """
    Plot accuracy cutoff based on f1_score (precision and recall) for all classes

    Inputs:
        df (pd.DF): dataframe to work on
        truth_col (str): column name for the truth
        class_column_dict (dict): key = class in truth col, value = probably column associated to class
        xlim (float pair): (xmin, xmax)
        ylim (float pair): (ymin, ymax)

    """
    #
    # this makes an accuracy curve for every class in a multiclass problem
    # - does so by comparing 1 vs ALL
    # - this one is a mess, need to refactor if time
    #
    import copy
    from sklearn.metrics import f1_score, recall_score, precision_score
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    from itertools import cycle
    import numpy as np

    def _compute_accuracy(cut, yhat, y, stat="f1_score"):
        # print cut
        yhat[yhat < cut] = 0
        yhat[yhat >= cut] = 1

        # w1 = np.where(y==1)
        # w0 = np.where(y==0)

        # c_rate = np.mean( y==yhat )
        if stat == "f1_score":
            f1_s = f1_score(y, yhat)
        elif stat == "precision_score":
            f1_s = precision_score(y, yhat)
        else:
            f1_s = recall_score(y, yhat)

        return f1_s

    colormap = plt.cm.nipy_spectral  # gist_ncar #nipy_spectral, Set1,Paired

    lw = 1.5
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    #
    truth = df[truth_col].values
    class_dict, _ = _generate_class_dicts(set(truth))
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(class_dict.keys()))]

    class_names = list(class_dict.keys())
    class_names.sort()

    stats = ["f1_score", "precision_score", "recall_score"]

    max_f1_dict = {}
    for stat in stats:
        for k, color in zip(class_names, colors):
            cut_off = -1

            # don't do anything for default class
            # if k == default_class:
            #    y = [1 if x == k else 0 for x in truth]
            #    size_default = len(np.where(np.array(y)==1)[0])
            #    size_total = len(y)
            #    continue
            if k not in class_column_dict:
                print("[cl]: skipping no class in dict %s".format(k))
                continue

            # multi class either 1 or 0
            y = [1 if x == k else 0 for x in truth]
            scores = df[class_column_dict[k]].values

            # run over the bins to compute the statistics
            bins = [x * 0.01 for x in range(101)]
            f1 = []
            max_val = -1
            for i in bins:
                labels = copy.deepcopy(np.array(y))
                preds = copy.deepcopy(np.array(scores))

                current_val = _compute_accuracy(i, preds, labels, stat)
                f1.append(current_val)
                if current_val > max_val:
                    max_val = current_val
                    cut_off = i

            # sace the cut_off if we're using f1_score
            if stat == "f1_score":
                max_f1_dict[k] = cut_off

            plt.plot(bins, f1, label=k, color=color)

        ##
        fig = plt.figure(1, figsize=(7, 7))
        plt.gca().xaxis.grid(True)
        # plt.xticks(list(plt.xticks()[0]) + [0.45])
        plt.xlabel("Probability Cutoff", fontsize=16)
        plt.ylabel("Score", fontsize=16)

        plt.xlim(xlim)
        plt.ylim(ylim)
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        plt.title(stat.upper().replace("_", " ") + " [1 vs ALL]", fontsize=18)
        # plt.legend(loc="center left", fontsize=8, bbox_to_anchor=(1, 0.5), title='CLASSES [S = %d/%d]'%(size_default,size_total))
        plt.legend(
            loc="center left", fontsize=10, bbox_to_anchor=(1, 0.5), title="CLASSES"
        )

        plt.show()

    return max_f1_dict


"""
  ___  ___   ___
 | _ \/ _ \ / __|
 |   / (_) | (__
 |_|_\\___/ \___|


"""


def roc_multi(
    truth_col, prob_col, default_class=0, x_min=-0.0, x_max=1, y_min=-0.0, y_max=1.0
):
    #
    # this makes an ROC curve for every class in a multiclass problem
    # - does so by comparing 1 vs ALL
    #
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    from itertools import cycle
    import numpy as np

    colormap = plt.cm.nipy_spectral  # gist_ncar #nipy_spectral, Set1,Paired

    lw = 1.5
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = [colormap(i) for i in np.linspace(0, 0.9, len(truth_col))]

    min_tpr = []
    max_tpr = []
    # min_tpr_mean = 1000
    # max_tpr_mean = -10
    auc_max = -1
    auc_min = 10000
    for ix, (k, color) in enumerate(zip(prob_col, colors)):

        y = truth_col[ix]
        scores = k

        fpr, tpr, thresholds = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)

        # print(fpr,tpr)
        tpr = interp(mean_fpr, fpr, tpr)
        mean_tpr += tpr
        mean_tpr[0] = 0.0

        # print(np.mean(tpr) )
        if roc_auc > auc_max:
            min_tpr = tpr
            auc_max = roc_auc
        if roc_auc < auc_min:
            max_tpr = tpr
            auc_min = roc_auc

    mean_tpr /= len(prob_col)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print(mean_auc)

    fig = plt.figure(1, figsize=(5, 5))

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="k",
        linestyle="--",
        label="AUC %0.3f" % (mean_auc),
        lw=4.5,
    )
    # print(mean_fpr, min_tpr, max_tpr)
    plt.fill_between(
        mean_fpr,
        min_tpr,
        max_tpr,
        color="g",
        alpha=0.7,
        label="Min-Max AUC (%0.3f,%0.3f)" % (auc_min, auc_max),
        lw=4.5,
    )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=lw, color="b")

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel("False Positive Rate", size=15)
    plt.ylabel("True Positive Rate", size=15)
    plt.title("Averaged ROC Over N=%s" % len(truth_col))
    plt.legend(
        loc="lower right", fontsize=10
    )  # , bbox_to_anchor=(1, 0.5))#, title='CLASSES [S = %d/%d]'%(1,2))
    plt.show()


def roc(
    df,
    truth_col="y",
    class_column_dict={},
    default_class=0,
    xlim=(-0.0, 1),
    ylim=(-0.0, 1.0),
    show_individual_classes=True,
):
    """
    ROC curve with AUC numbers

    Inputs:
        df (pd.DF): dataframe to work on
        truth_col (str): column name for the truth
        class_column_dict (dict): key = class in truth col, value = probably column associated to class
        xlim (float pair): (xmin, xmax)
        ylim (float pair): (ymin, ymax)
        show_individual_classes (bool): plot individual classes on ROC
    """
    #
    # this makes an ROC curve for every class in a multiclass problem
    # - does so by comparing 1 vs ALL
    #
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    from itertools import cycle
    import numpy as np

    colormap = plt.cm.nipy_spectral  # gist_ncar #nipy_spectral, Set1,Paired

    fig = plt.figure(figsize=(5, 5))
    lw = 1.5
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    #
    truth = df[truth_col].values

    class_dict, _ = _generate_class_dicts(set(truth))
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(class_dict.keys()))]

    class_names = list(class_dict.keys())
    class_names.sort()

    auc_max = -1
    auc_min = 10000
    # print(prob_col)
    for k, color in zip(class_names, colors):
        print(k, color)

        if k not in class_column_dict:
            print("[cl]: skipping no class in dict {}".format(k))
            continue

        # multi class either 1 or 0
        y = [1 if x == k else 0 for x in truth]

        scores = df[class_column_dict[k]].values
        fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        # print(len(tpr),'asda',len(y),len(scores),len(thresholds))
        # print(tpr[-2],thresholds[-2])
        # print(fpr,tpr)

        # if roc_auc > auc_max:
        #     min_tpr = tpr
        #     auc_max = roc_auc
        # if roc_auc < auc_min:
        #     max_tpr = tpr
        #     auc_min = roc_auc

        if show_individual_classes:
            plt.plot(
                fpr,
                tpr,
                lw=lw,
                color=color,
                label="%s (%d) %0.3f"
                % (k, len(np.where(np.array(y) == 1)[0]), roc_auc),
            )

    mean_tpr /= len(class_dict.keys()) - 1
    # print(mean_tpr)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="k",
        linestyle="--",
        label="AUC %0.3f" % (mean_auc),
        lw=4.5,
    )

    # print(len(mean_fpr), len(min_tpr), len(max_tpr))
    # plt.fill_between(mean_fpr, min_tpr, max_tpr, color='g', alpha=0.7, label='Min-Max AUC (%0.3f,%0.3f)'%(auc_min,auc_max), lw=4.5)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=lw, color="b")

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title("ROC")
    plt.legend(
        loc="lower right", fontsize=12
    )  # , bbox_to_anchor=(1, 0.5))#, title='CLASSES [S = %d/%d]'%(1,2))
    plt.show()


def delta_roc(
    df,
    truth_col="y",
    class_column_dict={},
    class_column_dict2={},
    default_class=0,
    xlim=(-0.0, 1),
    ylim=(-0.5, 0.5),
    show_individual_classes=True,
):
    """
    Delta ROC curve with AUC numbers (delta True Positive rate)

    Inputs:
        df (pd.DF): dataframe to work on
        truth_col (str): column name for the truth
        class_column_dict (dict): key = class in truth col, value = probably column associated to class
        class_column_dict2 (dict): key = class in truth col, value = probably column associated to class for the second class (delta = 2 - 1)
        xlim (float pair): (xmin, xmax)
        ylim (float pair): (ymin, ymax)
        show_individual_classes (bool): plot individual classes on ROC
    """
    #
    # this makes an ROC curve for every class in a multiclass problem
    # - does so by comparing 1 vs ALL
    #
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    from itertools import cycle
    import numpy as np

    colormap = plt.cm.nipy_spectral  # gist_ncar #nipy_spectral, Set1,Paired

    fig = plt.figure(figsize=(5, 5))
    lw = 1.5
    # mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    # mean_tpr2 = 0.0
    mean_fpr2 = np.linspace(0, 1, 100)

    #
    truth = df[truth_col].values

    class_dict, _ = _generate_class_dicts(set(truth))
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(class_dict.keys()))]

    class_names = list(class_dict.keys())
    class_names.sort()

    auc_max = -1
    auc_min = 10000
    # print(prob_col)
    for k, color in zip(class_names, colors):
        print(k, color)

        if k not in class_column_dict:
            print("[cl]: skipping no class in dict {}".format(k))
            continue

        # multi class either 1 or 0
        y = [1 if x == k else 0 for x in truth]

        scores = df[class_column_dict[k]].values
        fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
        mean_tpr = interp(mean_fpr, fpr, tpr)

        # set by hand the frist value
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        # repeat for the second dictionary
        scores2 = df[class_column_dict2[k]].values
        fpr2, tpr2, thresholds2 = roc_curve(y, scores2, drop_intermediate=False)
        mean_tpr2 = interp(mean_fpr, fpr2, tpr2)
        mean_tpr2[0] = 0.0
        roc_auc2 = auc(fpr2, tpr2)

        if show_individual_classes:
            plt.plot(
                mean_fpr,
                mean_tpr2 - mean_tpr,
                lw=lw,
                color=color,
                label="%s (%d) %0.3f"
                % (k, len(np.where(np.array(y) == 1)[0]), roc_auc2),
            )

    mean_tpr /= len(class_dict.keys()) - 1
    # print(mean_tpr)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # plt.plot(mean_fpr, mean_tpr, color='k', linestyle='--',
    #     label='AUC %0.3f'%(mean_auc), lw=4.5)

    mean_tpr2 /= len(class_dict.keys()) - 1
    # print(mean_tpr)
    mean_tpr2[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_tpr2)

    plt.plot(
        mean_fpr2,
        mean_tpr2 - mean_tpr,
        color="k",
        linestyle="--",
        label="Delta AUC %0.3f" % (mean_auc2 - mean_auc),
        lw=4.5,
    )

    # print(len(mean_fpr), len(min_tpr), len(max_tpr))
    # plt.fill_between(mean_fpr, min_tpr, max_tpr, color='g', alpha=0.7, label='Min-Max AUC (%0.3f,%0.3f)'%(auc_min,auc_max), lw=4.5)
    plt.plot([0, 1], [0, 0], linestyle="--", lw=lw, color="b")

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("Delta True Positive Rate", fontsize=15)
    plt.title("Delta ROC")
    plt.legend(
        loc="lower right", fontsize=12
    )  # , bbox_to_anchor=(1, 0.5))#, title='CLASSES [S = %d/%d]'%(1,2))
    plt.show()


"""
  _ ___ _  _ ___   ___ _ ___ _____ ___
 | |  |_ _| \| | __| | _ \ |  / _ \_   _/ __|
 | |__ | || .` | _|  |  _/ |_| (_) || | \__ \
 |____|___|_|\_|___| |_| |____\___/ |_| |___/

 """


def statistic_trend(
    df,
    column_list,
    label_list=None,
    truth_col="Tag",
    stats=["f1_score", "recall_score", "precision_score", "accuracy_score"],
    int_to_str=None,
):
    """
    Plotting trend lines over iterations
    Input:
        df (pd.DataFrame): data frame where all the data lives
        column_list (str list): column names in order of display
        label_list (str list): x-axis labels (defaults to column list)
        truth_col (str)  : column where truth data is held
        stats (str or str list) : f1_score, precision_score, or recall_score
        int_to_str (dict): int to str if legend needs to be manually fixed from class labels
    """
    return statistic_trend_bags(
        [df],
        column_list=column_list,
        truth_col=truth_col,
        stats=stats,
    )


def statistic_trend_bags(
    dfs,
    column_list,
    label_list=None,
    truth_col="Tag",
    stats=["f1_score", "recall_score", "precision_score", "accuracy_score"],
    int_to_str=None,
):
    """
    Plotting trend lines over iterations
    Input:
        df (pd.DataFrame): data frame where all the data lives
        column_list (str list): column names in order of display
        label_list (str list): x-axis labels (defaults to column list)
        truth_col (str)  : column where truth data is held
        stat (str)   : f1_score, precision_score, or recall_score
        int_to_str (dict): int to str if legend needs to be manually fixed from class labels
    """
    from sklearn.metrics import (
        f1_score,
        recall_score,
        precision_score,
        accuracy_score,
        confusion_matrix,
    )
    import numpy as np
    import matplotlib.pyplot as plt
    import copy
    import pandas as pd

    colormap = plt.cm.nipy_spectral
    if type(dfs) == pd.core.frame.DataFrame:
        dfs = [dfs]

    # get the labels of the class
    labels = dfs[0][truth_col].unique()
    labels.sort()
    colors = [colormap(i) for i in np.linspace(0, 0.95, len(labels) + 1)]

    original = column_list[0]
    column_list = column_list[1:]

    if type(stats) == str:
        stats = [stats]

    fig, ax = plt.subplots(1, figsize=(5, 5))

    linestyles = ["--", "-.", ":", "-"]
    first = False
    for im, stat in enumerate(stats):
        temp_vals = []
        for df in dfs:
            if stat == "f1_score":
                st_orig = f1_score(
                    df[truth_col], df[original], average=None, labels=labels
                )
            elif stat == "recall_score":
                st_orig = recall_score(
                    df[truth_col], df[original], average=None, labels=labels
                )
            elif stat == "precision_score":
                st_orig = precision_score(
                    df[truth_col], df[original], average=None, labels=labels
                )
            elif stat == "accuracy_score":
                st_orig = [
                    accuracy_score(df[truth_col], df[original])
                ]  # np.diagonal(confusion_matrix(df[truth_col], df[original]))/ (np.diagonal(confusion_matrix(df[truth_col], df[truth_col]))+1e-10)

            else:
                raise ValueError()
            temp_vals.append(st_orig)

        # compute stats for each col
        # for each class
        statistics_mean = []
        statistics_sd = []

        val_mean = np.mean([x[0] for x in temp_vals])
        val_sd = np.std([x[0] for x in temp_vals])

        if stat != "accuracy_score":
            val_mean_2 = np.mean([x[1] for x in temp_vals])
            val_sd_2 = np.std([x[1] for x in temp_vals])

            statistics_mean.append([val_mean, val_mean_2])
            statistics_sd.append([val_sd, val_sd_2])
        else:
            statistics_mean.append([val_mean])
            statistics_sd.append([val_sd])

        for c in column_list:
            temp_vals = []
            for df in dfs:
                if stat == "f1_score":
                    st = f1_score(df[truth_col], df[c], average=None, labels=labels)
                elif stat == "recall_score":
                    st = recall_score(df[truth_col], df[c], average=None, labels=labels)
                elif stat == "precision_score":
                    st = precision_score(
                        df[truth_col], df[c], average=None, labels=labels
                    )
                elif stat == "accuracy_score":
                    st = [
                        accuracy_score(df[truth_col], df[c])
                    ]  # np.diagonal(confusion_matrix(df[truth_col], df[original]))/ (np.diagonal(confusion_matrix(df[truth_col], df[truth_col]))+1e-10)
                else:
                    raise ValueError()
                temp_vals.append(st)

            val_mean = np.mean([x[0] for x in temp_vals])
            val_sd = np.std([x[0] for x in temp_vals])

            if stat != "accuracy_score":
                val_mean_2 = np.mean([x[1] for x in temp_vals])
                val_sd_2 = np.std([x[1] for x in temp_vals])

                statistics_mean.append([val_mean, val_mean_2])
                statistics_sd.append([val_sd, val_sd_2])
            else:
                statistics_mean.append([val_mean])
                statistics_sd.append([val_sd])
        # statistics.append(st)

        # print(statistics_mean)
        # transpose such that each set
        statistics_mean = [list(i) for i in zip(*statistics_mean)]
        statistics_sd = [list(i) for i in zip(*statistics_sd)]

        # print(stat, statistics_mean)

        # make the plot
        counter = 0
        if stat != "accuracy_score":
            for s, color in zip(statistics_mean, colors):
                label_name = labels[counter]
                if int_to_str is not None:
                    print(int_to_str[labels[counter]])
                    label_name = int_to_str[labels[counter]]

                plt.plot(
                    s,
                    color=color,
                    lw=3,
                    label=stat + " " + str(label_name),
                    linestyle=linestyles[im],
                )
                counter += 1
        else:
            plt.plot(
                statistics_mean[0],
                color=colors[-1],
                lw=3,
                linestyle=linestyles[-1],
                label=stat,
            )

        if (stat == "accuracy_score" or stat == "f1_score") and len(dfs) > 1:
            if not first:
                plt.fill_between(
                    range(len(statistics_mean[0])),
                    np.array(statistics_mean[0]) - np.array(statistics_sd[0]),
                    np.array(statistics_mean[0]) + np.array(statistics_sd[0]),
                    alpha=0.1,
                    color="b",
                    label="+/- 1 s.d.",
                )
                first = True
            else:
                plt.fill_between(
                    range(len(statistics_mean[0])),
                    np.array(statistics_mean[0]) - np.array(statistics_sd[0]),
                    np.array(statistics_mean[0]) + np.array(statistics_sd[0]),
                    alpha=0.1,
                    color="b",
                )

            if stat == "f1_score":
                plt.fill_between(
                    range(len(statistics_mean[0])),
                    np.array(statistics_mean[1]) - np.array(statistics_sd[1]),
                    np.array(statistics_mean[1]) + np.array(statistics_sd[1]),
                    alpha=0.1,
                    color="b",
                )

    # plt.xlabel('Model Varieties',size=15)
    plt.ylabel("Metric", size=15)
    plt.title("Model Varieties", size=15)
    plt.grid(linestyle="dotted")
    plt.legend(loc="center left", fontsize=8, bbox_to_anchor=(1, 0.5), title="CLASSES")

    if label_list is None:
        label_list = [original] + column_list

    # ax.set_xticklabels(label_list)
    plt.xticks(range(len(label_list)), label_list, rotation="vertical", size=13)
    plt.xlim([0, len(label_list) - 1])


"""
   ___ ___  _  _ ___ _   _ ___ ___ ___  _  _   __  __   _ _____ ___ ___ ___ ___ ___
  / __/ _ \| \| | __| | | / __|_ _/ _ \| \| | |  \/  | /_\_   _| _ \_ _/ __| __/ __|
 | (_| (_) | .` | _|| |_| \__ \| | (_) | .` | | |\/| |/ _ \| | |   /| | (__| _|\__ \
  \___\___/|_|\_|_|  \___/|___/___\___/|_|\_| |_|  |_/_/ \_\_| |_|_\___\___|___|___/


 """


def cm_by_row(truth, predicted, is_recall=True):
    """
    Make bar chart of a CM matrix for more accurate visualsiation of classes

    Input:
        truth (str/int list)    : truth list of values
        predicted (str/int list): predicted list of values
        is_recall (bool)        : recall or precision flag
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np

    # convert to int and generate labels if string
    if isinstance(truth[0], str) and isinstance(predicted[0], str):
        class_dict, labels = _generate_class_dicts(set(truth))
        truth = [class_dict[x] for x in truth]
        predicted = [class_dict[x] for x in predicted if x in class_dict]

    conf_mat = confusion_matrix(truth, predicted)
    conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
    print(conf_mat)
    rc = 0
    # labels = tag_rev_class_dict
    for row in conf_mat:
        plt.figure(figsize=(6, 3))
        plt.bar(range(0, len(row)), row)
        plt.title("Actual {}".format(labels[rc]))
        plt.plot(
            (0, len(labels)), (0.6, 0.6), "k-", alpha=0.5, color="red", linewidth=5
        )

        plt.ylabel("Recall")
        plt.ylim(0, 1)
        _ = plt.xticks(
            [x + 0.5 for x in range(len(labels))],
            [labels[l] for l in labels],
            rotation=90,
            fontsize=8,
        )
        plt.xlabel("Predicted Label")
        rc += 1


def _generate_class_dicts(classes):
    """Generate class dictionary to ints and reverse dictionary of ints to class.

    Args:
        classes (str list): List of classes
    Returns:
        class_dict (dict): classes where key = (string), values = (int)
        reverse_class_dict (dict): classes where key = (int) , values = (string)
    """
    class_dict = {}
    reverse_class_dict = {}
    counter = 0
    for i in sorted(classes):
        class_dict[i] = counter
        reverse_class_dict[counter] = i
        counter += 1
    return class_dict, reverse_class_dict


def confusion_matrix(
    truth,
    predicted,
    labels={},
    title="Confusion Matrix",
    norm=1,
    suppress_values=False,
    diagonal_values=False,
    font_size=14,
    clim=(0, 1),
    cut_off=1,
    is_recall=True,
    resort="asc",
):
    """
    Input:
        truth (str/int list)    : truth list of values
        predicted (str/int list): predicted list of values
        labels (dict)         :
        title (str)   : str title
        norm (boolean): raw count true/false
        suppress_values (boolean): don't overlay all numbers
        diagonal_values (boolean): overlay just diagonal numbers
        font_size (int):
        clim (float pair): (cmin, cmax)
        cut_off (float): cut_off value above which append an * to name
        is_recall (bool): recall or precision
        resort (str)    : 'asc' for ascending sort, 'desc' for decending
    """
    # make confusion matrix from truth and predicted for classes
    # define the confusion matrix
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import numpy as np
    import matplotlib.pyplot as plt

    # If passed is not list, try to convert from pd.Series.  If
    # that fails, safely default.
    if type(truth) is not list:
        try:
            truth = truth.tolist()
        except:
            print("ERROR: truth was not list or Series!")
            return 0
    if type(predicted) is not list:
        try:
            predicted = predicted.tolist()
        except:
            print("ERROR: predicted was not list or Series!")
            return 0

    # Join predicted and truth. Gives all labels.
    joined = truth + predicted

    # convert to int and generate labels if string
    if isinstance(truth[0], str) and isinstance(predicted[0], str):
        class_dict, labels = _generate_class_dicts(set(joined))
        truth = [class_dict[x] for x in truth]
        predicted = [class_dict[x] for x in predicted]

    conf_mat = confusion_matrix(truth, predicted)
    conf_mat = (
        conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
    )  # normalized for colorbar

    # Flip the matrix if flipping label order
    if resort == "desc":
        conf_mat = np.flip(conf_mat)

    fig, ax = plt.subplots(figsize=(7, 7))
    width = np.shape(conf_mat)[1]
    height = np.shape(conf_mat)[0]

    res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation="nearest")
    cb = fig.colorbar(res)

    res.set_clim(clim[0], clim[1])

    conf_mat = confusion_matrix(truth, predicted)  # refresh for values on conf_mat plot
    # normalise
    if norm:
        if is_recall:
            conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
        else:
            conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=0)

        # Convert all NaNs to zeros
        nans = np.isnan(conf_mat)
        conf_mat[nans] = 0

    # add number overlay
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if (not suppress_values or (diagonal_values and i == j)) and c >= 0:
                cent = 0.1
                if diagonal_values:
                    cent = 0.3

                if norm:
                    d = round(c, 2)
                    plt.text(j - cent, i + 0.0, d, fontsize=12)
                else:
                    plt.text(j - cent, i + 0.0, c, fontsize=12)

            if (i == j) and c > cut_off:
                cent = 0.3
                plt.text(j - cent, i + 0.0, "*", fontsize=12)

    # set axes
    if labels == {}:
        labels = [(x, str(x)) for x in np.unique(truth)]
        labels = dict(labels)

    # Re-sort? _generate_class_dicts does asc sort
    if resort == "desc":
        _ = plt.xticks(
            range(len(labels)),
            [labels[l] for l in reversed(labels.keys())],
            rotation=90,
            fontsize=font_size,
        )
        _ = plt.yticks(
            range(len(labels)),
            [labels[l] for l in reversed(labels.keys())],
            fontsize=font_size,
        )
    else:
        _ = plt.xticks(
            range(len(labels)),
            [labels[l] for l in labels],
            rotation=90,
            fontsize=font_size,
        )
        _ = plt.yticks(
            range(len(labels)), [labels[l] for l in labels], fontsize=font_size
        )

    print(
        classification_report(
            truth, predicted, target_names=[l for l in labels.values()]
        )
    )

    plt.xlabel("Predicted", fontsize=font_size + 4)
    plt.ylabel("Truth", fontsize=font_size + 4)
    plt.title(title, fontsize=font_size + 5)

    cb.ax.get_yaxis().labelpad = 20

    heat_label = "Precision"
    if is_recall:
        heat_label = "Recall"
    cb.ax.set_ylabel(heat_label, rotation=270, size=18)


def delta_matrix(
    truth,
    predicted,
    predicted_2,
    labels={},
    save_name="",
    title="Delta Confusion Matrix",
    norm=1,
    suppress_values=True,
    diagonal_values=True,
    font_size=16,
    clim=(-0.2, 0.2),
    cut_off=1,
    norm_recall=True,
):
    """
    Input:
        truth (str/int list)    : truth list of values
        predicted (str/int list): predicted list of values
        predicted_2 (str/int list): predicted list of values (pred_2 - pred)
        labels (dict)         :
        title (str)   : str title
        norm (boolean): raw count true/false
        suppress_values (boolean): don't overlay all numbers
        diagonal_values (boolean): overlay just diagonal numbers
        font_size (int):
        c_min (float)
        c_max (float)
        cut_off (float): cut_off value above which append an * to name
        is_recall (bool): recall or precision
    """
    # make confusion matrix from truth and predicted for classes
    # define the confusion matrix
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import numpy as np

    # convert to int and generate labels if string
    if (
        isinstance(truth[0], str)
        and isinstance(predicted[0], str)
        and isinstance(predicted_2[1], str)
    ):
        class_dict, labels = _generate_class_dicts(set(truth))
        truth = [class_dict[x] for x in truth]
        predicted = [class_dict[x] if x in class_dict else "NONE" for x in predicted]
        predicted_2 = [
            class_dict[x] if x in class_dict else "NONE" for x in predicted_2
        ]

    conf_mat = confusion_matrix(truth, predicted)
    conf_mat_2 = confusion_matrix(truth, predicted_2)

    # normalise
    title_type = ""
    if norm:
        if norm_recall:
            conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
            conf_mat_2 = (
                conf_mat_2.astype("float") / conf_mat_2.sum(axis=1)[:, np.newaxis]
            )
            title_type = "Recall"
        else:
            # print('[delta_matrix]: plotting precision')
            conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=0)
            conf_mat_2 = conf_mat_2.astype("float") / conf_mat_2.sum(axis=0)
            title_type = "Precision"
    # take the delta map
    delta_conf_mat = conf_mat_2 - conf_mat

    # fig = plt.figure(figsize=(10,10))
    fig, ax = plt.subplots(figsize=(7, 7))

    width = np.shape(delta_conf_mat)[1]
    height = np.shape(delta_conf_mat)[0]

    res = plt.imshow(
        np.array(delta_conf_mat), cmap=plt.cm.RdYlGn, interpolation="nearest"
    )
    cb = fig.colorbar(res)

    res.set_clim(clim[0], clim[1])

    # add number overlay
    for i, row in enumerate(delta_conf_mat):
        for j, c in enumerate(row):
            if not suppress_values or (diagonal_values and i == j):
                cent = 0.1
                if diagonal_values:
                    cent = 0.1

                if norm:
                    d = round(c, 2)
                    plt.text(j - cent, i + 0.0, d, fontsize=font_size - 2)
                else:
                    plt.text(j - cent, i + 0.0, c, fontsize=font_size - 2)

            if (i == j) and c > cut_off:
                cent = 0.3
                plt.text(j - cent, i + 0.0, "X", fontsize=font_size - 2)

    # set axes
    if labels != {}:
        _ = plt.xticks(
            range(len(labels)),
            [labels[l] for l in labels],
            rotation=90,
            fontsize=font_size,
        )
        _ = plt.yticks(
            range(len(labels)), [labels[l] for l in labels], fontsize=font_size
        )
        print(
            classification_report(
                truth, predicted, target_names=[l for l in labels.values()]
            )
        )
        print(
            classification_report(
                truth, predicted_2, target_names=[l for l in labels.values()]
            )
        )

    plt.xlabel("Predicted", fontsize=font_size + 4)
    plt.ylabel("Truth", fontsize=font_size + 4)
    plt.title(title_type + " " + title, fontsize=font_size + 5)

    cb.ax.get_yaxis().labelpad = 30
    cb.ax.tick_params(labelsize=font_size - 1)
    if norm:
        cb.ax.set_ylabel("Delta Percentage Points", rotation=270, size=font_size + 2)
    else:
        cb.ax.set_ylabel("Delta Count", rotation=270, size=font_size + 2)

    if save_name != "":
        plt.savefig(save_name)


"""
  ___ ___ ___   _   _   _   __  ___ ___ ___ ___ ___ ___ ___ ___  _  _
 | _ \ __/ __| /_\ | |  | |   / / | _ \ _ \ __/ __|_ _/ __|_ _/ _ \| \| |
 |   / _| (__ / _ \| |__| |__   / /  |  _/   / _| (__ | |\__ \| | (_) | .` |
 |_|_\___\___/_/ \_\____|____| /_/   |_| |_|_\___\___|___|___/___\___/|_|\_|

"""


def rpc(truth, predicted, labels={}, xlim=(0, 1.1), ylim=(0, 1.1)):
    """
    plot recall (x) vs precision (y) vs count (colour)

    Input:
        truth (list str/int): list of class truths
        predicted (list str/int): list of class predictions
        labels (dict): int to str optional if truth/predicted
        xlim (float pair): (xmin,xmax)
        ylim (float pair): (ymin,ymax)
    """
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score

    import numpy as np

    # convert to int and generate labels if string
    if isinstance(truth[0], str) and isinstance(
        predicted[0], str
    ):  # and isinstance(predicted_2[1], str):
        class_dict, labels = _generate_class_dicts(set(truth))
        truth = [class_dict[x] for x in truth]
        predicted = [class_dict[x] for x in predicted if x in class_dict]
        # predicted_2 = [class_dict[x] for x in predicted_2 if x in class_dict]

    # get counts of the truth set
    y_counts_train = []
    for i in range(len(set(truth))):
        y_counts_train.append(len([x for x in truth if x == i]))

    labels = [labels[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title("Precision vs Recall", fontsize=18)
    plt.xlabel("Recall", fontsize=18)
    plt.ylabel("Precision", fontsize=18)

    # colors.LogNorm(vmin=Z1.min(), vmax=Z1.max())

    # plot the points
    from matplotlib.colors import LogNorm

    cax = plt.scatter(
        recall_score(truth, predicted, average=None),
        precision_score(truth, predicted, average=None),
        marker="o",
        s=200,
        c=y_counts_train,  # _train #y_counts
        cmap=plt.get_cmap("Spectral")
        # , norm= LogNorm(np.min(y_counts_train), vmax=np.max(y_counts_train))#, cmap='PuBu_r'
        # ,
    )

    plt.ylim(ylim[0], ylim[1])
    plt.xlim(xlim[0], xlim[1])
    cbar = plt.colorbar()

    # add labels to each point
    for label, x, y in zip(
        labels,
        recall_score(truth, predicted, average=None),
        precision_score(truth, predicted, average=None),
    ):
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(-10, 20),
            textcoords="offset points",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.2),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    # plt.plot((0, 0.6), (0.6, 0.6), 'k-',alpha=0.5,color='red',linewidth=5)
    # plt.plot((0.6, 0.6), (0, 0.6), 'k-',alpha=0.5,color='red',linewidth=5)
    circle1 = plt.Circle((0, 0), 0.6, color="r", fill=False, lw=2)
    ax.add_artist(circle1)
    circle2 = plt.Circle((0, 0), 0.8, color="g", fill=False, lw=2)
    ax.add_artist(circle2)

    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel("Number of Class Instances", rotation=270, size=18)


def delta_rpc(
    truth, predicted, predicted_2, labels={}, xlim=(-0.15, 0.15), ylim=(-0.15, 0.15)
):
    """
    plot recall (x) vs precision (y) vs count (colour) - (2nd predicted)

    Input:
        truth (list str/int): list of class truths
        predicted (list str/int): list of class predictions
        predicted_2 (list str/int): list of class predictions of improved model
        labels (dict): int to str optional if truth/predicted
        xlim (float pair): (xmin,xmax)
        ylim (float pair): (ymin,ymax)
    """
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix

    # convert to int and generate labels if string
    if (
        isinstance(truth[0], str)
        and isinstance(predicted[0], str)
        and isinstance(predicted_2[1], str)
    ):
        class_dict, labels = _generate_class_dicts(set(truth))
        truth = [class_dict[x] for x in truth]
        predicted = [class_dict[x] for x in predicted if x in class_dict]
        predicted_2 = [class_dict[x] for x in predicted_2 if x in class_dict]

    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 5))

    # get counts of the truth set
    y_counts_train = []
    for i in range(len(set(truth))):
        y_counts_train.append(len([x for x in truth if x == i]))
    print(y_counts_train)
    labels = [labels[l] for l in labels]

    # plt.figure(figsize=(20,10))
    plt.title("Precision vs Recall", fontsize=15)
    plt.xlabel("Recall", fontsize=18)
    plt.ylabel("Precision", fontsize=18)

    # colors.LogNorm(vmin=Z1.min(), vmax=Z1.max())
    # plot the points
    delta_recall = recall_score(truth, predicted_2, average=None) - recall_score(
        truth, predicted, average=None
    )
    delta_precision = precision_score(
        truth, predicted_2, average=None
    ) - precision_score(truth, predicted, average=None)

    # delta_accuracy = accuracy_score(truth, predicted_2) - accuracy_score(truth, predicted)

    # delta_accuracy = #np.diagonal(confusion_matrix(truth, predicted_2)- confusion_matrix(truth, predicted)) / (np.diagonal(confusion_matrix(truth, truth))+1e-10)

    print(recall_score(truth, predicted_2, average=None))
    print(recall_score(truth, predicted, average=None))
    # print((delta_accuracy))

    from matplotlib.colors import LogNorm

    cax = plt.scatter(
        delta_recall,
        delta_precision,
        marker="o",
        s=200,
        c=y_counts_train,  # recall_score(truth, predicted_2,average=None)  #_train #y_counts
        cmap=plt.get_cmap("Spectral")
        # , norm= LogNorm(np.min(y_counts_train),
        #   vmax=np.max(y_counts_train))#, cmap='PuBu_r'
        # , norm= LogNorm(np.min(delta_accuracy) #recall_score(truth, predicted_2,average=None) ),
        # , vmax=np.max(delta_accuracy))
        # recall_score(truth, predicted,average=None) ))#, cmap='PuBu_r'
        # ,
        # ,vmin=cmin
        # ,vmax=cmax
    )
    plt.ylim(ylim[0], ylim[1])
    plt.xlim(xlim[0], xlim[1])
    cbar = plt.colorbar()

    # cbar.ax.set_ylabel('Original Recall', rotation=270)

    # add labels to each point
    for label, x, y in zip(labels, delta_recall, delta_precision):
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(-10, 20),
            textcoords="offset points",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.2),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel("Number of Class Instances", rotation=270, size=18)
    plt.plot((0, 0.0), (0.0, 1), "k-", alpha=0.5, color="g", linewidth=5)
    plt.plot((0, 1.0), (0.0, 0), "k-", alpha=0.5, color="g", linewidth=5)
    plt.plot((0, 0.0), (0.0, -1), "k-", alpha=0.5, color="r", linewidth=5)
    plt.plot((0, -1.0), (0.0, 0), "k-", alpha=0.5, color="r", linewidth=5)
    # plt.plot((0.6, 0.6), (0, 0.6), 'k-',alpha=0.5,color='red',linewidth=5)
