#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  ___ ___   _     _  _ ___ _    ___ ___ ___ 
 | __|   \ /_\   | || | __| |  | _ \ __| _ \
 | _|| |) / _ \  | __ | _|| |__|  _/ _||   /
 |___|___/_/ \_\_|_||_|___|____|_| |___|_|_\
              |___|                         											
	SFL plotting module
	
		- mainly for EDA guidance
		- some exploratory plots for class distributions
	
	SFL Scientific updated 28.Feb.18
"""


def abline(slope, intercept, labs="Perfect Predictions"):
    """
    DESC: Plot a line from slope and intercept
    INPUT: slope(float), intercept(float)
    -----
    OUTPUT: matplotlib plot with plotted line of desired slope and intercept
    """
    import matplotlib.pyplot as plt
    import numpy as np

    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--", c="b", label=labs)


def line_of_best_fit(
    x,
    y,
    ci,
    label,
    line_width=3,
    line_color="yellow",
    marker_color="orange",
    scatter=True,
):
    import seaborn as sns

    """
    DESC: Plot a line of best fit from scatter plot
    INPUT: x-coordinates(list/array), y-coordinates(list/array), confidence-interval(float), label(str)
    -----
    OUTPUT: seaborn plot with plotted line of best fit with confidence interval and equation for line of best fit
    """
    sns.regplot(
        x,
        y,
        fit_reg=True,
        scatter=scatter,
        label=label,
        ci=ci,
        scatter_kws={"color": marker_color},
        line_kws={"color": line_color, "lw": line_width},
    )
    return np.polynomial.polynomial.polyfit(x, y, 1)


def density_estimation(m1, m2, x_min, x_max, y_min, y_max, depth=100j):
    from scipy import stats

    X, Y = np.mgrid[x_min:x_max:depth, y_min:y_max:depth]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


def spline_fit(x, y, k_spline=3, s_spline=None, c_spline="k", lw_spline=3):
    from scipy.interpolate import UnivariateSpline
    import pandas as pd
    import matplotlib.pyplot as plt

    # sort x-axis, change data type to int/float
    df_tmp = (
        pd.DataFrame({"X": x, "Y": y}).sort_values(by="X").reset_index().astype(float)
    )

    # group equal x values and average over corresponding y
    df_tmp = df_tmp.groupby(["X"])["Y"].mean().reset_index()

    # raw data
    x = df_tmp["X"]
    y = df_tmp["Y"]

    #
    spl = UnivariateSpline(df_tmp["X"], df_tmp["Y"], k=k_spline, s=s_spline)
    xs = df_tmp["X"]
    plt.plot(xs, spl(xs), "b", lw=lw_spline, color=c_spline)

    return xs, spl(xs)


"""
Functions specific to text data
"""


def sort_common_words(text, ngram=1, n=10, tfidf=False):
    """
    Return incidence of the n words that appear in the highest proportion of text samples

    Input:
        text (pd.Series): text to be analyzed
        n-gram(int): n-gram to analyze (default: unigram)
        n (int): number of words to return
        tfidf (boolean): if True, use tr-idf vector instead of binary count
    Output:
        top_n_words: proportion of text samples that contain each of the top n words
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from scipy.sparse import csr_matrix

    # Transform data into vectorized word binary counts or tf-idf counts
    if tfidf == True:
        vect = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(ngram, ngram),
            stop_words="english",
            binary=True,
        )
    else:
        vect = CountVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(ngram, ngram),
            stop_words="english",
            binary=True,
        )
    word_counts = vect.fit_transform(text)
    vocab = vect.get_feature_names()
    num_entries = word_counts.shape[0]

    # Convert sparse matrix to a 1-column pandas DataFrame then to a pandas Series
    word_counts = word_counts.sum(axis=0)
    word_counts = pd.DataFrame(word_counts)
    word_counts.columns = vocab
    word_counts = word_counts.transpose()
    word_counts = word_counts.iloc[:, 0]

    # Sort by word's prevalence and convert to proportion of text entires that includes the word
    top_n_words = word_counts.nlargest(n) / num_entries

    return top_n_words


def analyze_word_similarity(text, labels, ngram=1, n=1000, relative=False, tfidf=False):
    """
    Return incidence of the n words that appear in the highest proportion of both true and false text samples

    Input:
        text (pd.Series): responses to be analyzed
        labels (pd.Series): boolean or int labels for text by which to separate
        n-gram(int): n-gram to analyze (default: unigram)
        n (int): number of words to analyze
        relative (boolean): if True, differences between True and False are calculated to be relative rather than absolute
        tfidf (boolean): if True, use tr-idf vector instead of binary count
    Output:
        top_n_words_true: proportion of text samples with true labels that contain each of the top n words
        top_n_words_false: proportion of text samples with false labels that contain each of the top n words
        top_n_diff_true: difference in proportion where true is greater than false for top n words (only words common to both)
        top_n_diff_false: difference in proportion where false is greater than true for top n words (only words common to both)
    """
    top_n_words_true = sort_common_words(text[labels == True], ngram, n, tfidf)
    top_n_words_false = sort_common_words(text[labels == False], ngram, n, tfidf)

    if relative == True:
        top_n_diff_true = (
            (top_n_words_true - top_n_words_false) / top_n_words_true * 100
        )
        top_n_diff_false = (
            (top_n_words_false - top_n_words_true) / top_n_words_false * 100
        )
    else:
        top_n_diff_true = top_n_words_true - top_n_words_false
        top_n_diff_false = top_n_words_false - top_n_words_true

    top_n_diff_true = top_n_diff_true.nlargest(n)
    top_n_diff_false = top_n_diff_false.nlargest(n)

    return top_n_words_true, top_n_words_false, top_n_diff_true, top_n_diff_false
