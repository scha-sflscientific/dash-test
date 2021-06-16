#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   ___ ___   _     ___ _    ___ _____ ___ 
 | __|   \ /_\   | _ \ |  / _ \_   _/ __|
 | _|| |) / _ \  |  _/ |_| (_) || | \__ \
 |___|___/_/ \_\ |_| |____\___/ |_| |___/
										 
											
	SFL plotting module
	
		- mainly for EDA
		- some exploratory plots for class distributions
	
	SFL Scientific updated 28.Feb.18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def correlation_bar_graph(
    df,
    var,
    top_n_feats=10,
    negative_corrs=False,
    include_self=False,
    title="Correlation Plot",
):
    """
    Function plots a bar graph with the top N highest correlations for a given variable
    Input:
        dat (pd.DataFrame): pandas DataFrame (m row by n columns)
        var (string): name of the variable for which correlations are to be calculated
        top_n_feats (int): number of features to plot, in order of highest correlation
        negative_corrs (boolean): true if the plot should have features with negative instead of positive correlation
        include_self (boolean): true if the variable itself should be included in plot (correlation of 1)
        title (str): title of the plot, default is 'Correlation Plot'
    Output:
        a ranked bar graph of top N features
    """
    # Generate pairwise correlation
    corr = df.corr()

    # If negative, invert correlation
    if negative_corrs == True:
        corr = corr * -1

    # Identify labels and values of top n features
    # If including the variable itself, make adjustments
    if include_self == False:
        objects = corr.loc[var].nlargest(top_n_feats + 1).index[1 : top_n_feats + 1]
    else:
        objects = corr.loc[var].nlargest(top_n_feats).index
    y_pos = np.arange(len(corr.loc[var, objects]))
    values = corr.loc[var, objects]

    # Set up the figure
    fig = plt.gcf()
    # If plot is too small, adjust here: fig.set_size_inches(18.5, 10.5)

    # Plot bars, labels, and titles
    plt.barh(y_pos, values, align="center", alpha=0.7)
    plt.yticks(y_pos, objects, fontsize=9)
    if negative_corrs == True:
        plt.xlabel("Negative Correlation")
    else:
        plt.xlabel("Correlation")
    plt.title(title)
    plt.show()


def correlation_matrix(df, size=7, title="", v_min=-0.5, v_max=0.5):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    Input:
        df: pandas DataFrame (m row by n columns)
        size: vertical and horizontal size of the plot
        title: title of plot
        v_min: lowest correlation on scale
        v_max: highest correlation on scale
    Output:
        n by n correlation matrix
    """
    # Generate pairwise correlation
    corr = df.corr()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, vmin=v_min, vmax=v_max, cmap="jet", aspect="auto")

    plt.xticks(range(len(corr.columns)), corr.columns, rotation="vertical")
    plt.yticks(range(len(corr.columns)), corr.columns)

    # Add legend bar and title
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel(title, rotation=270, size=20, labelpad=20)

    plt.show()


def hist2D(df, cols=[], labels=[], save_name=""):
    """
    Plot 2D Histogram from df and cols
    Input:
        df(pd.DataFrame): Input Dataframe
        cols(list): A list of 2 columns:[col1,col2]
        labels(list): A list of 2 labels: [label1,label2]
    Output:
        matplotlib 2D Histogram plot
    """

    # Generate frequency counts
    freq = df.groupby(cols).size().unstack(fill_value=0)
    piv = freq.T

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(piv, cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_yticks(range(len(piv.index)))
    ax.set_xticklabels(piv.columns, rotation=90)
    ax.set_yticklabels(piv.index)
    ax.set_xlabel(labels[0], fontsize=14)
    ax.set_ylabel(labels[1], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(labels[0] + " vs. " + labels[1], fontsize=16)

    # Add bar legend
    for y in range(piv.shape[0]):
        for x in range(piv.shape[1]):
            if piv.values[y, x] > 0:
                plt.text(
                    x,
                    y,
                    "%.0f" % piv.values[y, x],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=12,
                )
    plt.show()

    # Save figure if desired
    if save_name:
        fig.savefig(save_name)


def spaghetti_plot(
    series1,
    series2,
    labels=["series1", "series2"],
    xtitle=None,
    y1title=None,
    y2title=None,
    y1_lim=None,
    y2_lim=None,
    title=None,
    save_name=None,
):
    """
    Plot two series over time
    Input:
        series1, series2 (pd.Series): Input series to be plotted
        labels(list of strings): Labels that will appear in legend
        xtitle, y1title, y2title, y1_lim, y2_lim, title: Graph parameters
        save_name: If specified, saves under that name to working directory
    Output:
        matplotlib spaghetti plot
    """
    # Create plot
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(series1, label=labels[0])
    ax.legend(loc=1)

    ax2 = ax.twinx()
    ax2.plot(series2, "pink", label=labels[1])
    ax2.legend(loc=2)

    if y1_lim:
        ax.set_ylim((0, y1_lim))
    if y2_lim:
        ax2.set_ylim((0, y2_lim))

    # Add axis labels and title
    ax.set_xlabel(xtitle, fontsize=14)
    ax.set_ylabel(y1title, fontsize=14)
    ax2.set_ylabel(y2title, fontsize=14)
    plt.title(title, fontsize=16)
    plt.show()

    # Save figure if desired
    if save_name:
        fig.savefig(save_name)


def violin_plot(x, y, xlab, ylab, title):
    """
    Creates a two-class violin plot given labels x and values y

    Inputs:
        x (pd.Series): labels by which to separate values
        y (pd.Series): values to plot
        xlab, ylab, title (strings): axes labels and title

    Outputs:
        Two-class violin plot to compare values
    """
    import seaborn as sns

    # Construct temporary dataframe to plot
    dfPlot = pd.DataFrame()
    dfPlot["xlabs"] = x
    dfPlot["yvals"] = y

    # Construct plot and add labels
    sns.violinplot(y="yvals", x="xlabs", data=dfPlot, split=True, inner="quart")
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.title(title, fontsize=15)
    plt.show()


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


def word_plot(words, title="Title", xlabel="Axis title", xscale=1.0, yscale=1.0):
    """
    Plots bar chart of words

    Input:
        words (pd.Series): index of series are the words, values are their incidence
        title, xlabel (strings): Title and axis label of chart
        xscale, yscale (floats): Multipliers of plot size
    Output:
        bar chart of words
    """

    objects = words.index[::-1]
    y_pos = np.arange(len(words))
    values = words[::-1]

    fig = plt.gcf()
    fig.set_size_inches(11 * xscale, 8 * yscale)

    plt.barh(y_pos, values, align="center", alpha=0.7)
    plt.yticks(y_pos, objects, fontsize=9)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


"""
Below this line, no examples yet constructed - more cleaning / information would be helpful
"""


def get_cmap(n, name="hsv"):
    """
    Returns a function that maps each index to a distinct RGB color
    Input:
        n: n-1 indices are mapped to a distinct color
        name: the keyword argument name must be a standard mpl colormap name
    Output:
        color mapping
    """
    return plt.cm.get_cmap(name, n)


def get_color_list(n):
    """
    Returns list of n colors using get_cmap
    """
    cmap = get_cmap(n)
    color_list = []
    for i in range(n):
        color_list.append(cmap(i))
    return color_list


# def stack_spaghetti_plot():
# 	'''
# 	CLEAN ME
# 	'''

# 	cmap = plt.cm.get_cmap("Blues", 6)

# 	fig = plt.figure(figsize=(10,10),frameon=True,edgecolor='grey')

# 	ax = plt.subplot(212)
# 	#ax = fig.add_subplot(111)
# 	x = range(1,5)

# 	y = y_new_user_b1[1:]
# 	plt.plot(x,y,c=cmap(5),alpha=0.9,label = '2014')
# 	plt.text(x[-1], y[-1], 'New User in Certificate Bucket',fontsize = 20,color =cmap(5),alpha=0.9)


# 	y = y_new_user_b2[1:]
# 	plt.plot(x,y,c=cmap(4),alpha=0.9,label = '2014')
# 	plt.text(x[-1], y[-1], 'New User in Subscription Bucket',fontsize = 20,color =cmap(4),alpha=0.9)
# 	#plt.plot(range(1,9),revenue_df[list_2017].sum().values,c=cmap(4),label = '2017')

# 	y = y_new_user_b3[:]
# 	plt.plot(x,y,c=cmap(3),alpha=0.9,label = '2014')
# 	plt.text(x[-1], y[-1], 'New User in Feed Mill Bucket',fontsize = 20,color =cmap(3),alpha=0.9)

# 	y = y_new_user_b4[1:]
# 	plt.plot(x,y,c=cmap(2),alpha=0.9,label = '2014')
# 	plt.text(x[-1], y[-1]+50, 'New User in Lab Bucket',fontsize = 20,color =cmap(2),alpha=0.9)

# 	y = y_new_user_b5[1:]
# 	plt.plot(x,y,c=cmap(1),alpha=0.9,label = '2014')
# 	plt.text(x[-1], y[-1], 'New User in Other Bucket',fontsize = 20,color =cmap(1),alpha=0.9)

# 	plt.xticks(range(1,5),x_new_user[1:],fontsize = 20,color = 'grey')
# 	plt.yticks(fontsize = 20,color = 'grey')


# 	ax.spines['top'].set_visible(False)
# 	ax.spines['right'].set_visible(False)
# 	ax.spines['bottom'].set_visible(True)
# 	ax.spines['left'].set_visible(True)

# 	ax.spines['bottom'].set_color('grey')
# 	ax.spines['bottom'].set_color('grey')
# 	plt.ylabel('Counts',fontsize=24)
# 	plt.xlabel('Year',fontsize=24)


# 	ax2 = plt.subplot(211, sharex=ax)
# 	y = y_active_user
# 	plt.plot(x,y,c=cmap(6),alpha=0.9,label = '2014')
# 	plt.text(x[-1], y[-1], 'Active User',fontsize = 20,color =cmap(6),alpha=0.9)
# 	ax2.spines['top'].set_visible(False)
# 	ax2.spines['right'].set_visible(False)
# 	ax2.spines['bottom'].set_visible(True)
# 	ax2.spines['left'].set_visible(True)
# 	ax2.spines['bottom'].set_color('grey')
# 	plt.yticks(fontsize = 20,color = 'grey')
# 	plt.xticks(range(1,5),x_new_user[1:],fontsize = 20,color = 'grey')

# 	plt.ylabel('Counts',fontsize=24)
# 	#plt.xlabel('Year',fontsize=24)
# 	plt.title('Active Customer and New Customer in Each Buckets',fontsize=28)

# 	plt.savefig('images/Active Customer and New Customer in Each Buckets.png',bbox_inches='tight')
# 	plt.show()


# def Y2Y_analysis(df,group,col,output='sum',
# 	xlabel=None,ylabel=None,title=None,legends=[],save_name=None):

#     '''
#     CLEAN ME
#     '''
#     if output=='sum':
#     	temp = df.groupby(group)[col].sum().reset_index(level=1)
#     if output=='mean':
#     	temp = df.groupby(group)[col].sum().reset_index(level=1)

#     index_list= temp.index.unique().tolist()

#     plt.figure(figsize=(12,10))
#     ############
#     ## To Do: ##
#     ############

#     '''
#     Instead of spaghetti plots; doing nice and well orgnized Y2Y plots
#     '''
#     for d in index_list:
#         plt.plot(range(1,6),df.loc[d][col])

#     plt.legend(legends)
#     #plt.legend(district)
#     plt.xticks(range(1,6),[2013,2014,2015,2016,2017],fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlabel(xlabel,fontsize=16)
#     plt.ylabel(ylabel,fontsize=16)
#     plt.title(title,fontsize=16)
#     plt.savefig('images/suburban segment2 suburban.png',bbox_inches='tight')
#     plt.show()

# def horizontal_bar(data,segments=None,labels=[],legends=[],title=None,xlabel=None,save_name=None):

# 	'''
# 	Example input:
# 		segments = 4
# 		labels = ('2014-2015','2015-2016','2016-2017')
# 		data = 3 + 10* np.random.rand(segments, len(people))
# 		d1 = [88,645,819,97];d2=[242,1411,170,100];d3=[688,1207,197,0]
# 		data[:,0]=d1;data[:,1]=d2;data[:,2]=d3
# 		percentages = data/data.sum(axis=0)
# 		percentages = percentages*100
# 		percentages = np.transpose(percentages)
# 	'''
# 	y_pos = np.range(len(labels))

# 	fig = plt.figure(figsize=(10,8))
# 	ax = fig.add_subplot(111)
# 	colors = plt.cm.get_cmap("Blues", segments+1) #'rgbwmc'

# 	patch_handles = []
# 	left = np.zeros(len(labels)) # Left alignment of data starts at zero
# 	for i, d in enumerate(data):
# 	    patch_handles.append(ax.barh(y_pos, d,
# 	      color=colors(i+1), align='center',
# 	      left=left))
# 	    # Accumulate the left-hand offsets
# 	    left += d

# 	# Go through all of the bar segments and annotate
# 	for j in xrange(len(patch_handles)):
# 	    for i, patch in enumerate(patch_handles[j].get_children()):
# 	        bl = patch.get_xy()
# 	        x = 0.5*patch.get_width() + bl[0]
# 	        y = 0.5*patch.get_height() + bl[1]
# 	        ax.text(x,y, "%d%%" % (percentages[i,j]), ha='center',fontsize=14)

# 	ax.set_yticks(y_pos)
# 	plt.xticks(fontsize = 16,color = 'black')
# 	ax.set_yticklabels(labels,fontsize=16)
# 	plt.xlabel(xlabel,fontsize=16)
# 	plt.legend(legends,fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
# 	plt.title(title,fontsize=20)
# 	if save_name:
# 		plt.savefig(save_name,bbox_inches='tight')
