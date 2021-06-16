

```python
import sys
sys.path.append('../')

# library to check
import validation
import validation.classification_plots as cpl
```


```python
%config InlineBackend.figure_format='retina'
%matplotlib inline
```


```python
#import matplotlib_defaults
```


```python
import pandas as pd
import numpy as np
```


```python

df = pd.DataFrame(np.random.random(size=(100, 8)), 
                  columns=list('ABCDEFGH'))

```


```python
df['class_g'] = df['G'].apply(lambda x: 'GOOD' if x > 0.5 else 'BAD' )
df['class_f'] = df['F'].apply(lambda x: 'GOOD' if x > 0.5 else 'BAD' )
df['class_h'] = df['H'].apply(lambda x: 'GOOD' if x > 0.5 else 'BAD' )
df['class_truth'] = df['G'].apply(lambda x: 'GOOD' if x > 0.5 else 'BAD' )
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>class_g</th>
      <th>class_f</th>
      <th>class_h</th>
      <th>class_truth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.229812</td>
      <td>0.680561</td>
      <td>0.795325</td>
      <td>0.352670</td>
      <td>0.632187</td>
      <td>0.431413</td>
      <td>0.102126</td>
      <td>0.117515</td>
      <td>BAD</td>
      <td>BAD</td>
      <td>BAD</td>
      <td>BAD</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.042678</td>
      <td>0.876537</td>
      <td>0.743886</td>
      <td>0.417394</td>
      <td>0.533802</td>
      <td>0.062206</td>
      <td>0.835067</td>
      <td>0.727176</td>
      <td>GOOD</td>
      <td>BAD</td>
      <td>GOOD</td>
      <td>GOOD</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.574522</td>
      <td>0.511271</td>
      <td>0.133636</td>
      <td>0.805165</td>
      <td>0.234899</td>
      <td>0.095265</td>
      <td>0.282009</td>
      <td>0.970655</td>
      <td>BAD</td>
      <td>BAD</td>
      <td>GOOD</td>
      <td>BAD</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.372001</td>
      <td>0.472355</td>
      <td>0.705815</td>
      <td>0.346783</td>
      <td>0.622888</td>
      <td>0.960432</td>
      <td>0.712771</td>
      <td>0.544330</td>
      <td>GOOD</td>
      <td>GOOD</td>
      <td>GOOD</td>
      <td>GOOD</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.180839</td>
      <td>0.941374</td>
      <td>0.542609</td>
      <td>0.406328</td>
      <td>0.417875</td>
      <td>0.488931</td>
      <td>0.551878</td>
      <td>0.624694</td>
      <td>GOOD</td>
      <td>BAD</td>
      <td>GOOD</td>
      <td>GOOD</td>
    </tr>
  </tbody>
</table>
</div>



## Statistic Trends
def statistic_trend(df, column_list, label_list = None, truth_col='Tag', stat='f1_score', int_to_str={}):
    """
    Plotting trend lines over iterations
    Input:
        df (pd.DataFrame): data frame where all the data lives
        column_list (str list): column names in order of display
        label_list (str list): x-axis labels (defaults to column list)
        truth_col (str)  : column where truth data is held
        stat (str)       : f1_score, precision_score, or recall_score
        int_to_str (dict): int to str if legend needs to be manually fixed from class labels
    """

```python
cpl.statistic_trend(df, 
                    column_list = ['class_g', 'class_h', 'class_f',  'class_f', 'class_f'], 
                    truth_col='class_truth',
                    stats=['f1_score','accuracy_score', 'recall_score']
                   )
```


![png](classification_plots.py_files/classification_plots.py_8_0.png)

def statistic_trend_bags(df, column_list, label_list = None, truth_col='Tag', stat='f1_score', int_to_str={}):
    """
    Plotting trend lines over iterations
    Input:
        dfs (pd.DataFrame or list pd.df): data frame where all the data lives, the list of dfs used to generate error bands
        column_list (str list): column names in order of display
        label_list (str list): x-axis labels (defaults to column list)
        truth_col (str)  : column where truth data is held
        stat (str)       : f1_score, precision_score, or recall_score
        int_to_str (dict): int to str if legend needs to be manually fixed from class labels
    """

```python
cpl.statistic_trend_bags([df, df, df], 
                    column_list = ['class_g', 'class_h', 'class_f',  'class_f', 'class_f'], 
                    truth_col='class_truth',
                    stats=['f1_score','accuracy_score'], 
                   )
```


![png](classification_plots.py_files/classification_plots.py_10_0.png)


## Confusion Matrix
def confusion_matrix(truth, 
                    predicted, 
                    labels={}, 
                    title='Confusion Matrix', 
                    norm=1, 
                    suppress_values=False,
                    diagonal_values=False,
                    font_size=14,
                    cmin=0,cmax=1,
                    cut_off = 1,
                    is_recall=True
                    ):
    """
    Input: 
        truth (str/int list)    : truth list of values
        predicted (str/int list): predicted list of values
        labels (dict)           : 
        title (str)   : str title 
        norm (boolean): raw count true/false
        suppress_values (boolean): don't overlay all numbers
        diagonal_values (boolean): overlay just diagonal numbers
        font_size (int): 
        clim (float pair): (cmin, cmax)
        cut_off (float): cut_off value above which append an * to name
        is_recall (bool): recall or precision
    """

```python
cpl.confusion_matrix(df['class_truth'], 
                     df['class_f'],
                     norm=True,
                     cut_off=0.3,
                     #clim=(5,30)
                    )
```

                  precision    recall  f1-score   support
    
             BAD       0.40      0.43      0.41        42
            GOOD       0.56      0.53      0.55        58
    
       micro avg       0.49      0.49      0.49       100
       macro avg       0.48      0.48      0.48       100
    weighted avg       0.49      0.49      0.49       100
    



![png](classification_plots.py_files/classification_plots.py_13_1.png)



```python
# Confusion matrix with "OKAY" class in Truth but not Preds, and sorted in reverse.
cpl.confusion_matrix(df['class_truth'].append(pd.Series(["OKAY"])), 
                     df['class_f'].append(pd.Series(["BAD"])),
                     norm=True,
                     #norm=False,
                     cut_off=0.3,
                     #clim=(5,30),
                     resort='desc'
                    )
```

    /Users/yuanjieli/Environments/ml/lib/python2.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


                  precision    recall  f1-score   support
    
             BAD       0.39      0.43      0.41        42
            GOOD       0.56      0.53      0.55        58
            OKAY       0.00      0.00      0.00         1
    
       micro avg       0.49      0.49      0.49       101
       macro avg       0.32      0.32      0.32       101
    weighted avg       0.49      0.49      0.49       101
    



![png](classification_plots.py_files/classification_plots.py_14_2.png)

def delta_matrix(truth, 
                predicted, 
                predicted_2, 
                labels={}, 
                save_name='',
              title='Delta Confusion Matrix', 
              norm=1, 
              suppress_values=True,
              diagonal_values=True,
              font_size=16,
              clim=(-0.2,0.2),
              cut_off = 1,
              norm_recall = True
                    ):
    """
    Input: 
        truth (str/int list)    : truth list of values
        predicted (str/int list): predicted list of values
        predicted_2 (str/int list): predicted list of values (pred_2 - pred)
        labels (dict)           : 
        title (str)   : str title 
        norm (boolean): raw count true/false
        suppress_values (boolean): don't overlay all numbers
        diagonal_values (boolean): overlay just diagonal numbers
        font_size (int): 
        clim (float pair): (cmin, cmax)
        cut_off (float): cut_off value above which append an * to name
        is_recall (bool): recall or precision
    """

```python
cpl.delta_matrix(df['class_truth'], 
                 df['class_f'],
                 df['class_h'],
                 norm=True,
                 cut_off=0.3,
                 clim=(-0.5,0.5)
                )
```

                 precision    recall  f1-score   support
    
            BAD       0.51      0.40      0.45        52
           GOOD       0.47      0.58      0.52        48
    
    avg / total       0.49      0.49      0.49       100
    
                 precision    recall  f1-score   support
    
            BAD       0.56      0.52      0.54        52
           GOOD       0.52      0.56      0.54        48
    
    avg / total       0.54      0.54      0.54       100
    



![png](classification_plots.py_files/classification_plots.py_16_1.png)

def cm_by_row(truth, predicted, is_recall=True):
    """
        Make bar chart of a CM matrix for more accurate visualsiation of classes

        Input: 
            truth (str/int list)    : truth list of values
            predicted (str/int list): predicted list of values
            is_recall (bool)        : recall or precision flag
    """

```python
cpl.cm_by_row(df['class_truth'], df['class_f'])
```

    [[0.40384615 0.59615385]
     [0.41666667 0.58333333]]



![png](classification_plots.py_files/classification_plots.py_18_1.png)



![png](classification_plots.py_files/classification_plots.py_18_2.png)


# Recall Precision Count plots
def rpc(truth,predicted,labels={}, xlim=(0,1.1), ylim=(0,1.1)):
    """
        plot recall (x) vs precision (y) vs count (colour)
        
        Input:
            truth (list str/int): list of class truths
            predicted (list str/int): list of class predictions
            labels (dict): int to str optional if truth/predicted 
            xlim (float pair): (xmin,xmax)
            ylim (float pair): (ymin,ymax)
    """

```python
cpl.rpc(df['class_truth'], 
        df['class_f'],
        labels={})
```


![png](classification_plots.py_files/classification_plots.py_21_0.png)

def delta_rpc(truth, predicted, predicted_2, labels={}, xlim=(-0.15, 0.15), ylim=(-0.15, 0.15)):
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

```python
cpl.delta_rpc(df['class_truth'], 
              df['class_f'],
              df['class_h'],
              xlim=(-0.5, 0.5),
              ylim=(-0.5, 0.5),
        labels={})
```

    [52, 48]
    [0.51923077 0.5625    ]
    [0.40384615 0.58333333]



![png](classification_plots.py_files/classification_plots.py_23_1.png)



```python

```

# ROC
def roc(df, 
    truth_col='y', 
    class_column_dict = {}, 
    default_class = 0, 
    xlim=(-0.0, 1), 
    ylim=(-0.0, 1.0), 
    show_indivual_classes=True):
    """
    ROC curve with AUC numbers
    
    Inputs:
        df (pd.DF): dataframe to work on
        truth_col (str): column name for the truth
        class_column_dict (dict): key = class in truth col, value = probably column associated to class
        xlim (float pair): (xmin, xmax)
        ylim (float pair): (ymin, ymax)
        show_indivual_classes (bool): plot individ classes on ROC 
    """

```python
temp = ['GOOD'] * 30 + ['BAD'] * 30  + ['AVERAGE'] *40

# set up some tempory columns for tests
df['Probability GOOD'] = df['A']
df['Probability BAD'] = df['B']
df['Probability AVERAGE'] = df['C']

# notice the column names and the class labels have to match to be picked up
df['class_gba_test'] = ['GOOD'] * 30 + ['BAD'] * 30  + ['AVERAGE'] *40

df.head()
```


```python
class_column_dict = {}
class_column_dict['GOOD'] = 'A'
class_column_dict['BAD'] = 'B'
#class_column_dict['AVERAGE'] = 'Probability AVERAGE'
```


```python
class_column_dict2 = {}
class_column_dict2['GOOD'] = 'C'
class_column_dict2['BAD'] = 'D'
#class_column_dict['AVERAGE'] = 'Probability AVERAGE'
```


```python
from imp import reload
```


```python
reload(cpl)
```




    <module 'validation.classification_plots' from '../validation/classification_plots.py'>




```python
reload(cpl)
```




    <module 'validation.classification_plots' from '../validation/classification_plots.py'>




```python
cpl.roc(df, 
        truth_col='class_gba_test', 
        class_column_dict=class_column_dict,
        show_indivual_classes=True
        )
```

    AVERAGE (0.0, 0.0, 0.0, 1.0)
    [cl]: skipping no class in dict AVERAGE
    BAD (0.0, 0.6026137254901961, 0.0, 1.0)
    GOOD (0.8640843137254902, 0.0, 0.0, 1.0)



![png](classification_plots.py_files/classification_plots.py_34_1.png)

def delta_roc(df, 
    truth_col='y', 
    class_column_dict = {}, 
    class_column_dict2 = {},
    default_class = 0, 
    xlim=(-0.0, 1), 
    ylim=(-0.5, 0.5), 
    show_indivual_classes=True):
    """
    Delta ROC curve with AUC numbers (delta True Positive rate)
    
    Inputs:
        df (pd.DF): dataframe to work on
        truth_col (str): column name for the truth
        class_column_dict (dict): key = class in truth col, value = probably column associated to class
        class_column_dict2 (dict): key = class in truth col, value = probably column associated to class for the second class (delta = 2 - 1)
        xlim (float pair): (xmin, xmax)
        ylim (float pair): (ymin, ymax)
        show_indivual_classes (bool): plot individ classes on ROC 
    """

```python
cpl.delta_roc(df, 
        truth_col='class_gba_test', 
        class_column_dict=class_column_dict,
        class_column_dict2=class_column_dict2,
        show_indivual_classes=True
        )
```

    AVERAGE (0.0, 0.0, 0.0, 1.0)
    [cl]: skipping no class in dict AVERAGE
    BAD (0.0, 0.6026137254901961, 0.0, 1.0)
    GOOD (0.8640843137254902, 0.0, 0.0, 1.0)



![png](classification_plots.py_files/classification_plots.py_36_1.png)


# Logit Cut-off Analysis
def logit_sp(truth, 
             predicted_probability, 
             default_class,
             xlim = (0,1)):
    """
    logit plot for thresholding - sensitivity, specificity and f1
    
    Inputs:
        truth (str list): class labels in truth
        predicted_probability (float list): probability of class
        default_class (str): value which should be considered 0
        xlim (float pair): xmin, xmax
    """

```python
cpl.logit_sp(
        df['class_truth'].values, 
        df['G'].values,
        default_class='BAD')
```

    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2920: RuntimeWarning: Mean of empty slice.
      out=out, **kwargs)
    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/numpy/core/_methods.py:85: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)



![png](classification_plots.py_files/classification_plots.py_39_1.png)

def logit(truth, 
         predicted_probability, 
         default_class,             
         xlim = (0,1)):
    """
    logit plot for thresholding - precision,recall,accuracy and f1
    
    Inputs:
        truth (str list): class labels in truth
        predicted_probability (float list): probability of class
        default_class (str): value which should be considered 0
        xlim (float pair): xmin, xrange
    """

```python
cpl.logit(
        df['class_truth'].values, 
        df['F'].values,
default_class='BAD')
```

    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)



![png](classification_plots.py_files/classification_plots.py_41_1.png)

def accuracy_cut(df, 
    truth_col='class_gba_test', 
    class_column_dict=class_column_dict, 
    xlim=(-0.01, 1.05), 
    ylim=(-0.05, 1.05)):
    """
    plot accuracy cutoff based on f1_score (precision and recall) for all classes
    
    Inputs:
        df (pd.DF): dataframe to work on
        truth_col (str): column name for the truth
        class_column_dict (dict): key = class in truth col, value = probably column associated to class
        xlim (float pair): (xmin, xmax)
        ylim (float pair): (ymin, ymax)
    
    """

```python
cpl.accuracy_cut(df, 
    truth_col='class_gba_test', 
    class_column_dict=class_column_dict)
```

    [cl]: skipping no class in dict %s


    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)



![png](classification_plots.py_files/classification_plots.py_43_2.png)


    [cl]: skipping no class in dict %s


    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)



![png](classification_plots.py_files/classification_plots.py_43_5.png)


    [cl]: skipping no class in dict %s



![png](classification_plots.py_files/classification_plots.py_43_7.png)





    {'BAD': 0.25, 'GOOD': 0.05}


def probability_distribution(df, truth_col, class_column_dict, xlim=(0,1), ylim=(0,10), normed=False, fill=True):

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

```python
cpl.probability_distribution(df,
                             truth_col='class_gba_test', 
                             class_column_dict=class_column_dict, 
                             fill=True)
```

    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6571: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "



![png](classification_plots.py_files/classification_plots.py_45_1.png)



```python

```


```python

```
