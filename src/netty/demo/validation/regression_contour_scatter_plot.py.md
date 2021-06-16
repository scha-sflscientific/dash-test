

```python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

module_path = os.path.abspath(os.path.join('../validation/'))
if module_path not in sys.path:
   sys.path.append(module_path)
from regression_contour_scatter_plot import actual_v_predictions_plot

%pwd
```




    '/Users/m_grossenbacher/Documents/sfl_scientific/projects/netty/examples'




```python
x = np.random.random(100) * 100 
y = np.random.random(100) * 100
c = np.random.randint(3, size=100)
```

### With Scatter and  Contour


```python
actual_v_predictions_plot(actual=y, # array or pd.Series
                          preds=x, # array or pd.Series
                          title='Actual vs. Prediction Plot',
                          fig_size=(7,7),
                          labels=c, # needs to be an array
                          save=False,
                          vmin=-10,
                          vmax=110,
                          scatter=True,
                          contour=True)
```

    /anaconda3/envs/vertex/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](regression_contour_scatter_plot.py_files/regression_contour_scatter_plot.py_3_1.png)


    Line of Best Fit: 		 y = -0.09191726989627072x + 51.71355578516676


### With Scatter and Without Contour


```python
actual_v_predictions_plot(actual=y,
                          preds=x,
                          title='Actual vs. Prediction Plot',
                          fig_size=(7,7),
                          labels=c,
                          save=False,
                          vmin=-10,
                          vmax=110,
                          scatter=True,
                          contour=False)
```

    /anaconda3/envs/vertex/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](regression_contour_scatter_plot.py_files/regression_contour_scatter_plot.py_5_1.png)


    Line of Best Fit: 		 y = -0.09191726989627072x + 51.71355578516676


### Without Scatter and with Contour


```python
actual_v_predictions_plot(actual=y,
                          preds=x,
                          title='Actual vs. Prediction Plot',
                          fig_size=(7,7),
                          labels=c,
                          save=False,
                          vmin=-10,
                          vmax=110,
                          scatter=False,
                          contour=True)
```

    /anaconda3/envs/vertex/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](regression_contour_scatter_plot.py_files/regression_contour_scatter_plot.py_7_1.png)


    Line of Best Fit: 		 y = -0.09191726989627072x + 51.71355578516676


# With Label Dummy Variables


```python
df = pd.DataFrame({'actual':y,'preds':x, 'label':c})
c_dummies = pd.get_dummies(c)
df.drop('label', axis=1, inplace=True)
df_dummies = pd.concat([df, c_dummies], axis=1)
df_dummies.head()
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
      <th>actual</th>
      <th>preds</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49.806111</td>
      <td>73.755892</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.873096</td>
      <td>6.449912</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27.110370</td>
      <td>22.848228</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61.988305</td>
      <td>69.668125</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.218747</td>
      <td>73.909858</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
actual_v_predictions_plot(actual=df['actual'], # array or pd.Series
                          preds=df['preds'], # array or pd.Series
                          title='Actual vs. Prediction Plot',
                          fig_size=(7,7),
                          label_dummies=df_dummies[[0,1,2]], # needs to be a pd.DataFrame
                          save=False,
                          vmin=-10,
                          vmax=110,
                          scatter=True,
                          contour=True)
```

    /anaconda3/envs/vertex/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](regression_contour_scatter_plot.py_files/regression_contour_scatter_plot.py_10_1.png)


    Line of Best Fit: 		 y = -0.09191726989627072x + 51.71355578516676



```python

```
