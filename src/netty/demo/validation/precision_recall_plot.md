

```python
import matplotlib.pyplot as plt
import numpy as np
```

### Helper function to calculate precision recall curve


```python
def calc_precision_recall_curve(y_pred, y_true, threshold_values):
    list_precision = []
    list_recall = []
    list_thresh = []
    for thresh in threshold_values:
        y_correct = (y_pred > thresh) & y_true
        precision = sum(y_correct)/sum(y_pred > thresh)
        recall = sum(y_correct)/sum(y_true)
        list_precision.append(precision)
        list_recall.append(recall)
        list_thresh.append(thresh)
    return list_precision, list_recall, list_thresh
```

### Generate sample data


```python
y_true = np.random.uniform(0,1,200)>0.5
y_pred_1 = (y_true + np.random.uniform(0,1,200)*2)/3
y_pred_2 = (y_true + np.random.uniform(0,1,200)*1.2)/2.2

precision1, recall1, _ = calc_precision_recall_curve(y_pred_1, y_true, np.arange(0.0,1.0,0.01))
precision2, recall2, _ = calc_precision_recall_curve(y_pred_2, y_true, np.arange(0.0,1.0,0.01))
```

### Use case 1: Illustrate and validate the selection of decision boundary


```python
plt.rcParams['font.size'] = 18
fig, ax = plt.subplots(figsize=(10,8))
plt.plot(recall1, precision1)
plt.scatter(recall1[30], precision1[30], s=100, c='darkorange', label='current threshold (0.30)')
plt.scatter(recall1[61], precision1[61], s=100,c='limegreen', label='optimal threshold (0.62)')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
plt.grid()
plt.legend()
plt.title('Precision-recall curve and threshold selection')
```




    Text(0.5, 1.0, 'Precision-recall curve and threshold selection')




![png](precision_recall_plot_files/precision_recall_plot_6_1.png)


### Use case 2: Compare the performance of two models


```python
plt.rcParams['font.size'] = 18
fig, ax = plt.subplots(figsize=(10,8))
plt.plot(recall1, precision1, label='model 1')
plt.plot(recall2, precision2, label='model 2')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
plt.grid()
plt.legend()
plt.title('Precision-recall curves of two models')
```




    Text(0.5, 1.0, 'Precision-recall curves of two models')




![png](precision_recall_plot_files/precision_recall_plot_8_1.png)

