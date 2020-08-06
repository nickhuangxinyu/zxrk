import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from xgboost import plot_importance
import itertools
from tensorflow.keras import Model
import xgboost

def GetMidRes(model, layer_name, x):
  mid_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
  return mid_model.predict(x)

def PlotRoc(y_test, y_pred):
  from sklearn.metrics import roc_curve, auc
  fpr, tpr, thr = roc_curve(y_test, y_pred)
  roc_auc = auc(fpr, tpr)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC curve')
  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.legend()
  plt.show()

def PlotFactorCorr(x, y, topn=20, ax=None):
  corr = {}
  for i in range(len(x[0])):
    c = pd.Series(x[:, i]).corr(pd.Series(y))
    if not np.isnan(c):
      corr[i] = (c)
  if ax == None:
    plt.title('corr hist between xi and y')
    plt.hist(corr.values())
    plt.grid()
    plt.show()
  else:
    ax.set_title('corr between xi and y')
    ax.hist(corr.values())
    ax.grid()
  return corr

def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    classes = range(cm.shape[0])
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def ClsReport(model, y_test, y_pred, binary=False):
  cm = confusion_matrix(y_test, y_pred)
  plot_confusion_matrix(cm)
  target_names = ['class%d'%(i) for i in range(cm.shape[0])]
  print(classification_report(y_test, y_pred, target_names=target_names))
  if binary:
    PlotRoc(y_test, y_pred)
  if isinstance(model, xgboost.sklearn.XGBClassifier):
    ax = plot_importance(model, max_num_features=15)
    #return [yt._text for yt in ax.yticklabels]
    return [yt.label._text for yt in ax.yaxis.majorTicks]
  return None
