import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


def plot_roc_curve(model, X, y, sample_size, n_samples, title=None, ax=None):
  """
  Plot a ROC curve with SD
  :param model: Predictive model
  :param X: Feature matrix
  :param y: Target labels
  :param sample_size: Size of each random sample of examples from X to calculate SD
  :param n_samples: Number of random samples of examples from X to calculate SD
  :param title: Figure's title
  :param ax: Figure's axis
  """
  if title is None:
    title = 'ROC Curve'
  if ax is None:
    plt_show = True
    fig, ax = plt.subplots(figsize=(8, 6))
  else:
    plt_show = False

  pool = np.arange(y.shape[0])
  tprs = []
  x_vals = np.linspace(0, 1, 100)

  for i in range(n_samples):
    sample = np.random.choice(pool, size=sample_size, replace=False)
    sample_pred_probs = model.predict_proba(X.iloc[sample, :])[:, 1]
    sample_true = y.iloc[sample]

    fpr_arr, tpr_arr, threshold = roc_curve(sample_true, sample_pred_probs)

    interp_tpr = np.interp(x_vals, fpr_arr, tpr_arr)
    interp_tpr[0] = 0
    tprs.append(interp_tpr)

  y_pred_probs = model.predict_proba(X)[:, 1]

  fpr_arr, tpr_arr, threshold = roc_curve(y, y_pred_probs)
  area = roc_auc_score(y, y_pred_probs)
  interp_tpr = np.interp(x_vals, fpr_arr, tpr_arr)
  interp_tpr[0] = 0

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(interp_tpr + std_tpr, 1)
  tprs_lower = np.maximum(interp_tpr - std_tpr, 0)

  ax.plot(x_vals, interp_tpr, color='b', label='ROC Curve (AUC = %0.2f)' % area)
  ax.plot([0,1], [0,1], '--', color='orange', label='Luck (AUC = 0.5)')
  ax.fill_between(x_vals, tprs_lower, tprs_upper, color='grey', label='SD')
  ax.set_title(title)
  ax.set_xlabel('FPR')
  ax.set_ylabel('TPR')
  ax.legend(loc='lower right')

  if plt_show:
    plt.show()


def plot_pr_curve(model, X, y, sample_size, n_samples, title=None, ax=None):
  """
    Plot a PR curve with SD
    :param model: Predictive model
    :param X: Feature matrix
    :param y: Target labels
    :param sample_size: Size of each random sample of examples from X to calculate SD
    :param n_samples: Number of random samples of examples from X to calculate SD
    :param title: Figure's title
    :param ax: Figure's axis
    """
  if title is None:
    title = 'PR Curve'
  if ax is None:
    plt_show = True
    fig, ax = plt.subplots(figsize=(8, 6))
  else:
    plt_show = False

  pool = np.arange(y.shape[0])
  precisions = []
  x_vals = np.linspace(0, 1, 100)

  for i in range(n_samples):
    sample = np.random.choice(pool, size=sample_size, replace=False)
    sample_pred_probs = model.predict_proba(X.iloc[sample, :])[:, 1]
    sample_true = y.iloc[sample]

    precision_arr, recall_arr, thresholds = precision_recall_curve(sample_true, sample_pred_probs)

    interp_precision = np.interp(x_vals, recall_arr[::-1], precision_arr[::-1])
    interp_precision[0] = 1
    precisions.append(interp_precision)

  y_pred_probs = model.predict_proba(X)[:, 1]

  precision_arr, recall_arr, thresholds = precision_recall_curve(y, y_pred_probs)
  area = average_precision_score(y, y_pred_probs)
  interp_precision = np.interp(x_vals, recall_arr[::-1], precision_arr[::-1])
  interp_precision[0] = 1

  std_precision = np.std(precisions, axis=0)
  precisions_upper = np.minimum(interp_precision + std_precision, 1)
  precisions_lower = np.maximum(interp_precision - std_precision, 0)

  ax.plot(x_vals, interp_precision, color='b', label='Avg precision score = %0.2f' % area)
  ax.fill_between(x_vals, precisions_lower, precisions_upper, color='grey', label='SD')
  ax.plot(x_vals, [(sum(y) / len(y)) for _ in x_vals], '--', color='orange', label='Baseline precision score = %0.2f' % (sum(y) / len(y)))
  ax.set_title(title)
  ax.set_xlabel('Recall')
  ax.set_ylabel('Precision')
  ax.legend(loc='upper right')

  if plt_show:
    plt.show()
