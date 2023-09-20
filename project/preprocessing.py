import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def features_with_low_correlation_with_target_to_drop(X_train, y_train, corrs, threshold):
  return [col for col in X_train.columns[np.abs(corrs) < threshold] if col not in ['age', 'gender']]


def features_with_high_correlation_with_each_other_to_drop(X_train, corrs, threshold, cols_to_ignore=[]):
  corrs = corrs.drop(['age', 'gender'], axis=0, inplace=False)
  corrs.drop(['age', 'gender'], axis=1, inplace=True)
  corrs.drop(cols_to_ignore, axis=0, inplace=True)
  corrs.drop(cols_to_ignore, axis=1, inplace=True)

  high_corr_cols = []
  above_threshold = {}
  for i in range(corrs.shape[0]):
    for j in range(i+1, corrs.shape[1]):
      if corrs.iloc[i, j] > threshold:
        if i not in above_threshold:
          above_threshold[i] = set()
        above_threshold[i].add(j)
        if j not in above_threshold:
          above_threshold[j] = set()
        above_threshold[j].add(i)

  while len(above_threshold) > 0:
    min_apps = min(above_threshold.items(), key=lambda item: len(item[1]))
    min_apps_key = min_apps[0]
    min_apps_val = min_apps[1]

    if len(min_apps_val) == 1:
      (key,) = min_apps_val
      del above_threshold[key]
    else:
      for key in min_apps_val:
        if key in above_threshold:
          above_threshold[key].remove(min_apps_key)

    del above_threshold[min_apps_key]
    high_corr_cols.append(corrs.columns[min_apps_key])

  return high_corr_cols


class DropFeaturesTransformer(BaseEstimator, TransformerMixin):
  """
  Provide this to a pipeline in order to drop features with low correlation to target and/or high correlation to other features
  """
  def __init__(self, X_train, y_train, Xy_corrs=None, Xy_threshold=None, XX_corrs=None, XX_threshold=None):
    self.X_train = X_train
    self.y_train = y_train
    self.Xy_corrs = Xy_corrs
    self.Xy_threshold = Xy_threshold
    self.XX_corrs = XX_corrs
    self.XX_threshold = XX_threshold

    if Xy_corrs is not None:
      X_y_cols_to_drop = features_with_low_correlation_with_target_to_drop(X_train, y_train, Xy_corrs, Xy_threshold)
    else:
      X_y_cols_to_drop = []

    if XX_corrs is not None:
      X_X_cols_to_drop = features_with_high_correlation_with_each_other_to_drop(X_train, XX_corrs, XX_threshold, cols_to_ignore=X_y_cols_to_drop)
    else:
      X_X_cols_to_drop = []
    self.cols_to_drop = np.union1d(X_y_cols_to_drop, X_X_cols_to_drop)

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    return X.drop(self.cols_to_drop, axis=1, inplace=False)


class AgeGenderImputer(BaseEstimator, TransformerMixin):
  """
  Provide this to a pipeline in order to impute missing values according to value's subject's age-gender group
  """
  def __init__(self):  # If needed, you can add parameters to this function
    self.imputers = None

  def fit(self, X, y=None):
    X_copy = X.copy()
    X_copy['age_gender_group'] = pd.cut(X_copy['age'], bins=[-1, 19, 29, 39, 49, 59, 69, 79, float('inf')],
                                          labels=['0_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80p']
                                          ).astype(str) + '_' + X_copy['gender'].astype(str)

    # calculate means
    self.imputers = X_copy.groupby('age_gender_group').mean()
    return self

  def transform(self, X, y=None):
    X_copy = X.copy()
    X_copy['age_gender_group'] = pd.cut(X_copy['age'], bins=[-1, 19, 29, 39, 49, 59, 69, 79, float('inf')],
                                          labels=['0_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80p']
                                          ).astype(str) + '_' + X_copy['gender'].astype(str)

    # use the pre-calculated means to impute missing values
    for group, mean_values in self.imputers.iterrows():
      condition = (X_copy['age_gender_group'] == group)
      X_copy.loc[condition] = X_copy.loc[condition].fillna(mean_values)

    X_copy.drop(columns=['age_gender_group'], inplace=True)
    return X_copy


class SofaSapsiiImputer(BaseEstimator, TransformerMixin):
  """
  Provide this to a pipeline in order to replace missing SOFA and SAPS II values with 0
  """
  def __init__(self):  # If needed, you can add parameters to this function
    return

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X_copy = X.copy()
    X_copy['sofa'] = X_copy['sofa'].fillna(0)
    X_copy['sapsii'] = X_copy['sapsii'].fillna(0)
    return X_copy