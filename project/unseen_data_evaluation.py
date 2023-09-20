import numpy as np

import constants
import pandas as pd
from data_extraction import generate_hosps
from model_loading import load_models
import plotting
from data_patition import extract_X_y


def run_pipeline_on_unseen_data(subject_ids ,client):
  """
  Run your full pipeline, from data loading to prediction.

  :param subject_ids: A list of subject IDs of an unseen test set.
  :type subject_ids: List[int]

  :param client: A BigQuery client object for accessing the MIMIC-III dataset.
  :type client: google.cloud.bigquery.client.Client

  :return: DataFrame with the following columns:
              - subject_id: Subject IDs, which in some cases can be different due to your analysis.
              - mortality_proba: Prediction probabilities for mortality.
              - prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
              - readmission_proba: Prediction probabilities for readmission.
  :rtype: pandas.DataFrame
  """
  subject_ids_df = pd.DataFrame(subject_ids, columns=['subject_id'])
  hosps, filtered_subject_ids = generate_hosps(subject_ids_df, client)

  X, y = extract_X_y(hosps)

  missing_cols = [col for col in (constants.numerical_columns + constants.binary_columns) if col not in X.columns]
  assert all(col in X.columns for col in constants.numerical_columns + constants.binary_columns),\
    f'The next columns appeared in training but missing from test set: {missing_cols}'

  excess_cols = [col for col in X.columns if col not in (constants.numerical_columns + constants.binary_columns)]
  X.drop(excess_cols, axis=1, inplace=True)

  mortality_model, prolonged_model, readmit_model = load_models()

  mortality_proba = mortality_model.predict_proba(X)[:, 1]

  X['mortality_pred'] = mortality_proba
  prolonged_proba = prolonged_model.predict_proba(X)[:, 1]

  X['prolonged_pred'] = prolonged_proba
  readmission_proba = readmit_model.predict_proba(X)[:, 1]

  return pd.DataFrame({'subject_id': filtered_subject_ids,
                       'mortality_proba': mortality_proba,
                       'prolonged_LOS_proba': prolonged_proba,
                       'readmission_proba': readmission_proba})
