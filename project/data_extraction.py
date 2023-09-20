import constants
import numpy as np
import pandas as pd
from datetime import timedelta
from google.cloud import bigquery


def get_hosps(subject_ids, client):
  """
  Function to get hospitalizations data from Mimic-III
  :param subject_ids: IDs of subjects to fetch
  :param client: Client to Mimic-III
  :return: Hospitalizations data
  """
  hospquery = \
  """
  SELECT admissions.subject_id, admissions.hadm_id
  , admissions.admittime, admissions.dischtime
  , admissions.ethnicity, admissions.deathtime
  , admissions.admission_type, admissions.insurance
  , patients.gender, patients.dob, patients.dod
  FROM `physionet-data.mimiciii_clinical.admissions` admissions
  INNER JOIN `physionet-data.mimiciii_clinical.patients` patients
      ON admissions.subject_id = patients.subject_id
  WHERE admissions.has_chartevents_data = 1
  ORDER BY admissions.subject_id, admissions.hadm_id, admissions.admittime;
  """

  hosps = client.query(hospquery).result().to_dataframe().rename(str.lower, axis='columns')

  hosps = hosps.merge(subject_ids, on='subject_id', how='inner')

  hosps = hosps.sort_values(['subject_id', 'admittime'])
  return hosps


def feature_extraction(hosps):
  """
  Manipulates hospitalizations data to contain information in correct format
  :param hosps: Hospitalizations data
  :return: Hospitalizations data with information in correct format
  """
  # Generate feature columns for los, age and mortality
  def age(admittime, dob):
    if admittime < dob:
      return 0
    return admittime.year - dob.year - ((admittime.month, admittime.day) < (dob.month, dob.day))

  hosps['age'] = hosps.apply(lambda row: age(row['admittime'], row['dob']), axis=1)
  hosps['los_hosp_hr'] = (hosps.dischtime - hosps.admittime) / pd.Timedelta(hours=1)
  hosps['mort'] = np.where(~np.isnat(hosps.dod),1,0)

  # Ethnicity - one hot encoding
  hosps.ethnicity = hosps.ethnicity.str.lower()
  hosps.loc[(hosps.ethnicity.str.contains('^white')),'ethnicity'] = 'white'
  hosps.loc[(hosps.ethnicity.str.contains('^black')),'ethnicity'] = 'black'
  hosps.loc[(hosps.ethnicity.str.contains('^hisp')) | (hosps.ethnicity.str.contains('^latin')),'ethnicity'] = 'hispanic'
  hosps.loc[(hosps.ethnicity.str.contains('^asia')),'ethnicity'] = 'asian'
  hosps.loc[~(hosps.ethnicity.str.contains('|'.join(['white', 'black', 'hispanic', 'asian']))),'ethnicity'] = 'other'
  hosps = pd.concat([hosps, pd.get_dummies(hosps['ethnicity'], prefix='eth')], axis = 1)

  # Gender to binary
  hosps['gender'] = np.where(hosps['gender']=="M", 1, 0)

  # admission type and insurance - one hot encoding
  df_cat = hosps[['admission_type', 'insurance']].copy()
  df_num = hosps.drop(['admission_type', 'insurance'], axis = 1)
  df_cat = pd.get_dummies(df_cat, drop_first=True)
  hosps = pd.concat([df_num, df_cat], axis = 1)

  for col in constants.admission_type_columns + constants.insurance_columns:
    if col not in hosps.columns:
      hosps[col] = False

  return hosps


def generate_target_labels(hosps):
  """
  Adds target labels to hospitalizations data
  :param hosps: Hospitalizations data
  :return: Hospitalizations data with target labels
  """
  labels = ['mortality_label', 'prolonged_stay_label', 'readmit_label']

  hosps['mortality_label'] = np.zeros(hosps.shape[0])
  hosps['prolonged_stay_label'] = np.zeros(hosps.shape[0])

  # mortality label
  days_until_death = (hosps['dod'] - hosps['dischtime'].dt.floor('D')).dt.days
  hosps['mortality_label'] = (days_until_death <= constants.mortality_window).astype(int)

  # prolonged stay label
  hosps['prolonged_stay_label'] = (hosps['los_hosp_hr'] > constants.prolonged_stay * 24).astype(int)

  # readmit label
  hosps = hosps.sort_values(by=['subject_id', 'admittime'])
  hosps['next_admittime'] = hosps.groupby('subject_id')['admittime'].shift(-1)
  hosps['time_to_next_admit'] = hosps['next_admittime'] - hosps['dischtime']
  hosps['readmit_label'] = (hosps['time_to_next_admit'] <= timedelta(days=constants.hospital_readmission)).astype(int)
  hosps = hosps.drop(['next_admittime', 'time_to_next_admit'], axis=1)

  return hosps


def apply_exclusion(hosps):
  """
  Applies exclusion criteria to hospitalizations data
  :param hosps: Hospitalizations data
  :return: Hospitalizations data without excluded subjects
  """
  # include only first admissions
  hosps = hosps.sort_values('admittime').groupby('subject_id').first()

  # only patients who admitted for at least 48 hours
  hosps = hosps.loc[hosps['los_hosp_hr'] >= constants.min_los]

  # exclude patients who died before admission
  hosps = hosps.loc[(hosps['mort'] == 0) | (hosps['dod'] - hosps['admittime'].dt.floor('d') >= timedelta(days=0))].reset_index()
  return hosps


def get_lab_data(hosps, client):
  """
  Gets lab measurements from Mimic-III
  :param hosps: Hospitalizations data
  :param client: Client to Mimic-III
  :return: Lab measurements
  """
  labquery = \
  """--sql
    SELECT labevents.subject_id ,labevents.hadm_id ,labevents.charttime
    , labevents.itemid, labevents.valuenum
    FROM `physionet-data.mimiciii_clinical.labevents` labevents
      INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
      ON labevents.subject_id = admissions.subject_id
      AND labevents.hadm_id = admissions.hadm_id
      AND labevents.charttime >= (admissions.admittime)
      AND labevents.charttime <= DATE_ADD(admissions.admittime, INTERVAL 42 HOUR)
      AND labevents.subject_id in UNNEST(@subjectids)
      AND itemid in UNNEST(@itemids)
  """

  lavbevent_meatdata = pd.read_csv('labs_metadata.csv')
  job_config = bigquery.QueryJobConfig(
      query_parameters=[
        bigquery.ArrayQueryParameter("subjectids", "INTEGER", hosps['subject_id'].tolist()),
        bigquery.ArrayQueryParameter("itemids", "INTEGER", lavbevent_meatdata['itemid'].tolist()),
      ]
  )

  labs = client.query(labquery, job_config=job_config).result().to_dataframe().rename(str.lower, axis='columns')

  # filter invalid measurmnets
  labs = labs[labs['hadm_id'].isin(hosps['hadm_id'])]
  labs = pd.merge(labs, lavbevent_meatdata, on='itemid')
  labs = labs[labs['valuenum'].between(labs['min'], labs['max'], inclusive='both')]

  return labs


def get_vitals_data(hosps, client):
  """
  Gets vital signs from Mimic-III
  :param hosps: Hospitalizations data
  :param client: Client to Mimic-III
  :return: Vital signs
  """
  vitquery = \
  """--sql
  -- Vital signs include heart rate, blood pressure, respiration rate, and temperature

    SELECT chartevents.subject_id ,chartevents.hadm_id ,chartevents.charttime
    , chartevents.itemid, chartevents.valuenum
    FROM `physionet-data.mimiciii_clinical.chartevents` chartevents
    INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
    ON chartevents.subject_id = admissions.subject_id
    AND chartevents.hadm_id = admissions.hadm_id
    AND chartevents.charttime >= (admissions.admittime)
    AND chartevents.charttime <= DATE_ADD(admissions.admittime, INTERVAL 42 HOUR)
    AND chartevents.subject_id in UNNEST(@subjectids)
    AND itemid in UNNEST(@itemids)
    -- exclude rows marked as error
    AND chartevents.error IS DISTINCT FROM 1
  """

  vital_meatdata = pd.read_csv('vital_metadata.csv')
  job_config = bigquery.QueryJobConfig(
      query_parameters=[
        bigquery.ArrayQueryParameter("subjectids", "INTEGER", hosps['subject_id'].tolist()),
        bigquery.ArrayQueryParameter("itemids", "INTEGER", vital_meatdata['itemid'].tolist()),
      ]
  )

  vits = client.query(vitquery, job_config=job_config).result().to_dataframe().rename(str.lower, axis='columns')

  # filter invalid measurmnets
  vits = vits[vits['hadm_id'].isin(hosps['hadm_id'])]
  vits = pd.merge(vits, vital_meatdata, on='itemid')
  vits = vits[vits['valuenum'].between(vits['min'], vits['max'], inclusive='both')]

  # convert units from F to C
  vits.loc[(vits['feature name'] == 'TempF'),'valuenum'] = (vits[vits['feature name'] == 'TempF']['valuenum']-32)/1.8
  vits.loc[vits['feature name'] == 'TempF','feature name'] = 'TempC'

  return vits


def get_sofa_data(client):
  """
  Gets SOFA scores
  :param client: Client to Mimic-III
  :return: Sofa scores
  """
  sofa_query = \
  """--sql
      SELECT sofa.subject_id, sofa.hadm_id, sofa.sofa, sofa.icustay_id
      FROM `physionet-data.mimiciii_derived.sofa` sofa
          INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
          ON sofa.subject_id = admissions.subject_id
          AND sofa.hadm_id = admissions.hadm_id
  """
  sofa = client.query(sofa_query).result().to_dataframe().rename(str.lower, axis='columns')

  # only keep first icustay data
  sofa.sort_values(by=['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
  sofa.reset_index(drop=True, inplace=True)
  sofa = sofa.drop_duplicates(subset=['subject_id', 'hadm_id'], keep='first')
  sofa = sofa.drop('icustay_id', axis=1)

  return sofa


def get_sapsii_data(client):
  """
  Gets SAPS II scores
  :param client: Client to Mimic-III
  :return: SAPS II scores
  """
  sapsii_query = \
  """--sql
      SELECT sapsii.subject_id, sapsii.hadm_id, sapsii.sapsii, sapsii.icustay_id
      FROM `physionet-data.mimiciii_derived.sapsii` sapsii
      INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
          ON sapsii.subject_id = admissions.subject_id
          AND sapsii.hadm_id = admissions.hadm_id
  """
  sapsii = client.query(sapsii_query).result().to_dataframe().rename(str.lower, axis='columns')

  # only keep first icustay data
  sapsii.sort_values(by=['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
  sapsii.reset_index(drop=True, inplace=True)
  sapsii = sapsii.drop_duplicates(subset=['subject_id', 'hadm_id'], keep='first')
  sapsii = sapsii.drop('icustay_id', axis=1)

  return sapsii


def get_medication_data(hosps, client):
  """
  Gets medication data
  :param hosps: Hospitalizations data
  :param client: Client to Mimic-III
  :return: Medication data
  """
  medication_categories = {
      "vasopressors": ['Norepinephrine', 'Epinephrine', 'Vasopressin'],
      "sedatives": ['Propofol', 'Midazolam', 'Fentanyl'],
      "antibiotics": ['Amoxicillin', 'Vancomycin', 'Piperacillin'],
      "antiarrhythmics": ['Bretylium', 'Amiodarone', 'Lidocaine'],
      "anticoagulants": ['Heparin', 'Warfarin'],
      "inotropes": ['Dopamine', 'Dobutamine', 'Milrinone']
  }

  columns = list(medication_categories.keys())
  medications_df = pd.DataFrame(columns=['subject_id'] + columns)
  medications_df['subject_id'] = hosps['subject_id'].unique()
  medications_df.fillna(0, inplace=True)

  medication_to_itemid = {}

  # Populate the medication_to_itemid dictionary
  for category, medications in medication_categories.items():
   for medication in medications:
        query = f"""
            SELECT itemid
            FROM `physionet-data.mimiciii_clinical.d_items`
            WHERE LOWER(label) LIKE '%{medication.lower()}%'
        """
        query_job = client.query(query)
        rows = list(query_job)
        if rows:
          rows = [row.itemid for row in rows]
          rows = list(filter(lambda item: item > 30000, rows))
          medication_to_itemid[medication] = rows

  for category, medications in medication_categories.items():
    for medication in medications:
      itemid = medication_to_itemid.get(medication)
      if itemid is None or len(itemid) == 0:
        continue
      query = f"""
          SELECT DISTINCT subject_id, hadm_id, charttime FROM (
              SELECT inputevents_cv.subject_id, inputevents_cv.hadm_id, inputevents_cv.charttime FROM `physionet-data.mimiciii_clinical.inputevents_cv` inputevents_cv
              JOIN (
                    SELECT subject_id, hadm_id, MIN(admittime) AS first_admittime
                    FROM `physionet-data.mimiciii_clinical.admissions`
                    GROUP BY subject_id, hadm_id
                ) admissions_first
                    ON inputevents_cv.subject_id = admissions_first.subject_id
                    AND inputevents_cv.hadm_id = admissions_first.hadm_id
                    AND inputevents_cv.charttime >= (first_admittime)
                    AND inputevents_cv.charttime <= DATE_ADD(first_admittime, INTERVAL 42 HOUR)
                WHERE itemid IN ({', '.join(map(str, itemid))})
                UNION ALL
                SELECT inputevents_mv.subject_id, inputevents_mv.hadm_id, inputevents_mv.starttime as charttime FROM `physionet-data.mimiciii_clinical.inputevents_mv` inputevents_mv
                JOIN (
                    SELECT subject_id, hadm_id, MIN(admittime) AS first_admittime
                    FROM `physionet-data.mimiciii_clinical.admissions`
                    GROUP BY subject_id, hadm_id
                ) admissions_first
                    ON inputevents_mv.subject_id = admissions_first.subject_id
                    AND inputevents_mv.hadm_id = admissions_first.hadm_id
                    AND inputevents_mv.starttime >= (first_admittime)
                    AND inputevents_mv.starttime <= DATE_ADD(first_admittime, INTERVAL 42 HOUR)
                WHERE itemid IN ({', '.join(map(str, itemid))})
            )
        """
      query_job = client.query(query).result().to_dataframe().rename(str.lower, axis='columns')

      # filter only data relevant to each patients first admission & first 42 hours
      filtered_df = pd.merge(query_job, hosps, on=['subject_id', 'hadm_id'], how='right')
      filtered_df = filtered_df[
          (filtered_df['charttime'] >= filtered_df['admittime']) &
          (filtered_df['charttime'] <= filtered_df['dischtime'])
      ]
      subject_ids = filtered_df['subject_id'].unique()
      mask = medications_df['subject_id'].isin(subject_ids)
      medications_df.loc[mask, category.lower()] = 1

  for col in constants.medication_columns:
    if col not in medications_df.columns:
      medications_df[col] = False

  medications_df = medications_df.sort_values(by='subject_id')

  return medications_df


def get_procedure_events_data(hosps, client):
  """
  Gets procedure events data
  :param hosps: Hospitalizations data
  :param client: Client to Mimic-III
  :return: Procedure events data
  """
  procedureevents_mv_query = \
  """
  SELECT DISTINCT procedureevents_mv.subject_id, procedureevents_mv.hadm_id, procedureevents_mv.starttime, ordercategoryname
  FROM `physionet-data.mimiciii_clinical.procedureevents_mv` procedureevents_mv
  JOIN `physionet-data.mimiciii_clinical.admissions` admissions
    ON procedureevents_mv.subject_id = admissions.subject_id
    AND procedureevents_mv.starttime >= (admissions.admittime)
    AND procedureevents_mv.starttime <= DATE_ADD(admissions.admittime, INTERVAL 42 HOUR)
  ORDER BY procedureevents_mv.subject_id;
  """
  query_job = client.query(procedureevents_mv_query).result().to_dataframe().rename(str.lower, axis='columns')

  # filter only data relevant to each patients first admission & first 42 hours
  filtered_df = pd.merge(query_job, hosps, on=['subject_id', 'hadm_id'], how='right')
  filtered_df = filtered_df[
      (filtered_df['starttime'] >= filtered_df['admittime']) &
      (filtered_df['starttime'] <= filtered_df['dischtime'])
  ]
  filtered_df = filtered_df[['subject_id', 'ordercategoryname']]

  procedure_events_df = pd.DataFrame(columns=['subject_id'] + constants.procedure_columns)
  procedure_events_df['subject_id'] = hosps['subject_id'].unique()
  procedure_events_df.fillna(0, inplace=True)

  for category in constants.procedure_columns:
    subject_ids = filtered_df.loc[filtered_df['ordercategoryname'].str.lower() == category, 'subject_id'].tolist()
    mask = procedure_events_df['subject_id'].isin(subject_ids)
    procedure_events_df.loc[mask, category.lower()] = 1

  procedure_events_df = procedure_events_df.sort_values(by='subject_id')

  for col in constants.procedure_columns:
    if col not in procedure_events_df.columns:
      procedure_events_df[col] = False

  return procedure_events_df


def get_additional_data(hosps, client):
  """
  Gets hospitalizations data and returns dataset that includes features and target labels
  :param hosps: Hospitalizations data
  :param client: Client to Mimic-III
  :return: A dataset that includes features and target labels
  """
  print('Calling "get_lab_data"...')
  labs = get_lab_data(hosps, client)
  print('Calling "get_vitals_data"...')
  vits = get_vitals_data(hosps, client)
  print('Calling "get_sofa_data"...')
  sofa = get_sofa_data(client)
  print('Calling "get_sapsii_data"...')
  sapsii = get_sapsii_data(client)
  print('Calling "get_medication_data"...')
  medications_df = get_medication_data(hosps, client)
  print('Calling "get_procedure_events_data"...')
  procedure_events_df = get_procedure_events_data(hosps, client)

  merged_df = hosps.merge(procedure_events_df, how='left', on=['subject_id'])
  merged_df = merged_df.merge(medications_df, how='left', on=['subject_id'])

  vits['category'] = 'vits'
  vits_and_labs_df = pd.concat([vits, labs])

  vits_and_labs_df['feature name'] = vits_and_labs_df['feature name'].str.lower()
  grouped = vits_and_labs_df.groupby(['hadm_id', 'feature name'])           # group by 'hadm_id', 'feature_name'
  aggregated = grouped['valuenum'].agg(missing=lambda x: x.isna().any(), mean='mean', max='max', min='min')       # any, mean, max, and min for each group
  pivoted = aggregated.pivot_table(index=['hadm_id'], columns='feature name', values=['missing', 'mean', 'max', 'min']) # reshape
  pivoted.columns = ['_'.join(col).rstrip('_') for col in pivoted.columns]  # change names to min_feature, max_feature...
  pivoted.reset_index(inplace=True)
  merged_df = merged_df.merge(pivoted, how='left', on=['hadm_id'])

  # fix weight column
  weights = vits.loc[vits['feature name'] == 'Weight'].groupby(['subject_id'])['valuenum'].agg(['first'])
  weights.rename(columns={'first': 'weight'}, inplace=True)
  weights.reset_index(inplace=True)
  weights.rename(columns={'index': 'subject_id'}, inplace=True)
  merged_df = merged_df.drop(['min_weight', 'max_weight', 'mean_weight'], axis=1)
  merged_df = merged_df.merge(weights, how='left', on='subject_id')

  # add sofa and sapsii
  merged_df = merged_df.merge(sofa, how='left', on=['subject_id', 'hadm_id'])
  merged_df = merged_df.merge(sapsii, how='left', on=['subject_id', 'hadm_id'])

  # replace nan values with 1 in missing cloumns and convert int
  missing_cols = [col for col in merged_df.columns if col.startswith('missing_')]
  merged_df[missing_cols] = merged_df[missing_cols].fillna(1).astype(int)

  return merged_df


def generate_compound_features(hosps):
  """
  Gets a dataset that includes features and target labels and adds two features: number of medications and number of procedure events
  :param hosps: A dataset that includes features and target labels
  :return: The same dataset with 2 additional features: number of medications and number of procedure events
  """
  procedure_columns = ['ventilation', 'intubation/extubation', 'significant events', 'communication', 'peripheral lines',
                       'crrt filter change', 'continuous procedures', 'dialysis', 'imaging', 'peritoneal dialysis',
                       'procedures', 'invasive lines']

  medication_columns = ['vasopressors', 'sedatives', 'antibiotics', 'antiarrhythmics', 'anticoagulants', 'inotropes']

  hosps_copy = hosps.copy()
  hosps_copy['n_procedures'] = hosps_copy[procedure_columns].sum(axis=1)
  hosps_copy['n_medications'] = hosps_copy[medication_columns].sum(axis=1)

  return hosps_copy


def generate_hosps(subject_ids, client):
  """
  Gets subject ids and a client to Mimic-III and returns a dataset containing features and target labels, and a list of subject ids that were not filtered out
  :param subject_ids: IDs of subjects to fetch
  :param client: Client to Mimic-III
  :return: A dataset containing features and target labels, and a list of subject ids that were not filtered out
  """
  print('Calling "get_hosps"...')
  hosps = get_hosps(subject_ids, client)
  print('Calling "feature_extraction"...')
  hosps = feature_extraction(hosps)
  print('Calling "generate_target_labels"...')
  hosps = generate_target_labels(hosps)
  print('Calling "apply_exclusion"...')
  hosps = apply_exclusion(hosps)
  print('Calling "get_additional_data"...')
  hosps = get_additional_data(hosps, client)
  print('Calling "generate_compound_features"...')
  hosps = generate_compound_features(hosps)

  # drop all columns that might cause data leakage or are unnecessary
  filtered_subject_ids = hosps['subject_id']
  columns_to_drop = ['subject_id', 'admittime', 'dischtime', 'ethnicity', 'dob', 'dod', 'deathtime', 'mort', 'hadm_id', 'los_hosp_hr']
  hosps.drop(columns_to_drop, axis=1, inplace=True)

  # convert all binary columns to int
  for col in constants.binary_columns:
    hosps[col] = hosps[col].astype(int)

  print('Done data extraction')
  return hosps, filtered_subject_ids
