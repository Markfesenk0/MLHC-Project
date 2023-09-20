import constants
import os.path
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from preprocessing import AgeGenderImputer, SofaSapsiiImputer
from sklearn.calibration import CalibratedClassifierCV
from data_extraction import generate_hosps
from data_patition import extract_X_y


def train_best_models(X, y):
    """
    Gets a feature matrix and target labels matrices (Mortality, Prolonged Stay, and Readmission) and returns three trained models with the best parameters defined in advance
    :param X: Feature matrix
    :param y: Target labels matrix
    :return: Three trained models with the best parameters defined in advance
    """
    mortality_classifier = XGBClassifier(**constants.mortality_best_params)
    prolonged_classifier = XGBClassifier(**constants.prolonged_best_params)
    readmit_classifier = LogisticRegression(**constants.readmit_best_params)

    imputer1 = SofaSapsiiImputer()
    imputer2 = AgeGenderImputer()
    scaler = ColumnTransformer(transformers=[('num', StandardScaler(), constants.numerical_columns)], remainder='passthrough')

    mortality_pipe = Pipeline(steps=[
        ('imputer1', imputer1),
        ('imputer2', imputer2),
        ('model', mortality_classifier)])
    prolonged_pipe = Pipeline(steps=[
        ('imputer1', imputer1),
        ('imputer2', imputer2),
        ('model', prolonged_classifier)])
    readmit_pipe = Pipeline(steps=[
        ('imputer1', imputer1),
        ('imputer2', imputer2),
        ('scaler', scaler),
        ('model', readmit_classifier)])

    mortality_model = CalibratedClassifierCV(mortality_pipe, method='sigmoid', n_jobs=-1)
    mortality_model.fit(X, y[constants.mortality_label])
    print('Done fitting mortality model')

    X['mortality_pred'] = mortality_model.predict_proba(X)[:, 1]
    prolonged_model = CalibratedClassifierCV(prolonged_pipe, method='sigmoid', n_jobs=-1)
    prolonged_model.fit(X, y[constants.prolonged_label])
    print('Done fitting prolonged stay model')

    X['prolonged_pred'] = prolonged_model.predict_proba(X)[:, 1]
    readmit_model = CalibratedClassifierCV(readmit_pipe, method='sigmoid', n_jobs=-1)
    readmit_model.fit(X, y[constants.readmit_label])
    print('Done fitting readmission model')

    return mortality_model, prolonged_model, readmit_model


def run(subject_ids_df, client):
    """
    Gets a dataframe containing subject ids and a client to Mimic-III and returns three models predicting Mortality, Prolonged Stay, and Readmission, trained on the given subjects
    :param subject_ids_df: Dataframe containing subject ids
    :param client: Client to Mimic-III
    :return: Three models predicting Mortality, Prolonged Stay, and Readmission, trained on the given subjects
    """
    subject_ids_df['subject_id'] = subject_ids_df['subject_id']
    if os.path.isfile('hosps_cohort.xlsx'):
        hosps = pd.read_excel('hosps_cohort.xlsx', index_col=0)
    else:
        hosps, _ = generate_hosps(subject_ids_df, client)
        hosps.to_excel('hosps_cohort.xlsx')

    X, y = extract_X_y(hosps)
    print('Calling "train_best_models"...')
    return train_best_models(X, y)
