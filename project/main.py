import pandas as pd
import google.cloud.bigquery as bigquery
from unseen_data_evaluation import run_pipeline_on_unseen_data
import os.path
import modeling
import pickle

if __name__ == '__main__':
    client = bigquery.client.Client(project='mimic3-383317')

    if not (os.path.isfile('mortality_final_model.pkl') and os.path.isfile('prolonged_final_model.pkl') and os.path.isfile('readmit_final_model.pkl')):
      subject_ids_df = pd.read_csv('initial_cohort.csv')
      mortality_final_model , prolonged_final_model, readmit_final_model = modeling.run(subject_ids_df, client)
      pickle.dump(mortality_final_model, open('mortality_final_model.pkl', 'wb'))
      pickle.dump(prolonged_final_model, open('prolonged_final_model.pkl', 'wb'))
      pickle.dump(readmit_final_model, open('readmit_final_model.pkl', 'wb'))

    subject_ids_df = pd.read_csv('test_example.csv')
    results = run_pipeline_on_unseen_data(subject_ids_df['subject_id'].to_list(), client)
    results.to_excel('results.xlsx')
