In addition to the files provided, the project includes the next files

constants.py: Constant values used by functions in the project

data_extraction.py: Extraction of information from MIMIC-III and creation of the dataset for the project

data_partition.py: Partition of the dataset to X (features) and y (target labels) and partition of X and y to train and test sets

main.py: Running the pipeline - modeling on inital cohort (if not done already) and evaluation on test example

model_loading.py: Loading the trained ML models

modeling.py: Generating the dataset based on the inital cohort and training the models on the dataset

plotting.py: Plotting ROC and PR curves

preprocessing.py: The preprocessing modules needed to train and evaluate the models