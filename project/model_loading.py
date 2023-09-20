import pickle


def load_models():
    """
    Loads three trained models: model predicting Mortality, model predicting Prolonged Stay, and model predicting Readmission
    :return: Three trained models: model predicting Mortality, model predicting Prolonged Stay, and model predicting Readmission
    """
    return pickle.load(open('mortality_final_model.pkl', 'rb')),\
           pickle.load(open('prolonged_final_model.pkl', 'rb')),\
           pickle.load(open('readmit_final_model.pkl', 'rb'))