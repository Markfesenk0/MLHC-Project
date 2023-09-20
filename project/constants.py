min_los = 48                # only patients with at least >= 48 hours of hospitalization data
mortality_window = 30       # mortality during or after hospitalization <= 30 days
prolonged_stay = 7          # prolonged stay > 7 days
hospital_readmission = 30   # hospital readmission in <= 30 days after discharge (not to be confused with ICU readmission within the same hospital admission)

procedure_columns = ['ventilation', 'intubation/extubation', 'significant events', 'communication', 'peripheral lines',
                     'crrt filter change', 'continuous procedures', 'dialysis', 'imaging', 'peritoneal dialysis',
                     'procedures', 'invasive lines']
medication_columns = ['vasopressors', 'sedatives', 'antibiotics', 'antiarrhythmics', 'anticoagulants', 'inotropes']
admission_type_columns = ['admission_type_EMERGENCY', 'admission_type_NEWBORN', 'admission_type_URGENT']
insurance_columns = ['insurance_Medicaid', 'insurance_Medicare', 'insurance_Private', 'insurance_Self Pay']
numerical_columns = ['age', 'weight', 'sofa', 'sapsii', 'max_albumin', 'max_anion gap', 'max_bicarbonate', 'max_bilirubin', 'max_bun',
                       'max_chloride', 'max_creatinine', 'max_diasbp', 'max_glucose', 'max_heartrate', 'max_hematocrit', 'max_hemoglobin',
                       'max_inr', 'max_lactate', 'max_magnesium', 'max_meanbp', 'max_phosphate', 'max_platelet', 'max_potassium', 'max_pt',
                       'max_ptt', 'max_resprate', 'max_sodium', 'max_spo2', 'max_sysbp', 'max_tempc', 'max_wbc', 'mean_albumin',
                       'mean_anion gap', 'mean_bicarbonate', 'mean_bilirubin', 'mean_bun', 'mean_chloride', 'mean_creatinine', 'mean_diasbp',
                       'mean_glucose', 'mean_heartrate', 'mean_hematocrit', 'mean_hemoglobin', 'mean_inr', 'mean_lactate',
                       'mean_magnesium', 'mean_meanbp', 'mean_phosphate', 'mean_platelet', 'mean_potassium', 'mean_pt', 'mean_ptt',
                       'mean_resprate', 'mean_sodium', 'mean_spo2', 'mean_sysbp', 'mean_tempc', 'mean_wbc', 'min_albumin',
                       'min_anion gap', 'min_bicarbonate', 'min_bilirubin', 'min_bun', 'min_chloride', 'min_creatinine', 'min_diasbp',
                       'min_glucose', 'min_heartrate', 'min_hematocrit', 'min_hemoglobin', 'min_inr', 'min_lactate', 'min_magnesium',
                       'min_meanbp', 'min_phosphate', 'min_platelet', 'min_potassium', 'min_pt', 'min_ptt', 'min_resprate', 'min_sodium',
                       'min_spo2', 'min_sysbp', 'min_tempc', 'min_wbc', 'n_procedures', 'n_medications']
binary_columns = ['admission_type_EMERGENCY', 'admission_type_NEWBORN', 'admission_type_URGENT', 'antiarrhythmics', 'antibiotics',
                  'anticoagulants', 'communication', 'continuous procedures', 'crrt filter change', 'dialysis', 'eth_asian', 'eth_black',
                  'eth_hispanic', 'eth_other', 'eth_white', 'gender', 'imaging', 'inotropes', 'insurance_Medicaid', 'insurance_Medicare',
                  'insurance_Private', 'insurance_Self Pay', 'intubation/extubation', 'invasive lines', 'missing_albumin', 'missing_anion gap',
                  'missing_bicarbonate', 'missing_bilirubin', 'missing_bun', 'missing_chloride', 'missing_creatinine', 'missing_diasbp',
                  'missing_glucose', 'missing_heartrate', 'missing_hematocrit', 'missing_hemoglobin', 'missing_inr', 'missing_lactate',
                  'missing_magnesium', 'missing_meanbp', 'missing_phosphate', 'missing_platelet', 'missing_potassium', 'missing_pt',
                  'missing_ptt', 'missing_resprate', 'missing_sodium', 'missing_spo2', 'missing_sysbp', 'missing_tempc', 'missing_wbc',
                  'missing_weight', 'peripheral lines', 'peritoneal dialysis', 'procedures', 'sedatives', 'significant events', 'vasopressors',
                  'ventilation']

target_labels = ['mortality_label', 'prolonged_stay_label', 'readmit_label']
mortality_label = target_labels[0]
prolonged_label = target_labels[1]
readmit_label = target_labels[2]

mortality_best_params = {'max_depth': 2, 'n_estimators': 200, 'scale_pos_weight': 1}
prolonged_best_params = {'max_depth': 2, 'n_estimators': 150}
readmit_best_params = {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}