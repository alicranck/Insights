from pymongo import MongoClient


client = MongoClient('localhost:27017')
db = client.icu_db

projectPath = 'C:\Almog\\flaskTry\FlaskApp'
newDataPath = 'C:\Almog\\flaskTry\FlaskApp\\backend\\new_data'
archivePath = 'C:\Almog\\flaskTry\FlaskApp\\backend\\archive'

MEDICAL_INFO_COLUMNS_CLUSTERING = ['isVentilated', 'chronic-category', 'Total Haemoglobin',
                                   'Arterial Pressure Diastolic (mmHg)', 'Arterial Pressure Systolic (mmHg)',
                                    'Base excess (vt) (mmol/l)', 'PH (ABG)', 'Bilirubin(Total) (mg/dl)',
                                   'Calcium (mg/ml)', 'Creatinine (mg/dl)', 'End Tidal CO2 (mmHg)', 'Inspired Oxygen (%)',
                                    'GastricResidue', 'GCS', 'Gluco-stick (mg%)', 'Glucose ABG (mg%)',
                                   'Haematocrit', 'Heart Rate (bpm)', 'Lactate ABG (mmol/l)', 'White Cell Count (10^9/l)',
                                    'Magnesium (mmol/l)', 'PaCO2', 'PaO2', 'Platelets (10^9/l)', 'PotassiumABG',
                                   'Total Respiratory Rate (insp/min)', 'SpO2', 'Na-ABG (mmol/l)', 'Urea (mmol/l)',
                                    'sepsis-categorical']

FEATURES_XGB=['Arterial Pressure Diastolic (mmHg)',
              'Arterial Pressure Systolic (mmHg)',
              'Base excess (vt) (mmol/l)',
              'Bilirubin(Total) (mg/dl)',
              'Calcium (mg/ml)',
              'Creatinine (mg/dl)',
              'End Tidal CO2 (mmHg)',
              'GCS',
              'GastricResidue',
              'Gluco-stick (mg%)',
              'Glucose ABG (mg%)',
              'Haematocrit',
               'Heart Rate (bpm)',
              'Inspired Oxygen (%)',
              'Lactate ABG (mmol/l)',
              'Magnesium (mmol/l)',
              'Na-ABG (mmol/l)',
              'PH (ABG)',
              'PaCO2',
              'PaO2',
              'Platelets (10^9/l)',
              'PotassiumABG',
              'Pre-albumin (mg/dl)',
              'SpO2',
              'Total Haemoglobin',
              'Total Respiratory Rate (insp/min)',
              'Urea (mmol/l)',
              'White Cell Count (10^9/l)',
              'age2',
              'chronic-category',
              'dayNum',
              'gender2',
              'isVentilated',
              'sepsis-categorical',
              'כמות שניתנה KCALENT',
              'כמות שניתנה KCALIV']
