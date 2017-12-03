from config import *
import pandas as pd
import numpy as np
import glob
import hashlib
from sklearn.cluster import KMeans
import sklearn.preprocessing
from operator import itemgetter
from datetime import timedelta
from pymongo import *
import pickle
import xgboost as xgb
from model_saving import *



# For patient discretion, removes patient's personal information
#   and hashes IDs and admission number
#
#input - a path to the folder where a demographic file from with new patient information exists
def hash_ids(path):
    paths = []
    paths.extend(glob.glob(path+'\icu_mort.csv'))
    paths.extend(glob.glob(path+'\excels\*.xls'))
    paths.extend(glob.glob(path+'\excels\*.xlsx'))
    for i, filename in enumerate(paths):
        print(filename)
        try:
            df = pd.read_csv(filename)
        except:
            df = pd.read_excel(filename)
        if u'מספר אשפוז' in df.columns:
            df[u'מספר אשפוז'] = df[u'מספר אשפוז'].apply(lambda x: hashlib.md5(str(x)).hexdigest())
        if 'MR Number' in df.columns:
            df['MR Number'] = df['MR Number'].apply(lambda x: hashlib.md5(str(x)).hexdigest())
        if 'mr number' in df.columns:
            df['MR Number'] = df['mr number'].apply(lambda x: hashlib.md5(str(x)).hexdigest())
        if 'ID' in df.columns:
            df['ID'] = df['ID'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())
        if 'adm.num' in df.columns:
            df['adm.num'] = df['adm.num'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())

        df = df.drop([u'שם החולה', 'Admission_Number (an)', 'name', 'adm.num.long'], axis=1, errors='ignore')
        if i==0:
            df.to_csv(path + '/' + filename.split("\\")[-1].split(".")[0] + '_discreet.csv', encoding='utf-8')
        else:
            df.to_excel(path + '\\hashed_excels\\' + filename.split("\\")[-1].split(".xl")[0] + '.xls', encoding='utf-8')
    return


# Format admission and discharge dates to common format. also computes if patient died
#     during admission.
#
#input - a path to the folder where a hashed demographic file exists
def format_dates_dem_file(path):
    df = pd.read_csv(r'' + path + '\icu_mort_discreet.csv')
    df['adm.date'] = pd.to_datetime(df['adm.unit'])

    df['age'] = pd.to_datetime(df['adm.date']).dt.year - pd.to_datetime(df['birth.date']).dt.year
    df['gender'] = pd.get_dummies(df['gender'])
    df['adm.year'] = pd.to_datetime(df['adm.unit']).dt.year

    cols_to_drop = ['adm.unit', 'dis.unit', 'adm.hosp', 'dis.hosp', 'birth.date', 'transfer.num',
                    'icd9', 'diag.desc', 'indication', 'atd.unit.code',  'atd.unit.name', 'ID']
    df.drop([col for col in df.columns if (col in cols_to_drop or "Unnamed" in col)], axis=1, inplace=True)

    df = df.groupby(['adm.num', 'adm.date'], as_index=False).mean()
    df.to_csv(r'' + path + '\icu_mort_dated.csv', encoding='utf-8')
    return


# Converts all measurment files in input folder from excel to csv.
#   averages same day measurments
#
# input - a path to the folder where 'excels' folder with input data exists, as well as the dated icu_mort file
def excels_to_csv(path):
    pathes = []
    pathes.extend(glob.glob(r'' + path + '\hashed_excels\*.xlsx'))
    pathes.extend(glob.glob(r'' + path + '\hashed_excels\*.xls'))
    pathes.extend(glob.glob(r'' + path + '\hashed_excels\*.csv'))
    for filename in pathes:
        print(filename)
        try:
            dfe = pd.read_excel(filename)
        except:
            dfe = pd.read_csv(filename)
        dfe = dfe.rename(columns={u'זמן': 'Time'})  # rename for convention
        dfe = dfe.rename(columns={u'זמן התחלה': 'Time'})
        dfe = dfe.rename(columns={u'תאריך': 'Time'})
        dfe = dfe.rename(columns={'MR Number': 'adm.num'})
        dfe = dfe.rename(columns={u'מספר אשפוז': 'adm.num'})
        if u'שם הפרמטר' in dfe.columns:
            dfe = pivot_param_name(dfe)
        dfe['day'] = pd.to_datetime(dfe['Time']).dt.date
        dfe['year'] = pd.to_datetime(dfe['Time']).dt.year
        dfe = dfe.groupby(['adm.num', 'day'], as_index=False).mean()

        dfe.to_csv(r'' + path + '/processed_measurements/' + filename.split("\\")[-1].split(".xl")[-2] + '.csv',
                    encoding='utf-8')
    return


# Merge all csvs to one file
def create_merged_csv(path):

    dfc = pd.read_csv(r'' + path + '\icu_mort_dated.csv')

    for i, filename in enumerate(glob.glob(r'' + path + '/processed_measurements/*.csv')):
        print(filename)
        dfe = pd.read_csv(filename)
        dfe.drop(['year'], axis=1, inplace=True)
        if i==0:
            dfc = pd.merge(dfc, dfe, on=['adm.num'])
        else:
            dfc = pd.merge(dfc, dfe, on=['adm.num', 'day'])
    dfc.to_csv(r'' + path + '/merged_csv.csv', encoding='utf-8')
    return


# Clean and format all data to a uniform structure
def clean_data(path):

    df = pd.read_csv(r'' + path + '\merged_csv.csv')

    df = df.rename(columns={u'chronic_category':'chronic-category'})
    df = df.rename(columns={u'sepsis':'sepsis-categorical'})
    df = df.rename(columns={u'age':'age2'})
    df = df.rename(columns={u'gender':'gender2'})

    df['day'] = pd.to_datetime(df['day'])
    df['dayNum'] = df['day'].dt.day-pd.to_datetime(df['adm.date'],format="%Y/%m/%d").dt.day

    # Remove unneccesary columns
    df.drop([col for col in df.columns if "Unnamed" in col or "_x" in col or "_y" in col], axis=1, inplace=True)
    df.drop(['Admission_Number (an)', 'adm.unit', 'dis.unit', 'birth.date', 'BilirubinSOFA', 'Na-ABG',
             'Inorganic Phosphate', 'ערך בסיס', 'adm.date'], axis=1, errors='ignore', inplace=True)
    df.drop([col for col in df.columns if ('_' in col or 'ID' in col or 'mort.adm' in col
                    or 'adm.unit' in col or 'mortality.date' in col or 'adm.year' in col)],axis=1, inplace=True)

    df=df[pd.notnull(df['day'])]  # Values without day are values from icu_mort.csv that are not in feature csvs
    df = df[df['dayNum']>=0]  # Delete negative dayNums
    pd.set_option("display.max_columns",70)
    df.sort_values(['adm.num','dayNum'], axis=0, ascending=[True,True], inplace=True, kind='quicksort', na_position='last')
    df['temp_adm_num'] = df['adm.num']
    df['mort_adm'] = df['isMortDay']
    df.drop(['day', 'adm.num'], axis=1, inplace=True)

    # Delete layouts
    df = df.drop(['Chloride (mmol/l)'],axis=1,errors='ignore') #3113 full examples
    df = df[df['PH (ABG)']<10]
    df = df[df['Arterial Pressure Diastolic (mmHg)']>10]
    df = df[df['Arterial Pressure Systolic (mmHg)']>10]

    # Fill missing values with personal means and general medians
    df['chronic-category'] = pd.get_dummies(df['chronic-category'])
    #df = df.groupby(['temp_adm_num'], as_index=False).transform(lambda x: x.fillna(x.mean()))
    df['isVentilated'] = df['isVentilated'].fillna(0)
    df['sepsis-categorical']=df['sepsis-categorical'].fillna(0)
    df = df.drop(['Temperature','Chloride (mmol/l)'],axis=1,errors='ignore')
    #for col in df.columns[2:]:
        #df[col].fillna(df[col].median(), inplace=True)
    df['Total Haemoglobin'] = df['Total Haemoglobin'].fillna(df['Total Haemoglobin'].median())
    df['APACHE II'] = -1

    df.to_csv(r'' + path + '/clean_csv.csv', encoding='utf-8')

    return


def pred_gen_mort(row): ### when presenting the general mortality predictions in the website, make sure to take only those with dayNum == 1
    row = row[FEATURES_XGB]
    row['chronic-category'] = (row['chronic-category']>0).astype(int)
    row['sepsis-categorical'] = row['sepsis-categorical'].apply(lambda x: 1 if x>=1 else 0)
    model = pickle.load(open(db.models_collection.find({'name': 'mortality_model'})[0]['path'], 'rb'))
    return model.predict(xgb.DMatrix(row))


def pred_3days_forward_mort(row):
    row = row[FEATURES_XGB]
    row = row.drop(['sepsis-categorical'], axis=1, errors='ignore')
    row['chronic-category']=(row['chronic-category']>0).astype(int)
    model = pickle.load(open(db.models_collection.find({'name': 'daysForward_model'})[0]['path'], 'rb'))
    return model.predict(xgb.DMatrix(row))


def assign_cluster(row):
    row = row.drop([x for x in row.columns if x not in MEDICAL_INFO_COLUMNS_CLUSTERING], axis=1, errors='ignore')
    row['chronic-category'] = (row['chronic-category'] > 0).astype(int)
    model = pickle.load(open(db.models_collection.find({'name': 'clustering_model'})[0]['path'], 'rb'))
    return model.predict(row)


def get_justification(df):
    df=df.drop([x for x in df.columns if ('justification' in x or '_id' in x or 'APACHE' in x or 'cluster' in x or 'next3days_mortality_prediction' in x or 'mortality_prediction' in x or 'MortDay' in x \
                  or 'Unnamed' in x or 'mort' in x or 'adm' in x or 'dayNum' in x)],axis=1)
    df['justification']=df.apply(lambda x: calculate_justification(x,df), axis=1)
    return df['justification']


def calculate_justification(row,df):
    coeffs = db.models_collection.find({'name':'linearRegression_model'})[0]['coeffs']
    weights = coeffs * np.array(row)
    most_important_features=[]
    for i in np.flipud(np.argsort(weights))[:5]:
        most_important_features.append((df.columns[i], row[i]))
    return str(most_important_features)

#---------------------------Auxiliary methods--------------------------------------------------

def pivot_param_name(df):

    df = df.rename(columns={u'ערך':df['שם הפרמטר'].iloc[1].split("_")[0]})
    df = df.rename(columns={u'ערך Unit':df['שם הפרמטר'].iloc[1].split("_")[0]})
    df.drop(['שם הפרמטר'], axis=1, inplace=True)
    df = df.rename(columns={'Potassium ABG': 'PotassiumABG'})
    df = df.rename(columns={'Nasogastric Output': 'GastricResidue'})

    return df
