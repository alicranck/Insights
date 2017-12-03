from config import *
from model_saving import *
from data_insert_aux import *
import shutil
from shutil import copytree, rmtree
import os

client = MongoClient('localhost:27017')
db = client.icu_db


# Parse new data in folder. convert to csv, merge and format aptly.
#   methods source code is at data_insert_aux.py
def parse_new_data():
    # Hash patients' information
    print("Hashing data...")
    hash_ids(newDataPath)

    # Format dates and mortality indicator in demographic file
    print("Formatting demographic file...")
    format_dates_dem_file(newDataPath)

    # Format xsl and xslx in input folder files to csvs
    print("Converting to csv...")
    excels_to_csv(newDataPath)

    # Merge measurments csvs with demographic file to one table
    print("Merging...")
    create_merged_csv(newDataPath)

    # Clean and organize data
    print("Cleaning data...")
    clean_data(newDataPath)

    return


# Assigning clusters and mortality predictions
# returns a pandas Dataframe with parsed data and assigned predictions
def assign_clusters_and_predictions():

    # Assign clusters and predict mortality
    df = pd.read_csv(r'' + newDataPath + '/clean_csv.csv')
    print("Assigning Clusters...")
    df['cluster'] = assign_cluster(df)

    print("Assigning mortality predictions...")
    df['mortality_prediction'] = pred_gen_mort(df)
    df['next3days_mortality_prediction'] = pred_3days_forward_mort(df)
    df['justification'] = get_justification(df)

    return df


# Updates DB with the new data and predictions. in case of mortality, updates
#   mortality status in all previous documents for future model training
#
# input - a pandas Dataframe including parsed data, cluster assignments, and mortality predictions
def update_db_features(df):

    # Insert predictions to db
    db.all_feats_collection.insert_many(df.to_dict('records'))

    # Update mort_adm of admitted patients for future training of model
    df2 = df[df['mort_adm']==1]
    patients_to_update = df2['temp_adm_num'].tolist()
    db.all_feats_collection.update_many(
        {'name' : {'$in' : patients_to_update}},
        {'$set' : {'mort_adm' : 1}})

    return


# Update personal details of new patients in DB
#
# input - a pandas Dataframe with parsed personal details ONLY - no measurements or features
def update_db_details(df):

    # Insert personal details of newly admitted patients to db
    df.drop(['isMortDay', 'isVentilated', 'sepsis', 'chronic-category', 'adm.year'], axis=1, inplace=True)
    df = df.rename(columns={'adm.num': 'adm_num'})
    df = df.rename(columns={'adm.date': 'adm_date'})
    df['update'] = 1

    for patient in df['adm_num']:
        if db.patient_details_collection.find_one({"adm_num" : patient}) is not None:
            df['update'] = 0

    df = df[df['update'] == 1]
    if not df.empty:
        db.patient_details_collection.insert_many(df.to_dict('records'))

    return


# Archives files that were updated in DB and clears the new data folder
def archive_files():

    date_str = '/' + datetime.datetime.today().strftime('%d-%m-%Y')
    copytree(newDataPath, archivePath+date_str)

    for file in os.listdir(newDataPath):
        file_path = os.path.join(newDataPath, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            for file in os.listdir(file_path):
                file_path2 = os.path.join(file_path, file)
                os.unlink(file_path2)

    return




# A periodical update of the predicting models -
#       Retraining on all data in the DB. methods code is in model_saving.py
def update_models():
    df = pd.DataFrame(list(db.all_feats_collection.find()))
    db.models_collection.remove({"name":"mortality_model"})
    db.models_collection.remove({"name":"clustering_model"})
    db.models_collection.remove({"name":"linearRegression_model"})
    build_clustering_model(df)
    build_mortality_model(df)
    build_lr_model(df)
    return


# Daily update of the DB with new data from the hospital -
#    new data is parsed and then patients are assigned clusters and
#    mortality predictions. all information is then inserted to the DB
#    and files are moved to archive
def daily_update():

    parse_new_data()
    df = assign_clusters_and_predictions()
    update_db_features(df)
    df2 = pd.read_csv(r'' + newDataPath + '/icu_mort_dated.csv')
    update_db_details(df2)
    archive_files()

    return


df = pd.DataFrame(list(db.all_feats_collection.find()))
