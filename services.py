import pymongo
from pymongo import MongoClient

client = MongoClient('localhost:27017')
db = client.icu_db
patients_collection = db.patient_details_collection
features_collection = db.all_feats_collection
users_collection = db.users
models_collection = db.models_collection


# Get patient's mortality predictions for last 7 days
def getPatientHistory(id):

    predictions = []
    ret = []
    docs = features_collection.find({"temp_adm_num":id})
    for doc in docs:
        predictions.append((doc['dayNum'], doc['mortality_prediction']))

    predictions.sort(key = lambda x:x[0])
    for pred in predictions:
        ret.append(pred[1])
    if len(ret)>7:
        ret = ret[-7:]
    return ret

# Get latest record of patient
def getPatientStatus(id):
    docs = features_collection.find({"temp_adm_num": id})
    latest_doc = None

    for i,doc in enumerate(docs):
        if i==0:
            latest_doc = doc
        else:
            if doc['dayNum']>latest_doc['dayNum']:
                latest_doc=doc

    return latest_doc


def getModelsMetaData():

    try:
        doc1 = list(models_collection.find({'name': 'mortality_model'}))[0]
        doc2 = list(models_collection.find({'name': 'daysForward_model'}))[0]
    except:
        return None,None

    mortality_model = {}
    days_forward_model = {}

    mortality_model['lastUpdate'] = doc1['last_trained']
    mortality_model['accuracy'] = doc1['accuracy']
    mortality_model['auc'] = doc1['auc']
    mortality_model['auc_path'] = doc1['roc_auc_curve_path']
    mortality_model['feature_importance_path'] = doc1['features_importance_path']

    days_forward_model['lastUpdate'] = doc2['last_trained']
    days_forward_model['auc'] = doc2['AUCs']

    return mortality_model, days_forward_model
