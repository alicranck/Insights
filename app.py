from flask import Flask, send_file, request, jsonify, redirect
import pymongo
from pymongo import MongoClient
import json,requests
from bson import Binary, Code
from bson.json_util import dumps
from services import *
from backend_methods import *
from config import *
import hashlib
import time
import atexit
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler


# Set up a scheduler to run daily and weekly updates
sched = BackgroundScheduler()
sched.add_job(daily_update, 'cron', hour='0', minute='0', second='0')
sched.add_job(update_models, 'cron',day_of_week='1', hour='0', minute='0', second='0')
sched.start()
atexit.register(lambda: sched.shutdown())


app = Flask(__name__)

@app.route("/")
@app.route("/about")
@app.route("/login")
@app.route("/patientInfo")
@app.route("/patientInfo/history")
@app.route("/patientInfo/details")
@app.route("/patientInfo/status")
@app.route("/modelInfo")
@app.route("/modelInfo/clusters")
@app.route("/modelInfo/models")
@app.route("/modelInfo/stats")
@app.route("/getPatient")
def index():
    return send_file('templates/index.html')


client = MongoClient('localhost:27017')
db = client.icu_db
patients_collection = db.patient_details_collection
features_collection = db.all_feats_collection
users_collection = db.users
models_collection = db.models_collection


@app.route("/getPatient",methods=['POST'])
"""
Gets a patient's personal details and history by ID
"""
def getPatient():
    json_data = request.get_json()
    patientID = hashlib.md5(str(int(json_data['id'])).encode()).hexdigest()
    details = patients_collection.find_one({'adm_num':patientID})
    if details is None:
        return "null"
    details = dumps(details)
    history = json.dumps(getPatientHistory(patientID))
    status = dumps(getPatientStatus(patientID))
    return jsonify(details=details, history=history, status=status)

@app.route("/getPatientAt",methods=['POST'])
def getPatientAt():
    """
    Gets a patient's history at a specific date.
    """
    json_data = request.get_json()
    patientID = hashlib.md5(str(int(json_data['id'])).encode()).hexdigest()
    day_num = json_data['day_num']
    historyAt = features_collection.find_one({'temp_adm_num':patientID, 'dayNum':day_num})
    if historyAt is None:
        return "null"
    df = pd.DataFrame(dict(historyAt), index=[0])
    df = df[FEATURES_XGB].drop(['age2', 'gender2', 'dayNum', 'isVentilated'], axis=1)
    return dumps(df.to_dict('records'))

@app.route("/getStatus",methods=['POST'])
def getStatus():
    """
    gets patient current status and latest records
    """
    json_data = request.get_json()
    patientID = hashlib.md5(str(int(json_data['id'])).encode()).hexdigest()
    doc = getPatientStatus(patientID)
    status = {}
    if doc is None:
        return "null"
    df = pd.DataFrame(dict(doc), index=[0])
    status['cluster'] = int(df.iloc[0]['cluster'])
    status['mortality_prediction'] = df.iloc[0]['mortality_prediction']
    status['days_prediction'] = df.iloc[0]['next3days_mortality_prediction']
    status['justification'] = dict(eval(df.iloc[0]['justification']))
    measurements = df[FEATURES_XGB].drop(['age2', 'gender2', 'dayNum', 'isVentilated'], axis=1)
    return jsonify(status=status, measurements=measurements.to_dict('records'))

@app.route("/getClustersInfo",methods=['POST'])
def getClustersInfo():
    """
    load from database last update and most important features
    of clustering model
    """
    doc = list(models_collection.find({'name':'clustering_model'}))[0]
    if doc is None:
        return "null"
    lastUpdate = doc['last_trained']
    df = pd.DataFrame.from_dict(doc['properties'])
    main_features = df.tail(1).to_dict('records')[0]
    print(main_features)
    return jsonify(lastUpdate=lastUpdate, main_features=main_features)

@app.route("/getModelsInfo",methods=['POST'])
def getModelsInfo():
    """
    loads from database last update, and model information (auc curve,
     significant features) of mortality models.
     """
    mortality_model, days_forward_model = getModelsMetaData()
    if mortality_model is None:
        return "null"
    shutil.copy(mortality_model['auc_path'], projectPath+'/static/img/auc.png' )
    shutil.copy(mortality_model['feature_importance_path'], projectPath+'/static/img/features.png' )
    return jsonify(mortality_model=mortality_model, days_forward_model=days_forward_model)


@app.route("/attempLogin",methods=['POST'])
def attempLogin() :
    """
    verify login credentials with DB.
    """
    content = request.get_json()
    userID = content['id']
    password = content['password']
    user = users_collection.find_one({'id':userID})
    if (user!=None and user['password']==password):
        return jsonify(name=user['name'], position=user['position'], status='OK')
    return jsonify(status=403,message='ID or password incorrect')

@app.route("/dailyUpdate", methods=['POST'])
def dailyUpdate():
    """
    run daily update routine defined in backend_methods.py
    """
    daily_update()
    return jsonify(status="OK")

@app.route("/updateModels", methods=['POST'])
def updateModels():
    """
    run periodicall model retraining defined in update_models()
    method in backend_methods.py
    """
    update_models()
    return jsonify(status="OK")


if ("__main__"==__name__) :
    app.run('127.0.0.1',8080,debug=True)
