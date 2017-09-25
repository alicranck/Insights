from flask import Flask, send_file, request, jsonify, redirect
import pymongo
from pymongo import MongoClient
import json,requests
from bson import Binary, Code
from bson.json_util import dumps


app = Flask(__name__)

@app.route("/")
@app.route("/about")
@app.route("/login")
@app.route("/patientInfo")
@app.route("/patientInfo/history")
@app.route("/patientInfo/details")
@app.route("/getPatient")
def index():
    return send_file('templates/index.html')


client = MongoClient('localhost:27017')
db = client.appDB
patients_collection = db.patients
users_collection = db.users
models_collection = db.models


@app.route("/getPatient",methods=['POST'])
def getPatient():
    json_data = request.get_json()
    patientID = json_data['id']
    entry = dumps(patients_collection.find_one({'id':patientID}))
    return entry

@app.route("/getModelInfo",methods=['POST'])
def getModelInfo():
    modelName = 'test'
    entry = dumps(models_collection.find_one({'name':modelName}))
    print(entry)
    return entry


@app.route("/insertPatient",methods=['POST'])
def insertPatient() :
    content = request.get_json()
    patientID = content['id']
    firstName = content['firstName']
    lastName = content['lastName']
    if (db.patients.find_one({'id':patientID})!=None):
        return jsonify(status='OK',message='Patient Exists')
    db.patients.insert_one({'id':patientID, 'firstName':firstName, 'lastName':lastName})
    return jsonify(status='OK',message='inserted successfully')

@app.route("/attempLogin",methods=['POST'])
def attempLogin() :
    content = request.get_json()
    userID = content['id']
    password = content['password']
    user = users_collection.find_one({'id':userID})
    if (user!=None and user['password']==password):
        return jsonify(name=user['name'], position=user['position'], status='OK')
    return jsonify(status=403,message='ID or password incorrect')


if ("__main__"==__name__) :
    app.run('127.0.0.1',8085,debug=True)
