from flask import Flask, send_file
app = Flask(__name__)

@app.route("/getPatient",methods=['POST'])
def getPatient():
    json_data = request.get_json(force=True)
    id = json_data['id']
    patient = db.patients.find_one({patientID:id})
    return patient 



@app.route("/insertPatient",methods=['POST'])
def insertPatient() :
    json_data = request.get_json(force=True)
    id = json_data['id']
    firstName = json_data['firstName']
    lastName = json_data['lastName']
    db.patients.insert_one({'id':id, 'firstName':firstName, 'lastName':lastName})
    return jsonify(status='OK',message='inserted successfully')
