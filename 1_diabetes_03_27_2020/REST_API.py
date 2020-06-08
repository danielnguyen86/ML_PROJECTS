import pickle
from flask import Flask, request,json, jsonify
import numpy as np

app = Flask(__name__)

# load the saved model
loaded_model = pickle.load(open('TRAINED_MODELS/diabetes.sav','rb'))

@app.route('/PREDICT',methods=['POST'])
def predict():

    # get features to predict
    features = request.json

    # create the features list for prediction
    features_list = [[features['Glucose'],features['BMI'],features['Age']]]

    # get the prediction class
    prediction = loaded_model.predict(features_list)

    # get the prediction probabilities
    confidence = loaded_model.predict_proba(features_list)

    # formulate the response to return to client
    response ={}
    response['prediction'] = int(prediction[0])
    response['confidence'] = str(round(confidence[0][0]*100,2))

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)


### NOTE ###
'''
The preceding code snippet accomplishes the following:
■ Creates a route /PREDICT using the route decorator.
■ The route is accessible through the POST verb.
■ To make a prediction, users make a call to this route and pass in the various features using a JSON string.
■ The result of the prediction is returned as a JSON string.

---------------------------------------------------------------------------------
To test the REST API:
1- run it in Terminal by entering the following command:
$ python REST_API.py

2- run it in another Terminal by entering the following command:
$ curl -H "Content-type: application/json" -X POST http://127.0.0.1:5000/PREDICT -d '{"BMI":30, "Age":29,"Glucose":100 }'

---------------------------------------------------------------------------------

'''