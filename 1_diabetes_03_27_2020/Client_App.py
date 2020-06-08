import json
import requests

def predict_diabetes(bmi,age,glucose):
    url = 'http://127.0.0.1:5000/PREDICT'
    data = {"BMI":bmi, "Age":age, "Glucose":glucose}
    data_json = json.dumps(data)
    headers = {'Content-type':'application/json'}
    response = requests.post(url,data=data_json,headers=headers)
    result = json.loads(response.text)
    return result

if __name__ == '__main__':
    bmi = input("BMI? ")
    age = input("Age? ")
    glucose = input("Glucose? ")
    predictions = predict_diabetes(bmi,age,glucose)
    print("====================================")
    print('Diabetic' if predictions['prediction'] == 1 else 'Not Diabetic')
    print(f"Confidence: {predictions['confidence']} %")
    print("====================================")