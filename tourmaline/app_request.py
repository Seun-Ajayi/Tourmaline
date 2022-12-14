import json
import requests


url = "https://elbaite-tourmaline.herokuapp.com/model/"
sample = {
 'age': 52,
 'workclass': 'Self-emp-not-inc',
 'fnlgt': 209642,
 'education': 'HS-grad',
 'education-num': 9,
 'marital-status': 'Married-civ-spouse',
 'occupation': 'Exec-managerial',
 'relationship': 'Husband',
 'race': 'White',
 'sex': 'Male',
 'capital-gain': 0,
 'capital-loss': 0,
 'hours-per-week': 45,
 'native-country': 'United-States',
}

headers = {"content-type": "application/json"} 
response = requests.post(url, data=json.dumps(sample), headers=headers)
print("Status Code:", response.status_code)
print("Prediction:", response.text)