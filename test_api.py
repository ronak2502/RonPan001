import requests
import pytest

API_URL = "http://127.0.0.1:5000/predict"

def test_rest_api():
    phrases = ["Das ist ein Test.", "wie gets", "Was ist die Hauptstadt von Deutschland?", "individuelle verpackung", "sandstrahlen von holz Lohn"]

    response = requests.post(API_URL, json={"phrases": phrases})
    assert response.status_code == 200
    predictions = response.json()
    assert len(predictions) == len(phrases)
    assert all(isinstance(prediction, str) for prediction in predictions)

def test_predict_single_phrase():
    input_data = {"phrases": ["This is a single phrase."]}
    response = requests.post(API_URL, json=input_data)
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_predict_special_characters():
    input_data = {"phrases": ["@#$%^&*()_+"]}
    response = requests.post(API_URL, json=input_data)
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_predict_long_phrase():
    input_data = {"phrases": ["This is a long sentence. " * 100]}
    response = requests.post(API_URL, json=input_data)
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_predict_non_german_phrases():
    input_data = {"phrases": ["Hello", "Machine Learning is amazing."]}
    response = requests.post(API_URL, json=input_data)
    assert response.status_code == 200

