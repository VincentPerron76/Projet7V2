import pytest
import requests
import json
from api import app  # Import your application Flask

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Bienvenue sur l'API de prediction !" in response.data  # or .decode('utf-8')

def test_bonjour(client):
    response = client.get("/bonjour?SK_ID_CURR=151142")
    assert response.status_code == 200
    assert b"vous avez demande l'identifiant 151142" in response.data  # or .decode('utf-8')

@pytest.mark.parametrize("client_id, expected_prediction", [
    (151142, 1),
    (234285, 0),
])
def test_predict_valid(client, client_id, expected_prediction):
    response = client.get(f"/predict?SK_ID_CURR={client_id}")
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json["SK_ID_CURR"] == str(client_id)
    assert response_json["prediction"] == expected_prediction
    assert "probability" in response_json

def test_predict_invalid_id(client):
    response = client.get("/predict?SK_ID_CURR=9999999")
    assert response.status_code == 404
    response_json = json.loads(response.data.decode('utf-8'))
    assert "error" in response_json
    assert "Client avec id 9999999 introuvable." in response_json["error"]

def test_predict_missing_id(client):
    response = client.get("/predict")
    assert response.status_code == 400
    response_json = json.loads(response.data.decode('utf-8'))
    assert "error" in response_json
    assert "SK_ID_CURR est requis en tant que parametre GET." in response_json["error"]