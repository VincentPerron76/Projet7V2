import pytest
from api import app  # Assurez-vous que l'import de votre fichier api.py est correct
import json

# Créer un client de test Flask
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Test de la route / (accueil)
def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Bienvenue sur l'API de prédiction" in response.data.decode("utf-8")  # Décoder en UTF-8 pour les caractères spéciaux

# Test de la route /bonjour
def test_bonjour(client):
    response = client.get("/bonjour?SK_ID_CURR=234285")
    assert response.status_code == 200
    assert "vous avez demandé l'identifiant 12345" in response.data.decode("utf-8")  # Décoder en UTF-8 pour les caractères spéciaux

# Test de la route /predict avec un client valide prêt remboursé
def test_predict_valid(client):
    response = client.get("/predict?SK_ID_CURR=234285")
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert "SK_ID_CURR" in response_json
    assert "prediction" in response_json
    assert "probability" in response_json

# Test de la route /predict avec un client valide prêt non remboursé
def test_predict_valid(client):
    response = client.get("/predict?SK_ID_CURR=295120")
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert "SK_ID_CURR" in response_json
    assert "prediction" in response_json
    assert "probability" in response_json    

# Test de la route /predict avec un client invalide
def test_predict_invalid(client):
    response = client.get("/predict?SK_ID_CURR=9999999")
    assert response.status_code == 404
    response_json = json.loads(response.data)
    assert "error" in response_json
    assert "Client avec id 9999999 introuvable." in response_json["error"]

# Test de la route /predict sans SK_ID_CURR
def test_predict_missing_id(client):
    response = client.get("/predict")
    assert response.status_code == 400
    response_json = json.loads(response.data)
    assert "error" in response_json
    assert "SK_ID_CURR est requis en tant que paramètre GET." in response_json["error"]
