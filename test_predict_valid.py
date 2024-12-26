import pytest
import json
from api import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.mark.parametrize("client_id, expected_prediction", [
    (151142, 1),
    (234285, 0),
    (280185, 0),
    (136865, 0),
    (372541, 1),
    (447393, 0),
    (415174, 0),
    (391249, 0),
    (361246, 0),
    (101768, 1),
    (377364, 1),
    (129865, 0),
    (291249, 0),
    (295120, 0),
    (367643, 1),
])
def test_predict_valid(client, client_id, expected_prediction):
    response = client.get(f"/predict?SK_ID_CURR={client_id}")
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json["SK_ID_CURR"] == str(client_id)
    assert response_json["prediction"] == expected_prediction
    assert "probability" in response_json