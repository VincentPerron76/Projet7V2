import pytest
import json
from api import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.mark.parametrize("client_id, expected_prediction", [
    (234285, 0),
    (235876, 0),
    (130522, 0),
    (112770, 0),
    (249059, 0),
    (151142, 1),
    (280185, 1),
    (136865, 1),
    (372541, 1),
    (447393, 1),
])

def test_predict_valid(client, client_id, expected_prediction):
    response = client.get(f"/predict?SK_ID_CURR={client_id}")
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json["SK_ID_CURR"] == str(client_id)
    assert response_json["prediction"] == expected_prediction
    assert "probability" in response_json