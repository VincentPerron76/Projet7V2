import pytest
import json
from api import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.mark.parametrize("client_id, expected_prediction", [
    (241603, 0),
    (350714, 0),
    (211868, 0),
    (268880, 0),
    (305344, 0),
    (180213, 1),
    (398182, 1),
    (443859, 1),
    (259596, 1),
    (187836, 1)
])

def test_predict_valid(client, client_id, expected_prediction):
    response = client.get(f"/predict?SK_ID_CURR={client_id}")
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json["SK_ID_CURR"] == str(client_id)
    assert response_json["prediction"] == expected_prediction
    assert "probability" in response_json