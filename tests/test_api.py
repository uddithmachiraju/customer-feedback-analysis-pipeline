import pytest 
from api.app import app 
import warnings 

warnings.filterwarnings("ignore") 

@pytest.fixture
def client():
    app.config['Testing'] = True 
    with app.test_client() as client:
        yield client 

def test_home_page(client):
    """
    Tests if the home page is working
    """
    response = client.get("/") 
    assert response.status_code == 200 
    assert b"customer-feedback-analysis-pipeline" in response.data

def test_predictions(client):
    """
    Testing if the model predicts an output.
    """
    test_input = "Worst experience ever, will never order again."
    response = client.post(
        "/", data = {
            "text" : test_input
        }
    )
    assert response.status_code == 200 
    assert b"Predicted Sentiment:" in response.data 

# To run this file : python -m pytest tests/