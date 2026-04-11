import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app

client = TestClient(app)

@patch("src.main.query_instance.collection", new_callable=MagicMock)
def test_health_check(mock_collection):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@patch("src.main.query_instance.search_context")
@patch("src.main.query_instance.generate_response")
def test_query_endpoint(mock_generate, mock_search):
    mock_search.return_value = [
        {"text": "Relevant context", "metadata": {}, "distance": 0.1}
    ]
    mock_generate.return_value = "This is a fake response."
    
    response = client.post("/query", json={"query": "What is Python?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is Python?"
    assert data["response"] == "This is a fake response."
    assert len(data["sources"]) == 1
    mock_search.assert_called_once_with("What is Python?", 3)
    mock_generate.assert_called_once()
