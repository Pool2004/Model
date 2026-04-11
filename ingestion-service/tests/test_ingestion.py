import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch("src.main.producer_instance.send_document")
def test_upload_document_success(mock_send):
    # Simular que el envío a Kafka funcionó
    mock_send.return_value = True
    
    response = client.post(
        "/upload",
        json={"text": "Este es un documento de prueba", "metadata": {"source": "test"}}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert data["status"] == "queued"
    mock_send.assert_called_once()

@patch("src.main.producer_instance.send_document")
def test_upload_document_failure(mock_send):
    # Simular fallo en Kafka
    mock_send.return_value = False
    
    response = client.post(
        "/upload",
        json={"text": "Prueba fallo"}
    )
    
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to queue document for processing"

def test_upload_document_empty():
    response = client.post(
        "/upload",
        json={"text": "   "}
    )
    assert response.status_code == 400
