import pytest
from fastapi.testclient import TestClient
import sys
import os

# Agregar el directorio padre al path para importar main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

client = TestClient(app)

def test_api_running():
    """Test para verificar que la API está funcionando"""
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"message": "API is working"}

def test_predict_survival_male_third_class():
    """Test predicción: hombre joven de tercera clase (probablemente no sobrevive)"""
    body = {
        "edad": 22,
        "clase": "Third",
        "sexo": "M"
    }
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    assert "nivel de salades" in response.json()
    print(f"Predicción: {response.json()}")

def test_predict_survival_female_first_class():
    """Test predicción: mujer de primera clase (probablemente sobrevive)"""
    body = {
        "edad": 35,
        "clase": "First",
        "sexo": "F"
    }
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    assert "nivel de salades" in response.json()
    result = response.json()["nivel de salades"]
    print(f"Predicción: {result}")
    # Mujeres de primera clase tenían alta probabilidad de sobrevivir
    assert result == "De buenas"

def test_predict_survival_child():
    """Test predicción: niño"""
    body = {
        "edad": 5,
        "clase": "Second",
        "sexo": "M"
    }
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    assert "nivel de salades" in response.json()
    print(f"Predicción: {response.json()}") 




    