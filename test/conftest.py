import pytest
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np

@pytest.fixture(scope="session", autouse=True)
def setup_test_model():
    """
    Crea un modelo pequeño mockeado solo para testing.
    Se ejecuta una vez por sesión de tests.
    """
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'titanic_model.pkl')
    
    # Si el modelo no existe, crear uno mock pequeño
    if not os.path.exists(model_path):
        print("Creando modelo mock para tests...")
        # Crear un modelo simple con datos mínimos
        X = np.array([[22, 3, 0], [35, 1, 1], [5, 2, 0]])
        y = np.array([0, 1, 1])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
        model.fit(X, y)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modelo mock creado en {model_path}")
    
    yield
    
    # Opcional: limpiar el modelo después de los tests si lo prefieres
    # os.remove(model_path)
