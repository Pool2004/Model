from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np 
import pandas as pd
import pickle
import os


app = FastAPI()

# edad Usuario 

class Prediccion(BaseModel):
    edad: int
    clase: str
    sexo: str

# Modelo de sobrevivientes del Titanic

# Obtener la ruta absoluta del archivo del modelo
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'titanic_model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict_survival(data: Prediccion):
    # Convertir datos de entrada a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Preprocesamiento 
    input_data['sexo'] = input_data['sexo'].map({'M': 0, 'F': 1})
    input_data['clase'] = input_data['clase'].map({'First': 1, 'Second': 2, 'Third': 3})
    input_data = input_data.fillna(input_data.mean())
    features = input_data[['edad', 'clase', 'sexo']]
    prediction = model.predict(features)
    survival = 'De buenas' if prediction[0] == 1 else 'Pailas se muere'
    return {"nivel de salades": survival}

@app.get("/test")
def testApi():
    return {"message": "API is working"}