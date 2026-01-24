from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np 
import pandas as pd


app = FastAPI()

class Prediccion(BaseModel):
    edad: int
    clase: str
    sexo: str

@app.get("/test")
def testApi():
    return {"message": "API is working"}