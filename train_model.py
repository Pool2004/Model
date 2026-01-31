import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Crear datos de entrenamiento de ejemplo
data = {
    'edad': [22, 38, 26, 35, 35, 54, 2, 27, 14, 4, 58, 20, 39, 14, 55, 2, 31, 35, 34, 15,
             28, 8, 38, 19, 40, 66, 28, 42, 21, 18, 14, 40, 27, 3, 19, 18, 7, 21, 49, 29],
    'clase': [3, 1, 3, 1, 3, 1, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 3,
              3, 3, 1, 3, 1, 1, 3, 1, 3, 3, 3, 1, 3, 3, 3, 2, 3, 3, 1, 1],
    'sexo': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
             0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
    'survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1,
                 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Separar características y objetivo
X = df[['edad', 'clase', 'sexo']]
y = df['survived']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
accuracy = model.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy:.2f}")

# Guardar el modelo
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo guardado exitosamente como 'titanic_model.pkl'")
