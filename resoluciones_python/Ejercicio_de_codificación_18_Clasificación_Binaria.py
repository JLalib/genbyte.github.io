import pandas as pd
from sklearn.linear_model import LogisticRegression

def regresion_logistica(datos):
    # Separar las características (X) y la variable objetivo (y)
    X = datos[['Edad', 'Colesterol']]
    y = datos['Enfermedad']
    
    # Crear y entrenar el modelo de regresión logística
    modelo = LogisticRegression()
    modelo.fit(X, y)
    
    # Retornar el modelo entrenado
    return modelo