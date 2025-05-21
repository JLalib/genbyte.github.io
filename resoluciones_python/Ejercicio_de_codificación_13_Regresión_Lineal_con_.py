import pandas as pd
from sklearn.linear_model import LinearRegression

def regresion_ventas(datos):
    # Variables independientes (X) e independiente (y)
    X = datos[['TV', 'Radio', 'Periodico']]
    y = datos['Ventas']
    
    # Crear el modelo de regresión lineal
    modelo = LinearRegression()
    
    # Ajustar el modelo con los datos
    modelo.fit(X, y)
    
    # Retornar el modelo entrenado
    return modelo

    