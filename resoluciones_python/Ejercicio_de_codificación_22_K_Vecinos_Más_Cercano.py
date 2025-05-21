# Importar librerías necesarias
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Función que entrena un modelo KNN con k vecinos
def knn_clasificacion(datos, k=3):
    # Separar características (X) y etiquetas (y)
    X = datos[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = datos['species']

    # Crear y entrenar el modelo
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X, y)

    return modelo