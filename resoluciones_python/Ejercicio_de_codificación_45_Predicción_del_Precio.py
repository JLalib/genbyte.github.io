import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Simulación de datos
np.random.seed(42)
data = pd.DataFrame({
    'Superficie': np.random.uniform(50, 150, 200),
    'Habitaciones': np.random.randint(1, 6, 200),
    'Antigüedad': np.random.randint(0, 50, 200),
    'Distancia_centro': np.random.uniform(1, 20, 200),
    'Baños': np.random.randint(1, 4, 200),
    'Precio': np.random.uniform(100000, 500000, 200)
})

# ---------------------------------------------
# Función para entrenamiento y evaluación
# ---------------------------------------------
def entrenar_modelo(data):
    """Entrena un modelo de regresión lineal para predecir el precio de una vivienda."""
    
    # Separar características y variable objetivo
    X = data[['Superficie', 'Habitaciones', 'Antigüedad', 'Distancia_centro', 'Baños']]
    y = data['Precio']

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Evaluación del modelo
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
    print(f"R² del modelo: {r2:.2f}")

    return modelo

# ---------------------------------------------
# Entrenar el modelo
# ---------------------------------------------
modelo_entrenado = entrenar_modelo(data)

# ---------------------------------------------
# Crear una nueva vivienda para predecir el precio
# ---------------------------------------------
nueva_vivienda = pd.DataFrame({
    'Superficie': [120],
    'Habitaciones': [3],
    'Antigüedad': [10],
    'Distancia_centro': [5],
    'Baños': [2]
})

# Realizar la predicción
prediccion = modelo_entrenado.predict(nueva_vivienda)

# Interpretar el resultado
print(f"El precio estimado de la vivienda es: ${prediccion[0]:,.2f}")