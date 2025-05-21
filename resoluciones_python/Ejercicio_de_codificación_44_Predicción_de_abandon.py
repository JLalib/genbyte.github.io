import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Simulación de datos
np.random.seed(42)
data = pd.DataFrame({
    'Edad': np.random.randint(18, 30, 200),
    'Horas_estudio': np.random.uniform(0, 30, 200),
    'Asistencia': np.random.uniform(50, 100, 200),
    'Promedio': np.random.uniform(5, 10, 200),
    'Uso_online': np.random.uniform(0, 15, 200),
    'Abandono': np.random.choice([0, 1], size=200, p=[0.7, 0.3])
})

# ---------------------------
# Función completa
# ---------------------------
def entrenar_modelo(data):
    """
    Entrena un modelo de DecisionTreeClassifier con los datos de los estudiantes.
    
    Parámetros:
        data (pd.DataFrame): Datos de entrada con variables predictoras y columna 'Abandono'.
        
    Retorna:
        modelo entrenado (DecisionTreeClassifier)
    """

    # Separar características y etiqueta
    X = data.drop('Abandono', axis=1)
    y = data['Abandono']

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred)

    print(f"Precisión del modelo: {accuracy:.2f}")
    print("Reporte de clasificación:")
    print(reporte)

    return modelo

# Entrenar el modelo con los datos generados
modelo_entrenado = entrenar_modelo(data)

# Crear un nuevo estudiante para predecir si abandonará o no
nuevo_estudiante = pd.DataFrame({
    'Edad': [20],
    'Horas_estudio': [10],
    'Asistencia': [80],
    'Promedio': [7.5],
    'Uso_online': [5]
})

# Realizar la predicción
prediccion = modelo_entrenado.predict(nuevo_estudiante)

# Interpretar el resultado
resultado = "Abandonará" if prediccion[0] == 1 else "Seguirá estudiando"
print(f"El estudiante probablemente: {resultado}")