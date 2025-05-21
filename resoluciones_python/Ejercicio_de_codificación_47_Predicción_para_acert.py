import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------------------
# 1. Generar combinaciones de lotería
# -----------------------------------------
def generar_series(num_series):
    series = []
    for _ in range(num_series):
        combinacion = np.random.choice(range(1, 50), size=6, replace=False)
        series.append(sorted(combinacion))
    return series

# -----------------------------------------
# 2. Entrenar modelo con datos simulados
# -----------------------------------------
def entrenar_modelo():
    np.random.seed(42)
    
    # Generar 1000 combinaciones
    combinaciones = generar_series(1000)

    # Crear etiquetas: 10% éxito, 90% fracaso
    etiquetas = [1]*100 + [0]*900
    np.random.shuffle(etiquetas)

    # Convertir a DataFrame
    df = pd.DataFrame(combinaciones, columns=[f'num{i+1}' for i in range(6)])
    df['resultado'] = etiquetas

    # Separar características y etiquetas
    X = df.drop('resultado', axis=1)
    y = df['resultado']

    # Entrenar modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    # (Opcional) Evaluación en consola
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = modelo.predict(X_test)
    print("Reporte del modelo:")
    print(classification_report(y_test, y_pred))

    return modelo

# -----------------------------------------
# 3. Predecir mejor combinación
# -----------------------------------------
def predecir_mejor_serie(modelo, num_series):
    series = generar_series(num_series)
    df_series = pd.DataFrame(series, columns=[f'num{i+1}' for i in range(6)])
    probs = modelo.predict_proba(df_series)[:, 1]  # Probabilidad de éxito

    max_index = np.argmax(probs)
    mejor_serie = series[max_index]
    mejor_probabilidad = probs[max_index]

    return mejor_serie, mejor_probabilidad