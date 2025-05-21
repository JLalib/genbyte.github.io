from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def entrenar_y_evaluar_random_forest(X_train, y_train, X_test, y_test):
    # Crear el modelo con los parámetros requeridos
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Predecir sobre el conjunto de prueba
    predicciones = modelo.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, predicciones)
    matriz_confusion = confusion_matrix(y_test, predicciones)
    reporte = classification_report(y_test, predicciones, target_names=["Clase 0", "Clase 1", "Clase 2"])

    # Retornar los resultados en un diccionario
    return {
        "predicciones": predicciones,
        "accuracy": accuracy,
        "matriz_confusion": matriz_confusion,
        "reporte": reporte
    }