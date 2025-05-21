from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test):
    # Crear el modelo con los parámetros especificados
    modelo = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Realizar predicciones
    predicciones = modelo.predict(X_test)

    # Calcular métricas de evaluación
    accuracy = accuracy_score(y_test, predicciones)
    matriz = confusion_matrix(y_test, predicciones)
    reporte = classification_report(y_test, predicciones)

    # Retornar los resultados
    return {
        "predicciones": predicciones,
        "accuracy": accuracy,
        "matriz_confusion": matriz,
        "reporte": reporte
    }