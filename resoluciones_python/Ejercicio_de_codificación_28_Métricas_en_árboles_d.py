from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def entrenar_y_evaluar_arbol(X_train, y_train, X_test, y_test):
    # Crear el modelo con random_state fijo
    modelo = DecisionTreeClassifier(random_state=42)

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Predecir valores en el conjunto de prueba
    predicciones = modelo.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, predicciones)
    matriz_confusion = confusion_matrix(y_test, predicciones)
    reporte = classification_report(y_test, predicciones, target_names=['Setosa', 'Versicolor', 'Virginica'])

    # Devolver las métricas en un diccionario
    return {
        "predicciones": predicciones,
        "accuracy": accuracy,
        "matriz_confusion": matriz_confusion,
        "reporte": reporte
    }