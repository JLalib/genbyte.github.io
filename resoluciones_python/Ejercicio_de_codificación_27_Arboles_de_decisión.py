# Importar librerías necesarias
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# --------------------------------------
# Función para entrenar un árbol de decisión
# --------------------------------------
def entrenar_arbol_decision(X_train, y_train, X_test):
    # Crear el modelo con random_state para reproducibilidad
    modelo = DecisionTreeClassifier(random_state=42)

    # Entrenar el modelo con los datos
    modelo.fit(X_train, y_train)

    # Predecir las clases para el conjunto de prueba
    predicciones = modelo.predict(X_test)

    return predicciones

# --------------------------------------
# Ejemplo de uso con el dataset Iris
# --------------------------------------
if __name__ == "__main__":
    # Cargar los datos
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Dividir en entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Llamar a la función implementada
    predicciones = entrenar_arbol_decision(X_train, y_train, X_test)

    # Mostrar algunas predicciones y valores reales
    print("Predicciones del Árbol de Decisión:", predicciones[:10])
    print("Valores reales:                    ", y_test[:10])