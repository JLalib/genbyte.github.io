import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class IoTKNNClassifier:
    def __init__(self, n_neighbors=3, n_samples=50):
        """
        Inicializa el modelo con n vecinos y genera los datos sintéticos automáticamente.
        """
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        # Generación de datos sintéticos
        np.random.seed(42)
        paquetes_por_segundo = np.random.randint(10, 1000, self.n_samples)
        bytes_por_paquete = np.random.randint(50, 1500, self.n_samples)
        protocolo = np.random.randint(1, 4, self.n_samples)  # 1=TCP, 2=UDP, 3=HTTP
        seguro = np.random.randint(0, 2, self.n_samples)     # 0=peligroso, 1=seguro

        # DataFrame para facilidad de uso
        self.df = pd.DataFrame({
            "paquetes_por_segundo": paquetes_por_segundo,
            "bytes_por_paquete": bytes_por_paquete,
            "protocolo": protocolo,
            "seguro": seguro
        })

        # División de datos
        self.X = self.df.drop(columns=["seguro"])
        self.y = self.df["seguro"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train(self):
        """
        Entrena el modelo KNN con los datos de entrenamiento.
        """
        self.knn.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evalúa el modelo con los datos de prueba y retorna la precisión.
        """
        y_pred = self.knn.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def predict(self, nuevo_dispositivo):
        """
        Predice si un nuevo dispositivo es seguro o peligroso.
        nuevo_dispositivo: lista o array con [paquetes_por_segundo, bytes_por_paquete, protocolo]
        """
        prediccion = self.knn.predict([nuevo_dispositivo])
        return int(prediccion[0])