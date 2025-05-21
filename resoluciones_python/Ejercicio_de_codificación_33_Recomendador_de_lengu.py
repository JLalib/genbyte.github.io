import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------
# Función para generar dataset artificial
# -------------------------------------
def generate_dataset(n_samples=100, random_state=42):
    np.random.seed(random_state)
    X = []

    y = []

    for _ in range(n_samples):
        velocidad = np.random.rand()
        mantenimiento = np.random.rand()
        bibliotecas = np.random.rand()
        tipo_app = np.random.randint(0, 3)  # 0=web, 1=movil, 2=desktop
        rendimiento = np.random.rand()

        features = [velocidad, mantenimiento, bibliotecas, tipo_app, rendimiento]

        # Reglas heurísticas para asignar etiquetas
        if rendimiento > 0.8 and tipo_app == 2:
            label = 3  # C++
        elif tipo_app == 0 and mantenimiento > 0.6:
            label = 0  # Python
        elif tipo_app == 1 and velocidad > 0.6:
            label = 1  # JavaScript
        else:
            label = 2  # Java

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)

# -------------------------------------
# Clase principal del modelo
# -------------------------------------
class LanguagePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_map = {
            0: "Python",
            1: "JavaScript",
            2: "Java",
            3: "C++"
        }

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, features: np.ndarray) -> str:
        features_reshaped = features.reshape(1, -1)  # Convertir a matriz de una fila
        predicted_class = self.model.predict(features_reshaped)[0]
        return self.label_map[predicted_class]