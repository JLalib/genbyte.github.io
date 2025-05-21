import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --------------------------
# Simulador de jugadores
# --------------------------
class GameSimulator:
    def __init__(self, n_players=200, random_state=42):
        np.random.seed(random_state)
        self.n_players = n_players
        self.X = []
        self.y = []

    def run(self):
        for _ in range(self.n_players):
            partidas_ganadas = np.random.rand()
            horas_jugadas = np.random.rand()
            precision = np.random.rand()
            velocidad = np.random.rand()
            estrategia = np.random.rand()

            features = [partidas_ganadas, horas_jugadas, precision, velocidad, estrategia]

            # Heurística para etiquetar como profesional
            if (partidas_ganadas + precision + velocidad + estrategia) > 3.0 and horas_jugadas > 0.5:
                label = 1  # Profesional
            else:
                label = 0  # Casual

            self.X.append(features)
            self.y.append(label)

        self.X = np.array(self.X)
        self.y = np.array(self.y)

# --------------------------
# Clasificador de jugadores profesionales
# --------------------------
class ProPlayerClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, player_stats):
        player_stats = np.array(player_stats).reshape(1, -1)  # Convertir lista o array en matriz
        return self.model.predict(player_stats)[0]

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)