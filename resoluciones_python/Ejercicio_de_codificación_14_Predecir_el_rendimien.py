# Importar librerías necesarias
import pandas as pd
from sklearn.linear_model import LinearRegression

# Clase Player
class Player:
    def __init__(self, name, avg_session_time, avg_actions_per_min, avg_kills_per_session, victories=None):
        self.name = name
        self.avg_session_time = avg_session_time
        self.avg_actions_per_min = avg_actions_per_min
        self.avg_kills_per_session = avg_kills_per_session
        self.victories = victories

    def to_features(self):
        return [self.avg_session_time, self.avg_actions_per_min, self.avg_kills_per_session]

# Clase PlayerDataset
class PlayerDataset:
    def __init__(self, players):
        self.players = players

    def get_feature_matrix(self):
        return [player.to_features() for player in self.players]

    def get_target_vector(self):
        return [player.victories for player in self.players if player.victories is not None]

# Clase VictoryPredictor
class VictoryPredictor:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, dataset: PlayerDataset):
        X = dataset.get_feature_matrix()
        y = dataset.get_target_vector()
        self.model.fit(X, y)

    def predict(self, player: Player):
        features = [player.to_features()]
        return self.model.predict(features)[0]

# Datos de entrenamiento
players = [
    Player("Alice", 40, 50, 6, 20),
    Player("Bob", 30, 35, 4, 10),
    Player("Charlie", 50, 60, 7, 25),
    Player("Diana", 20, 25, 2, 5),
    Player("Eve", 60, 70, 8, 30)
]

# Crear dataset y entrenar modelo
dataset = PlayerDataset(players)
predictor = VictoryPredictor()
predictor.train(dataset)

# Jugador de prueba
test_player = Player("TestPlayer", 45, 55, 5)

# Predicción
predicted_victories = predictor.predict(test_player)

# Mostrar resultado
print(f"Victorias predichas para {test_player.name}: {predicted_victories:.2f}")