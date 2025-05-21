import numpy as np
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Clase que representa los datos de una partida
# -----------------------------
class PlayerMatchData:
    def __init__(self, kills, deaths, assists, damage_dealt, damage_received, healing_done, objective_time, won=None):
        self.kills = kills
        self.deaths = deaths
        self.assists = assists
        self.damage_dealt = damage_dealt
        self.damage_received = damage_received
        self.healing_done = healing_done
        self.objective_time = objective_time
        self.won = won  # 1 si ganó, 0 si no

    def to_dict(self, include_target=True):
        data = {
            'kills': self.kills,
            'deaths': self.deaths,
            'assists': self.assists,
            'damage_dealt': self.damage_dealt,
            'damage_received': self.damage_received,
            'healing_done': self.healing_done,
            'objective_time': self.objective_time
        }
        if include_target and self.won is not None:
            data['won'] = self.won
        return data

# -----------------------------
# Generación de datos sintéticos
# -----------------------------
def generate_synthetic_data(n=100):
    data = []
    for _ in range(n):
        kills = np.random.poisson(5)
        deaths = np.random.poisson(3)
        assists = np.random.poisson(2)
        damage_dealt = kills * 300 + np.random.normal(0, 100)
        damage_received = deaths * 400 + np.random.normal(0, 100)
        healing_done = np.random.randint(0, 301)
        objective_time = np.random.randint(0, 121)
        won = int(damage_dealt > damage_received and kills > deaths)
        match = PlayerMatchData(kills, deaths, assists, damage_dealt, damage_received, healing_done, objective_time, won)
        data.append(match)
    return data

# -----------------------------
# Modelo de predicción con regresión logística
# -----------------------------
class VictoryPredictor:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, data):
        X = [list(d.to_dict(include_target=False).values()) for d in data]
        y = [d.won for d in data]
        self.model.fit(X, y)

    def predict(self, player: PlayerMatchData):
        X = [list(player.to_dict(include_target=False).values())]
        return self.model.predict(X)[0]

# -----------------------------
# Ejemplo de uso
# -----------------------------
if __name__ == "__main__":
    # Generar datos de entrenamiento
    training_data = generate_synthetic_data(150)

    # Entrenar el modelo
    predictor = VictoryPredictor()
    predictor.train(training_data)

    # Crear un jugador de prueba
    test_player = PlayerMatchData(
        kills=8,
        deaths=2,
        assists=3,
        damage_dealt=2400,
        damage_received=800,
        healing_done=120,
        objective_time=90
    )

    # Predecir si ganará
    prediction = predictor.predict(test_player)
    print(f"¿El jugador ganará? {'Sí' if prediction == 1 else 'No'}")