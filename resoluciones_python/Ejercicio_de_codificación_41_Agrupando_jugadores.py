from sklearn.cluster import KMeans
from typing import List
import numpy as np
from collections import defaultdict

# ----------------------------
# Clase Player
# ----------------------------
class Player:
    def __init__(self, name: str, avg_session_time: float, missions_completed: int,
                 accuracy: float, aggressiveness: float):
        self.name = name
        self.avg_session_time = avg_session_time
        self.missions_completed = missions_completed
        self.accuracy = accuracy
        self.aggressiveness = aggressiveness

    def to_features(self):
        return [self.avg_session_time, self.missions_completed, self.accuracy, self.aggressiveness]


# ----------------------------
# Clase PlayerClusterer
# ----------------------------
class PlayerClusterer:
    def __init__(self):
        self.model = None
        self.players = []
        self.labels = []

    def fit(self, players: List[Player], n_clusters: int):
        self.players = players
        X = np.array([p.to_features() for p in players])
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.model.fit(X)
        self.labels = self.model.labels_

    def predict(self, player: Player) -> int:
        features = np.array(player.to_features()).reshape(1, -1)
        return int(self.model.predict(features)[0])  # ✅ Cast a int para evitar errores de tipo

    def get_cluster_centers(self):
        return self.model.cluster_centers_

    def print_cluster_summary(self, players: List[Player]):
        cluster_map = defaultdict(list)
        for player, label in zip(players, self.labels):
            cluster_map[label].append(player.name)

        for cluster_id in sorted(cluster_map.keys()):
            print(f"Cluster {cluster_id}:")
            for name in cluster_map[cluster_id]:
                print(f"  - {name}")


# ----------------------------
# Clase GameAnalytics
# ----------------------------
class GameAnalytics:
    def __init__(self):
        self.data = [
            ("Alice", 2.5, 100, 0.85, 0.3),
            ("Bob", 1.0, 20, 0.60, 0.7),
            ("Charlie", 3.0, 150, 0.9, 0.2),
            ("Diana", 0.8, 15, 0.55, 0.9),
            ("Eve", 2.7, 120, 0.88, 0.25),
            ("Frank", 1.1, 30, 0.62, 0.65),
            ("Grace", 0.9, 18, 0.58, 0.85),
            ("Hank", 3.2, 160, 0.91, 0.15)
        ]
        self.clusterer = PlayerClusterer()

    def run(self):
        # Crear objetos Player
        players = [Player(*item) for item in self.data]

        # Entrenar el modelo
        self.clusterer.fit(players, n_clusters=3)

        # Imprimir resumen de clusters
        self.clusterer.print_cluster_summary(players)

        # Predecir nuevo jugador
        zoe = Player("Zoe", 1.5, 45, 0.65, 0.5)
        cluster_zoe = self.clusterer.predict(zoe)

        print(f"\nJugador Zoe pertenece al cluster: {cluster_zoe}")