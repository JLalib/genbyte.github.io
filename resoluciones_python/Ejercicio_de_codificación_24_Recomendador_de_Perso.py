from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances

class Player:
    def __init__(self, name, level, aggressiveness, cooperation, exploration, preferred_class=None):
        self.name = name
        self.level = level
        self.aggressiveness = aggressiveness
        self.cooperation = cooperation
        self.exploration = exploration
        self.preferred_class = preferred_class  # Solo usado en entrenamiento

    def to_features(self):
        return [self.level, self.aggressiveness, self.cooperation, self.exploration]


class PlayerDataset:
    def __init__(self, players):
        self.players = players

    def get_X(self):
        return [p.to_features() for p in self.players]

    def get_y(self):
        return [p.preferred_class for p in self.players]


class ClassRecommender:
    def __init__(self, n_neighbors=3):
        self.k = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.X = []
        self.players = []

    def train(self, dataset: PlayerDataset):
        self.X = dataset.get_X()
        self.y = dataset.get_y()
        self.players = dataset.players
        self.model.fit(self.X, self.y)

    def predict(self, player: Player):
        return self.model.predict([player.to_features()])[0]

    def get_nearest_neighbors(self, player: Player):
        # Distancias a todos los jugadores en entrenamiento
        distances = euclidean_distances([player.to_features()], self.X)[0]
        # Índices de los k más cercanos
        indices = distances.argsort()[:self.k]
        return indices.tolist()