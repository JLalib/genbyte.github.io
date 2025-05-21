import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Clase Player
# -----------------------------
class Player:
    def __init__(self, player_name, character_type, avg_session_time, matches_played,
                 aggressive_actions, defensive_actions, items_bought, victories, style=None):
        self.player_name = player_name
        self.character_type = character_type
        self.avg_session_time = avg_session_time
        self.matches_played = matches_played
        self.aggressive_actions = aggressive_actions
        self.defensive_actions = defensive_actions
        self.items_bought = items_bought
        self.victories = victories
        self.style = style  # "aggressive" o "strategic"

    def to_features(self):
        return [
            self.avg_session_time,
            self.matches_played,
            self.aggressive_actions,
            self.defensive_actions,
            self.items_bought
        ]

# -----------------------------
# Clase GameModel
# -----------------------------
class GameModel:
    def __init__(self, players):
        self.players = players
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression()
        self.regressor = LinearRegression()
        self.cluster_model = None
        self.cluster_df = None

    def _prepare_dataframe(self):
        data = []
        for p in self.players:
            row = p.to_features()
            row.append(p.victories)
            row.append(p.style)
            row.append(p.player_name)
            row.append(p.character_type)
            data.append(row)

        columns = [
            'avg_session_time', 'matches_played',
            'aggressive_actions', 'defensive_actions', 'items_bought',
            'victories', 'style', 'player_name', 'character_type'
        ]
        return pd.DataFrame(data, columns=columns)

    def train_classification_model(self):
        df = self._prepare_dataframe()
        X = df[['avg_session_time', 'matches_played', 'aggressive_actions',
                'defensive_actions', 'items_bought']]
        y = df['style']
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)

    def train_regression_model(self):
        df = self._prepare_dataframe()
        X = df[['avg_session_time', 'matches_played', 'aggressive_actions',
                'defensive_actions', 'items_bought']]
        y = df['victories']
        X_scaled = self.scaler.transform(X)  # usar el mismo scaler
        self.regressor.fit(X_scaled, y)

    def train_clustering_model(self, n_clusters=2):
        df = self._prepare_dataframe()
        X = df[['avg_session_time', 'matches_played', 'aggressive_actions',
                'defensive_actions', 'items_bought']]
        X_scaled = self.scaler.transform(X)
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.cluster_model.fit_predict(X_scaled)
        df['cluster'] = cluster_labels
        self.cluster_df = df

    def predict_style(self, player):
        features = np.array(player.to_features()).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict(features_scaled)[0]

    def predict_victories(self, player):
        features = np.array(player.to_features()).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.regressor.predict(features_scaled)[0]

    def assign_cluster(self, player):
        features = np.array(player.to_features()).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return int(self.cluster_model.predict(features_scaled)[0])

    def mostrar_jugadores_por_cluster(self):
        if self.cluster_df is None:
            print("Primero debes entrenar el modelo de clustering.")
            return

        for cluster_id in sorted(self.cluster_df['cluster'].unique()):
            print(f"\nCluster {cluster_id}:")
            jugadores = self.cluster_df[self.cluster_df['cluster'] == cluster_id]
            for _, row in jugadores.iterrows():
                estilo = row['style'].capitalize() if pd.notna(row['style']) else "Desconocido"
                print(f"{row['player_name']} - {row['character_type'].capitalize()} - {estilo}")