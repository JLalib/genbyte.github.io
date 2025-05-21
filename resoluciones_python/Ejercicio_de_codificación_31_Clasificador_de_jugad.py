import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


class BasketballPlayer:
    def __init__(self, height, weight, avg_points, performance):
        self.height = height
        self.weight = weight
        self.avg_points = avg_points
        self.performance = performance  # "Bajo", "Medio", "Alto"

    def to_vector(self):
        return [self.height, self.weight, self.avg_points]


class BasketballDataGenerator:
    def __init__(self, num_samples=300):
        self.num_samples = num_samples

    def generate(self):
        players = []
        samples_per_class = self.num_samples // 3

        # Bajo rendimiento
        for _ in range(samples_per_class):
            height = np.random.normal(175, 5)
            weight = np.random.normal(70, 5)
            avg_points = np.random.uniform(2, 7.9)
            players.append(BasketballPlayer(height, weight, avg_points, "Bajo"))

        # Medio rendimiento
        for _ in range(samples_per_class):
            height = np.random.normal(185, 5)
            weight = np.random.normal(80, 5)
            avg_points = np.random.uniform(8, 15)
            players.append(BasketballPlayer(height, weight, avg_points, "Medio"))

        # Alto rendimiento
        for _ in range(samples_per_class):
            height = np.random.normal(195, 5)
            weight = np.random.normal(90, 5)
            avg_points = np.random.uniform(15.1, 25)
            players.append(BasketballPlayer(height, weight, avg_points, "Alto"))

        return players


class BasketballPerformanceClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def fit(self, players):
        X = [p.to_vector() for p in players]
        y = [p.performance for p in players]
        self.model.fit(X, y)

    def predict(self, height, weight, avg_points):
        return self.model.predict([[height, weight, avg_points]])[0]

    def evaluate(self, players):
        X = [p.to_vector() for p in players]
        y_true = [p.performance for p in players]
        y_pred = self.model.predict(X)

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))


class BasketballPredictionExample:
    def run(self):
        # Generar datos
        generator = BasketballDataGenerator()
        players = generator.generate()

        # Dividir en entrenamiento y prueba
        train_players, test_players = train_test_split(players, test_size=0.3, random_state=42)

        # Entrenar el clasificador
        classifier = BasketballPerformanceClassifier()
        classifier.fit(train_players)
        classifier.evaluate(test_players)

        # Predicción personalizada
        print("\n🎯 Predicción personalizada:")
        height = 198
        weight = 92
        avg_points = 17
        predicted = classifier.predict(height, weight, avg_points)
        print(f"Altura: {height} cm, Peso: {weight} kg, Prom. puntos: {avg_points} → Categoría predicha: {predicted}")

        # Visualización
        df = pd.DataFrame({
            "Altura": [p.height for p in test_players],
            "Peso": [p.weight for p in test_players],
            "Puntos": [p.avg_points for p in test_players],
            "Rendimiento": [p.performance for p in test_players]
        })

        colores = {"Bajo": "red", "Medio": "orange", "Alto": "green"}

        plt.figure(figsize=(8, 6))
        for categoria in df["Rendimiento"].unique():
            subset = df[df["Rendimiento"] == categoria]
            plt.scatter(subset["Altura"], subset["Puntos"], color=colores[categoria], label=categoria, alpha=0.7)

        # Nuevo jugador
        plt.scatter(height, avg_points, color="black", label="Nuevo jugador", marker="X", s=100)

        plt.xlabel("Altura (cm)")
        plt.ylabel("Prom. puntos por partido")
        plt.title("Clasificación de rendimiento de jugadores de baloncesto")
        plt.legend()
        plt.grid(True)
        plt.show()