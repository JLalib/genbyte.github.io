import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class SleepProfile:
    def __init__(self, duration, latency, wakeups, variability):
        self.duration = duration
        self.latency = latency
        self.wakeups = wakeups
        self.variability = variability

    def to_vector(self):
        return [self.duration, self.latency, self.wakeups, self.variability]

class SleepDatasetGenerator:
    def __init__(self, n=300):
        self.n = n

    def generate(self):
        np.random.seed(0)
        durations = np.random.normal(7, 1.2, self.n)
        latencies = np.abs(np.random.normal(20, 10, self.n))
        wakeups = np.random.poisson(1.5, self.n)
        variabilities = np.abs(np.random.normal(30, 15, self.n))

        profiles = []
        for d, l, w, v in zip(durations, latencies, wakeups, variabilities):
            profiles.append(SleepProfile(d, l, w, v))
        return profiles

class SleepClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.scaler = StandardScaler()

    def fit(self, profiles):
        X = np.array([p.to_vector() for p in profiles])
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        self.labels = self.kmeans.labels_
        return X_scaled, self.labels

    def get_cluster_centers(self):
        return self.scaler.inverse_transform(self.kmeans.cluster_centers_)

class SleepAnalysisExample:
    def run(self):
        generator = SleepDatasetGenerator()
        profiles = generator.generate()

        clusterer = SleepClusterer(n_clusters=3)
        X_scaled, labels = clusterer.fit(profiles)

        df = pd.DataFrame([p.to_vector() for p in profiles],
                          columns=["Duración", "Latencia", "Despertares", "Variabilidad"])
        df["Grupo"] = labels
        df["Grupo"] = df["Grupo"].astype(int)  # <-- Aquí está la solución

        print("📌 Centroides de los grupos:")
        centers = clusterer.get_cluster_centers()
        for i, c in enumerate(centers):
            print(f"Grupo {i}: Duración={c[0]:.2f}h, Latencia={c[1]:.1f}min, Despertares={c[2]:.1f}, Variabilidad={c[3]:.1f}min")

        colores = ['blue', 'green', 'orange']
        plt.figure(figsize=(8, 6))
        for i in range(clusterer.n_clusters):
            subset = df[df["Grupo"] == i]
            plt.scatter(subset["Duración"], subset["Variabilidad"],
                        c=colores[i], label=f"Grupo {i}", alpha=0.6)

        plt.xlabel("Duración del sueño (horas)")
        plt.ylabel("Variabilidad en horario de dormir (minutos)")
        plt.title("💤 Agrupación de perfiles de sueño (K-Means)")
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    example = SleepAnalysisExample()
    example.run()