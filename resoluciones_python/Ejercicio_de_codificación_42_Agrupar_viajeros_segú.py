import numpy as np
from sklearn.cluster import KMeans

# 1. Clase Traveler
class Traveler:
    def __init__(self, beach, mountain, city, countryside):
        self.beach = beach
        self.mountain = mountain
        self.city = city
        self.countryside = countryside

    def to_vector(self):
        return [self.beach, self.mountain, self.city, self.countryside]

# 2. Clase TravelerGenerator
class TravelerGenerator:
    def __init__(self, num_travelers=100):
        self.num_travelers = num_travelers

    def generate(self):
        travelers = []
        for _ in range(self.num_travelers):
            beach = np.round(np.random.uniform(0, 10), 2)
            mountain = np.round(np.random.uniform(0, 10), 2)
            city = np.round(np.random.uniform(0, 10), 2)
            countryside = np.round(np.random.uniform(0, 10), 2)

            traveler = Traveler(beach, mountain, city, countryside)
            travelers.append(traveler)
        return travelers

# 3. Clase TravelerClusterer
class TravelerClusterer:
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_centers = None

    def fit(self, travelers):
        X = [t.to_vector() for t in travelers]
        self.model.fit(X)
        self.cluster_centers = self.model.cluster_centers_

    def predict(self, traveler):
        return self.model.predict([traveler.to_vector()])[0]

    def get_cluster_centers(self):
        return self.cluster_centers

# 4. Clase TravelerClusteringExample
class TravelerClusteringExample:
    def run(self):
        # 1. Generar datos
        generator = TravelerGenerator(100)
        travelers = generator.generate()

        # 2. Entrenar el modelo
        clusterer = TravelerClusterer(n_clusters=3)
        clusterer.fit(travelers)

        # 3. Obtener y mostrar los centros de los clústeres
        centers = clusterer.get_cluster_centers()
        print("🏝️🏔️🏙️🌄 Centros de los Clústeres (Preferencias promedio):")
        for i, center in enumerate(centers):
            print(f"Cluster {i}: Playa={center[0]:.2f}, Montaña={center[1]:.2f}, Ciudad={center[2]:.2f}, Campo={center[3]:.2f}")

        # 4. Crear un viajero nuevo
        new_traveler = Traveler(beach=8.5, mountain=2.0, city=9.0, countryside=1.5)

        # 5. Predecir su grupo
        group = clusterer.predict(new_traveler)

        # 6. Mostrar resultados
        print("🧳 Nuevo viajero:")
        print(f"Beach: {new_traveler.beach}, Mountain: {new_traveler.mountain}, City: {new_traveler.city}, Countryside: {new_traveler.countryside}")
        print(f"📍 Pertenece al grupo: {group}")

# 5. Ejecutar el ejemplo
if __name__ == "__main__":
    example = TravelerClusteringExample()
    example.run()