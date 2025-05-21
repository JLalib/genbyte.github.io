import numpy as np
from sklearn.svm import SVC


# Clase que representa una muestra de aire con características numéricas
class AirSample:
    def __init__(self, pm25, pm10, o3, no2, quality=None):
        self.pm25 = pm25
        self.pm10 = pm10
        self.o3 = o3
        self.no2 = no2
        self.quality = quality  # 0 = saludable, 1 = contaminado

    def to_vector(self):
        return [self.pm25, self.pm10, self.o3, self.no2]

    def __repr__(self):
        return f"AirSample(pm25={self.pm25}, pm10={self.pm10}, o3={self.o3}, no2={self.no2})"


# Clase para generar datos sintéticos de muestras de aire
class AirDataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        samples = []
        for _ in range(self.num_samples):
            pm25 = np.round(np.random.uniform(10, 50), 2)
            pm10 = np.round(np.random.uniform(20, 70), 2)
            o3 = np.round(np.random.uniform(30, 80), 2)
            no2 = np.round(np.random.uniform(10, 60), 2)

            # Regla orientativa para determinar si el área está contaminada
            quality = int(pm25 > 35 or pm10 > 50 or no2 > 40)

            samples.append(AirSample(pm25, pm10, o3, no2, quality))
        return samples


# Clase para entrenar un clasificador usando SVM
class AirQualityClassifier:
    def __init__(self):
        self.model = SVC(kernel='linear', random_state=42)

    def fit(self, samples):
        X = [sample.to_vector() for sample in samples]
        y = [sample.quality for sample in samples]
        self.model.fit(X, y)

    def predict(self, sample):
        return self.model.predict([sample.to_vector()])[0]


# Clase de ejemplo para ejecutar el sistema de clasificación
class AirQualityExample:
    def run(self):
        # Generar 200 muestras aleatorias
        generator = AirDataGenerator(200)
        samples = generator.generate()

        # Entrenar el modelo
        classifier = AirQualityClassifier()
        classifier.fit(samples)

        # Crear una nueva muestra de aire para predecir
        new_sample = AirSample(pm25=22, pm10=30, o3=50, no2=35)

        # Predecir la calidad del aire
        prediction = classifier.predict(new_sample)

        # Mostrar los resultados
        print("🌍 Muestra de aire:")
        print(f"PM2.5: {new_sample.pm25}, PM10: {new_sample.pm10}, O3: {new_sample.o3}, NO2: {new_sample.no2}")
        if prediction == 0:
            print("✅ Predicción de calidad: Saludable ✅")
        else:
            print("❌ Predicción de calidad: Contaminado ❌")


# Ejecutar el ejemplo
if __name__ == "__main__":
    example = AirQualityExample()
    example.run()