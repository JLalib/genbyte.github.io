import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1️⃣ Registro meteorológico
class WeatherRecord:
    def __init__(self, humidity, pressure, rain):
        self.humidity = humidity
        self.pressure = pressure
        self.rain = rain

# 2️⃣ Generador de datos sintéticos
class WeatherDataGenerator:
    def __init__(self, n=60):
        self.n = n

    def generate(self):
        np.random.seed(42)
        records = []
        for _ in range(self.n):
            # Para simular lluvia: humedad alta y presión baja
            if np.random.rand() > 0.4:
                humidity = np.random.randint(70, 101)  # 70-100%
                pressure = np.random.randint(980, 1010)  # baja presión
                rain = 1
            else:
                humidity = np.random.randint(20, 60)  # baja humedad
                pressure = np.random.randint(1010, 1030)  # alta presión
                rain = 0
            records.append(WeatherRecord(humidity, pressure, rain))
        return records

# 3️⃣ Clasificador de lluvia con regresión logística
class WeatherRainClassifier:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, records):
        X = np.array([[r.humidity, r.pressure] for r in records])
        y = np.array([r.rain for r in records])
        self.model.fit(X, y)

    def predict(self, humidity, pressure):
        pred = self.model.predict([[humidity, pressure]])
        return pred[0]

    def get_model(self):
        return self.model

# 4️⃣ Ejemplo que orquesta todo y visualiza
class WeatherRainPredictionExample:
    def run(self):
        # Generar datos
        generator = WeatherDataGenerator()
        records = generator.generate()

        # Entrenar modelo
        classifier = WeatherRainClassifier()
        classifier.fit(records)

        # Datos para evaluación
        X = np.array([[r.humidity, r.pressure] for r in records])
        y = np.array([r.rain for r in records])
        y_pred = classifier.get_model().predict(X)

        # Métricas
        print(confusion_matrix(y, y_pred))
        print(classification_report(y, y_pred, zero_division=0))

        # Predicción para nuevas condiciones
        new_humidity = 80
        new_pressure = 995
        prediction = classifier.predict(new_humidity, new_pressure)
        print("🔍 Predicción para condiciones nuevas:")
        print(f"   Humedad: {new_humidity}%")
        print(f"   Presión: {new_pressure} hPa")
        print(f"   ¿Lloverá?: {'Sí ☔' if prediction == 1 else 'No'}")

        # Visualización
        plt.figure(figsize=(8,6))
        for r in records:
            color = 'blue' if r.rain == 0 else 'green'
            marker = 'o' if r.rain == 0 else 'x'
            plt.scatter(r.humidity, r.pressure, c=color, marker=marker, label='Lluvia' if r.rain == 1 else 'No lluvia', alpha=0.7)
        # Evitar duplicar leyendas
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.xlabel("Humedad (%)")
        plt.ylabel("Presión atmosférica (hPa)")
        plt.title("🌦️ Predicción de lluvia según humedad y presión")

        # Marcar punto nuevo
        plt.scatter(new_humidity, new_pressure, c='red', marker='*', s=200, label='Nuevo dato')
        plt.legend()
        plt.grid(True)
        plt.show()