import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Clase que representa un registro de vehículo
class VehicleRecord:
    def __init__(self, hours: float, wear: float):
        self.hours = hours
        self.wear = wear

# Clase para generar datos sintéticos de entrenamiento
class VehicleDataGenerator:
    def __init__(self, n_samples=100, noise=5, seed=42):
        np.random.seed(seed)
        self.n_samples = n_samples
        self.noise = noise

    def generate(self):
        hours = np.random.uniform(50, 500, self.n_samples)
        wear = 0.2 * hours + np.random.normal(0, self.noise, self.n_samples)
        return [VehicleRecord(h, w) for h, w in zip(hours, wear)]

# Clase para entrenar y predecir con regresión lineal
class VehicleWearRegressor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, records):
        X = np.array([[r.hours] for r in records])
        y = np.array([r.wear for r in records])
        self.model.fit(X, y)

    def predict(self, hours):
        return float(self.model.predict(np.array([[hours]]))[0])

    def get_model(self):
        return self.model

# Clase principal de ejecución y visualización
class VehicleWearPredictionExample:
    def __init__(self):
        self.generator = VehicleDataGenerator()
        self.regressor = VehicleWearRegressor()

    def run(self, new_hours=250):
        # Generar datos
        records = self.generator.generate()
        # Entrenar el modelo
        self.regressor.fit(records)
        # Predecir nuevo valor
        predicted_wear = self.regressor.predict(new_hours)
        # Visualizar resultados
        self._plot(records, new_hours, predicted_wear)
        # Imprimir resultado
        print(f"\n⏱ Horas de uso estimadas: {new_hours}")
        print(f"⚙️ Nivel de desgaste estimado: {predicted_wear:.2f}%")

    def _plot(self, records, new_hours, predicted_wear):
        hours = np.array([r.hours for r in records])
        wear = np.array([r.wear for r in records])
        model = self.regressor.get_model()

        # Rango para la línea de regresión
        x_range = np.linspace(hours.min(), hours.max(), 100).reshape(-1, 1)
        y_range = model.predict(x_range)

        plt.figure(figsize=(10, 6))
        plt.scatter(hours, wear, alpha=0.6, label="Datos reales")
        plt.plot(x_range, y_range, color='red', label="Regresión lineal")
        plt.scatter([new_hours], [predicted_wear], color='green', s=100, label="Predicción nueva")
        plt.xlabel("Horas de uso")
        plt.ylabel("Nivel de desgaste (%)")
        plt.title("Predicción del desgaste de vehículos militares")
        plt.legend()
        plt.grid(True)
        plt.show()