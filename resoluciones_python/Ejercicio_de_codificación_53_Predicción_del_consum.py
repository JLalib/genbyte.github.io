import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class EnergyRecord:
    def __init__(self, temperature, consumption):
        self.temperature = temperature  # °C
        self.consumption = consumption  # kWh

    def to_vector(self):
        return [self.temperature]


class EnergyDataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        temperatures = np.random.uniform(-5, 35, self.num_samples)  # Temperaturas entre -5 y 35 °C
        consumption = 100 + (np.abs(temperatures - 20) * 3) + np.random.normal(0, 5, self.num_samples)
        data = [EnergyRecord(t, c) for t, c in zip(temperatures, consumption)]
        return data


class EnergyRegressor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, records):
        X = np.array([r.to_vector() for r in records])  # Temperatura como X
        y = np.array([r.consumption for r in records])  # Consumo como y
        self.model.fit(X, y)

    def predict(self, temperature):
        return self.model.predict([[temperature]])[0]  # Devuelve un valor (no una lista)

    def get_model(self):
        return self.model


class EnergyPredictionExample:
    def run(self):
        # 1. Generar datos
        generator = EnergyDataGenerator(100)
        data = generator.generate()

        # 2. Entrenar el modelo
        regressor = EnergyRegressor()
        regressor.fit(data)

        # 3. Predecir consumo para una nueva temperatura
        test_temperature = 30
        prediction = regressor.predict(test_temperature)

        print(f"🔍 Temperatura: {test_temperature} °C")
        print(f"⚡ Predicción de consumo: {prediction:.2f} kWh")

        # 4. Visualización
        temperatures = [r.temperature for r in data]
        consumptions = [r.consumption for r in data]

        plt.figure(figsize=(10, 6))
        plt.scatter(temperatures, consumptions, color='blue', label='Datos observados', alpha=0.6)

        # Generar puntos para la línea de regresión
        line_x = np.linspace(-5, 35, 100).reshape(-1, 1)
        line_y = regressor.get_model().predict(line_x)

        plt.plot(line_x, line_y, color='red', linewidth=2, label='Regresión lineal')
        plt.xlabel('Temperatura (°C)', fontsize=12)
        plt.ylabel('Consumo energético (kWh)', fontsize=12)
        plt.title('Consumo energético en función de la temperatura', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    example = EnergyPredictionExample()
    example.run()