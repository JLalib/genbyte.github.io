import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1️⃣ Representación de satélites
class Satellite:
    def __init__(self, duracion_mision, paneles_sol, carga_util, consumo_diario):
        self.duracion_mision = duracion_mision
        self.paneles_sol = paneles_sol
        self.carga_util = carga_util
        self.consumo_diario = consumo_diario

    def to_dict(self):
        return {
            "duracion_mision_dias": self.duracion_mision,
            "paneles_sol": self.paneles_sol,
            "carga_util": self.carga_util,
            "consumo_diario": self.consumo_diario,
        }

# 2️⃣ Generador de datos sintéticos
class SatelliteDatasetGenerator:
    def __init__(self, n=300):
        self.n = n

    def generate(self):
        np.random.seed(42)
        duraciones = np.random.randint(100, 1000, self.n)
        paneles = np.random.uniform(10, 100, self.n)
        cargas = np.random.uniform(200, 2000, self.n)
        consumo = 5 + 0.01 * duraciones + 0.002 * cargas + np.random.normal(0, 1, self.n)

        satellites = []
        for d, p, c, e in zip(duraciones, paneles, cargas, consumo):
            satellites.append(Satellite(d, p, c, e))
        return satellites

# 3️⃣ Procesador de datos
class SatelliteDataProcessor:
    def __init__(self, satellites):
        self.df = pd.DataFrame([sat.to_dict() for sat in satellites])
        self.df["eficiencia_energia"] = self.df["consumo_diario"] / self.df["paneles_sol"]

    def get_dataframe(self):
        return self.df

# 4️⃣ Regresor de consumo energético
class EnergyConsumptionRegressor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def get_coefficients(self):
        return self.model.coef_, self.model.intercept_

# 5️⃣ Visualizador
class SatellitePlotter:
    def __init__(self, df, y_pred):
        self.df = df
        self.y_pred = y_pred

    def plot(self):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            self.df["duracion_mision_dias"],
            self.df["consumo_diario"],
            c=self.df["carga_util"],
            cmap="viridis",
            alpha=0.7,
        )
        plt.plot(self.df["duracion_mision_dias"], self.y_pred, color="red", label="Regresión lineal")
        plt.xlabel("Duración de la misión (días)")
        plt.ylabel("Consumo diario (kWh)")
        plt.title("🛰️ Consumo energético vs. duración de misión")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Carga útil (kg)")
        plt.grid(True)
        plt.legend()
        plt.show()

# 6️⃣ Clase orquestadora que integra todo
class SatelliteAnalysisExample:
    def run(self):
        # 1. Generar datos
        generator = SatelliteDatasetGenerator()
        satellites = generator.generate()

        # 2. Procesar datos
        processor = SatelliteDataProcessor(satellites)
        df = processor.get_dataframe()

        # 3. Regresión lineal
        X = df[["duracion_mision_dias"]]
        y = df["consumo_diario"]
        regressor = EnergyConsumptionRegressor()
        y_pred = regressor.fit(X, y)

        # 4. Evaluar modelo
        r2 = regressor.evaluate(y, y_pred)
        coef, intercept = regressor.get_coefficients()
        print(f"📈 Modelo: y = {coef[0]:.4f} * x + {intercept:.2f}")
        print(f"🔍 R² del modelo: {r2:.4f}")

        # 5. Visualización
        plotter = SatellitePlotter(df, y_pred)
        plotter.plot()