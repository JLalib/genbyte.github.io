import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

class Individual:
    def __init__(self, heart_rate, cortisol_level, skin_conductance, stress_level):
        self.heart_rate = heart_rate
        self.cortisol_level = cortisol_level
        self.skin_conductance = skin_conductance
        self.stress_level = stress_level

    def to_vector(self):
        return [self.heart_rate, self.cortisol_level, self.skin_conductance]

class StressDataGenerator:
    def __init__(self, n=300):  # Acepta n como número de muestras
        self.n = n

    def generate(self):
        individuals = []
        heart_rates = np.random.normal(75, 15, self.n)
        cortisol_levels = np.random.normal(12, 4, self.n)
        skin_conductances = np.random.normal(5, 1.5, self.n)

        for hr, cort, cond in zip(heart_rates, cortisol_levels, skin_conductances):
            if hr > 90 or cort > 18 or cond > 6.5:
                stress = "Alto"
            elif hr > 70 or cort > 10 or cond > 4.5:
                stress = "Moderado"
            else:
                stress = "Bajo"
            individuals.append(Individual(hr, cort, cond, stress))
        return individuals

class StressClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def fit(self, individuals):
        X = np.array([ind.to_vector() for ind in individuals])
        y = np.array([ind.stress_level for ind in individuals])
        self.model.fit(X, y)

    def predict(self, heart_rate, cortisol, conductance):
        return self.model.predict([[heart_rate, cortisol, conductance]])[0]

    def evaluate(self, test_data):
        X_test = np.array([ind.to_vector() for ind in test_data])
        y_true = np.array([ind.stress_level for ind in test_data])
        y_pred = self.model.predict(X_test)
        print("📊 Matriz de confusión:")
        print(confusion_matrix(y_true, y_pred))
        print("\n📝 Informe de clasificación:")
        print(classification_report(y_true, y_pred))

class StressAnalysisExample:
    def run(self):
        generator = StressDataGenerator(n=300)
        data = generator.generate()

        train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

        classifier = StressClassifier()
        classifier.fit(train_data)

        classifier.evaluate(test_data)

        print("🧠 Predicción para individuo personalizado:")
        hr_test, cort_test, cond_test = 95, 20, 7
        pred_stress = classifier.predict(hr_test, cort_test, cond_test)
        print(f"  Ritmo cardíaco: {hr_test}, Cortisol: {cort_test}, Conductancia: {cond_test}")
        print(f"  → Nivel estimado de estrés: {pred_stress}")

        df = pd.DataFrame({
            "Cortisol": [ind.cortisol_level for ind in data],
            "HeartRate": [ind.heart_rate for ind in data],
            "StressLevel": [ind.stress_level for ind in data]
        })

        colors = {"Bajo": "green", "Moderado": "orange", "Alto": "red"}
        plt.figure(figsize=(10, 6))
        for level in df["StressLevel"].unique():
            subset = df[df["StressLevel"] == level]
            plt.scatter(subset["Cortisol"], subset["HeartRate"], c=colors[level], label=level, alpha=0.7)

        plt.title("Nivel de estrés según Cortisol y Ritmo Cardíaco")
        plt.xlabel("Nivel de Cortisol (µg/dL)")
        plt.ylabel("Ritmo Cardíaco (pulsaciones por minuto)")
        plt.legend(title="Nivel de Estrés")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    example = StressAnalysisExample()
    example.run()