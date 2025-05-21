import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 1. Clase Snack
class Snack:
    def __init__(self, calories, sugar, protein, fat, fiber, is_healthy=None):
        self.calories = calories
        self.sugar = sugar
        self.protein = protein
        self.fat = fat
        self.fiber = fiber
        self.is_healthy = is_healthy  # Etiqueta opcional para clasificación

    def to_vector(self):
        # Convierte el snack a un vector de características que se puede usar en el modelo
        return [self.calories, self.sugar, self.protein, self.fat, self.fiber]


# 2. Clase SnackGenerator
class SnackGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        snacks = []
        for _ in range(self.num_samples):
            calories = np.random.randint(50, 501)  # Calorías entre 50 y 500
            sugar = np.round(np.random.uniform(0, 50), 2)  # Azúcar entre 0 y 50
            protein = np.round(np.random.uniform(0, 30), 2)  # Proteína entre 0 y 30
            fat = np.round(np.random.uniform(0, 30), 2)  # Grasa entre 0 y 30
            fiber = np.round(np.random.uniform(0, 15), 2)  # Fibra entre 0 y 15

            # Regla para clasificar como saludable
            is_healthy = int(
                calories < 200 and sugar < 15 and fat < 10 and (protein >= 5 or fiber >= 5)
            )

            snacks.append(Snack(calories, sugar, protein, fat, fiber, is_healthy))
        return snacks


# 3. Clase SnackClassifier
class SnackClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier()  # Árbol de decisión

    def fit(self, snacks):
        # Convierte snacks en vectores y etiquetas
        X = [snack.to_vector() for snack in snacks]
        y = [snack.is_healthy for snack in snacks]
        self.model.fit(X, y)  # Entrena el modelo

    def predict(self, snack):
        # Predice si el snack es saludable o no
        return self.model.predict([snack.to_vector()])[0]


# 4. Ejemplo de uso
class SnackRecommendationExample:
    def run(self):
        # Generar datos sintéticos
        generator = SnackGenerator(100)
        training_data = generator.generate()

        # Entrenar el clasificador
        classifier = SnackClassifier()
        classifier.fit(training_data)

        # Snack de prueba
        new_snack = Snack(calories=150, sugar=10, protein=6, fat=5, fiber=3)

        # Realizar predicción
        prediction = classifier.predict(new_snack)

        # Mostrar resultados
        print("🔍 Snack Info:")
        print(f"Calories: {new_snack.calories}, Sugar: {new_snack.sugar}g, "
              f"Protein: {new_snack.protein}g, Fat: {new_snack.fat}g, Fiber: {new_snack.fiber}g")
        print("✅ Predicción: Este snack", "es saludable." if prediction == 1 else "no es saludable.")