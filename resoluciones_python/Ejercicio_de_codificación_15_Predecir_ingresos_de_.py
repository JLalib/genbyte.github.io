# Importar librerías necesarias
from sklearn.linear_model import LinearRegression

# Clase App
class App:
    def __init__(self, name, downloads, rating, size_mb, reviews, revenue=None):
        self.name = name
        self.downloads = downloads          # en miles
        self.rating = rating                # entre 1 y 5
        self.size_mb = size_mb              # en MB
        self.reviews = reviews              # número de reseñas
        self.revenue = revenue              # en miles de dólares (puede ser None para predicción)

    def to_features(self):
        return [self.downloads, self.rating, self.size_mb, self.reviews]

# Clase RevenuePredictor
class RevenuePredictor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, apps):
        X = [app.to_features() for app in apps]
        y = [app.revenue for app in apps if app.revenue is not None]
        self.model.fit(X, y)

    def predict(self, app):
        features = [app.to_features()]
        return self.model.predict(features)[0]