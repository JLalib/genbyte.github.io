from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Clase App
class App:
    def __init__(self, app_name, monthly_users, avg_session_length, retention_rate, social_shares, success=None):
        self.app_name = app_name
        self.monthly_users = monthly_users
        self.avg_session_length = avg_session_length
        self.retention_rate = retention_rate
        self.social_shares = social_shares
        self.success = success  # 1 = éxito, 0 = fracaso

    def to_features(self):
        return [self.monthly_users, self.avg_session_length, self.retention_rate, self.social_shares]

# Clase AppDataset
class AppDataset:
    def __init__(self, apps):
        self.apps = apps

    def get_feature_matrix(self):
        return [app.to_features() for app in self.apps]

    def get_target_vector(self):
        return [app.success for app in self.apps if app.success is not None]

# Clase SuccessPredictor
class SuccessPredictor:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def train(self, dataset: AppDataset):
        X = dataset.get_feature_matrix()
        y = dataset.get_target_vector()
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, app: App):
        X = [app.to_features()]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]

    def predict_proba(self, app: App):
        X = [app.to_features()]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[0][1]  # Probabilidad de éxito (clase 1)