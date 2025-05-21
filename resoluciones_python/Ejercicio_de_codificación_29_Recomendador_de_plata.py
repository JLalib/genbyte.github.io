from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

class Project:
    def __init__(self, name, team_size, budget, duration_months,
                 realtime_required, needs_offline, target_users,
                 recommended_platform=None):
        self.name = name
        self.team_size = team_size
        self.budget = budget
        self.duration_months = duration_months
        self.realtime_required = int(realtime_required)
        self.needs_offline = int(needs_offline)
        self.target_users = target_users
        self.recommended_platform = recommended_platform

    def to_features(self):
        return [
            self.team_size,
            self.budget,
            self.duration_months,
            self.realtime_required,
            self.needs_offline,
            self.target_users  # será codificado por el dataset
        ]


class ProjectDataset:
    def __init__(self, projects):
        self.projects = projects
        self.encoder_target_users = LabelEncoder()
        self.encoder_platform = LabelEncoder()

        # Codificar target_users
        target_user_values = [p.target_users for p in projects]
        self.encoded_target_users = self.encoder_target_users.fit_transform(target_user_values)

        # Codificar plataformas solo si están disponibles (entrenamiento)
        platform_values = [p.recommended_platform for p in projects if p.recommended_platform is not None]
        if platform_values:
            self.encoder_platform.fit(platform_values)

    def get_X(self):
        X = []
        for i, p in enumerate(self.projects):
            features = p.to_features()
            features[5] = self.encoded_target_users[i]  # codificar target_users
            X.append(features)
        return X

    def get_y(self):
        y_raw = [p.recommended_platform for p in self.projects if p.recommended_platform is not None]
        return self.encoder_platform.transform(y_raw)

    def encode_target_users(self, user_label):
        return self.encoder_target_users.transform([user_label])[0]

    def decode_platform(self, encoded_label):
        return self.encoder_platform.inverse_transform([encoded_label])[0]


class PlatformRecommender:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)
        self.dataset = None

    def train(self, dataset: ProjectDataset):
        self.dataset = dataset
        X = dataset.get_X()
        y = dataset.get_y()
        self.model.fit(X, y)

    def predict(self, project: Project):
        if self.dataset is None:
            raise ValueError("Modelo no entrenado")

        # Codificar target_users
        encoded_user = self.dataset.encode_target_users(project.target_users)
        features = project.to_features()
        features[5] = encoded_user

        prediction_encoded = self.model.predict([features])[0]
        return self.dataset.decode_platform(prediction_encoded)