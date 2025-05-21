import numpy as np
from sklearn.ensemble import RandomForestClassifier

class VideoGame:
    def __init__(self, action, strategy, graphics, difficulty, liked=None):
        self.action = action
        self.strategy = strategy
        self.graphics = graphics
        self.difficulty = difficulty
        self.liked = liked

    def to_vector(self):
        return [self.action, self.strategy, self.graphics, self.difficulty]

class VideoGameGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        games = []
        for _ in range(self.num_samples):
            action = np.random.rand()
            strategy = np.random.rand()
            graphics = np.random.rand()
            difficulty = np.random.rand()

            liked = int((action > 0.7 or graphics > 0.7) and difficulty < 0.7)
            games.append(VideoGame(action, strategy, graphics, difficulty, liked))
        return games

class VideoGameClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def fit(self, games):
        X = [game.to_vector() for game in games]
        y = [game.liked for game in games]
        self.model.fit(X, y)

    def predict(self, game):
        pred = self.model.predict([game.to_vector()])[0]
        return pred

class VideoGameRecommendationExample:
    def run(self):
        generator = VideoGameGenerator(num_samples=100)
        games = generator.generate()

        classifier = VideoGameClassifier()
        classifier.fit(games)

        new_game = VideoGame(action=0.9, strategy=0.4, graphics=0.8, difficulty=0.3)

        prediction = classifier.predict(new_game)

        print("🎮 Nuevo juego:")
        print(f"Action: {new_game.action}, Strategy: {new_game.strategy}, Graphics: {new_game.graphics}, Difficulty: {new_game.difficulty}")
        print(f"✅ Le gustará al jugador el juego? {'Si!' if prediction == 1 else 'No :('}")

# Ejemplo de uso
if __name__ == "__main__":
    example = VideoGameRecommendationExample()
    example.run()