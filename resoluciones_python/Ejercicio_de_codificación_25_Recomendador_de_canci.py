import numpy as np
from sklearn.neighbors import NearestNeighbors


class Song:
    def __init__(self, title, artist, energy, danceability, duration, popularity):
        self.title = title
        self.artist = artist
        self.energy = energy
        self.danceability = danceability
        self.duration = duration
        self.popularity = popularity

    def to_vector(self):
        return [self.energy, self.danceability, self.duration, self.popularity]

    def __str__(self):
        return f"{self.title} by {self.artist}"


class SongRecommender:
    def __init__(self, k):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k + 1)  # +1 para excluir la propia canción si aparece
        self.songs = []

    def fit(self, song_list):
        self.songs = song_list
        features = [song.to_vector() for song in song_list]
        self.model.fit(features)

    def recommend(self, target_song):
        target_vector = np.array(target_song.to_vector()).reshape(1, -1)
        distances, indices = self.model.kneighbors(target_vector)

        recommendations = []
        for idx in indices[0]:
            song = self.songs[idx]
            if song.title != target_song.title:  # Excluye la canción objetivo
                recommendations.append(song)
            if len(recommendations) == self.k:
                break
        return recommendations


class SongGenerator:
    def __init__(self, num_songs=30):
        self.num_songs = num_songs

    def generate(self):
        songs = []
        for i in range(self.num_songs):
            title = f"Song{i+1}"
            artist = f"Artist{np.random.randint(1, 10)}"
            energy = np.random.uniform(0.4, 1.0)
            danceability = np.random.uniform(0.4, 1.0)
            duration = np.random.randint(180, 301)
            popularity = np.random.randint(50, 101)
            songs.append(Song(title, artist, energy, danceability, duration, popularity))
        return songs


class SongRecommendationExample:
    def run(self):
        generator = SongGenerator()
        song_list = generator.generate()

        # Canción personalizada como objetivo
        target_song = Song("Mi Canción", "Mi Artista", 0.8, 0.9, 240, 90)
        song_list.append(target_song)  # Añadirla para que el modelo la reconozca

        recommender = SongRecommender(k=3)
        recommender.fit(song_list)
        recommendations = recommender.recommend(target_song)

        print(f"\n🎵 Recomendaciones para '{target_song.title}':")
        for song in recommendations:
            print(f" - {song}")


# Ejecutar ejemplo
if __name__ == "__main__":
    example = SongRecommendationExample()
    example.run()