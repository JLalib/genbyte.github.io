from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

def entrenar_y_evaluar_kmeans(X, y, k):
    # Entrenar modelo KMeans
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo.fit(X)

    # Obtener asignaciones de clusters
    clusters = modelo.labels_

    # Calcular métricas
    inertia = modelo.inertia_
    silhouette = silhouette_score(X, clusters)
    rand_score = adjusted_rand_score(y, clusters)

    # Retornar resultados
    return {
        "clusters": clusters,
        "inertia": inertia,
        "silhouette_score": silhouette,
        "adjusted_rand_score": rand_score
    }