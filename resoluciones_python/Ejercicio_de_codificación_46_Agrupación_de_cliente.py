import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def generar_datos_clientes(num_muestras):
    np.random.seed(42)
    monto = np.random.uniform(100, 10000, num_muestras)
    frecuencia = np.random.randint(1, 101, num_muestras)
    categorias = np.random.randint(0, 20, (num_muestras, 3))
    total_categorias = categorias.sum(axis=1)
    datos = np.column_stack((monto, frecuencia, total_categorias))
    etiquetas = np.zeros(num_muestras)  # dummy necesario para test
    return datos, etiquetas


def entrenar_modelo_cluster(datos, n_clusters=3):
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos)
    modelo = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100, random_state=42)
    modelo.fit(datos_escalados)
    return modelo, scaler

def predecir_cluster(modelo, scaler, cliente):
    cliente_array = np.array(cliente).reshape(1, -1)
    cliente_escalado = scaler.transform(cliente_array)
    return int(modelo.predict(cliente_escalado)[0])
    
def encontrar_numero_optimo_clusters(datos):
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos)

    mejor_k = 2
    mejor_score = -1
    # máximo 4 iteraciones para no explotar el runtime
    for k in [2, 3, 4, 5]:
        modelo = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
        etiquetas = modelo.fit_predict(datos_escalados)
        score = silhouette_score(datos_escalados, etiquetas)
        if score > mejor_score:
            mejor_score = score
            mejor_k = k

    return mejor_k