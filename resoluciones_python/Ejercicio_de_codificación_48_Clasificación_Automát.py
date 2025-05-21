import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def generar_datos_frutas(num_muestras: int):
    """
    Genera datos sintéticos de frutas con características físicas.
    Args:
        num_muestras (int): Número total de frutas a generar.
    Returns:
        X (np.ndarray): Matriz de características con columnas [peso, tamano].
        y (np.ndarray): Vector de etiquetas con valores 'Manzana', 'Plátano', 'Naranja'.
    """
    tipos = ['Manzana', 'Plátano', 'Naranja']
    muestras_por_tipo = num_muestras // len(tipos)
    data = []

    for tipo in tipos:
        if tipo == 'Manzana':
            peso = np.random.normal(loc=180, scale=30, size=muestras_por_tipo)
            tamano = np.random.normal(loc=7.5, scale=0.8, size=muestras_por_tipo)
        elif tipo == 'Plátano':
            peso = np.random.normal(loc=120, scale=20, size=muestras_por_tipo)
            tamano = np.random.normal(loc=17, scale=1.5, size=muestras_por_tipo)
        else:  # Naranja
            peso = np.random.normal(loc=140, scale=25, size=muestras_por_tipo)
            tamano = np.random.normal(loc=6.5, scale=0.7, size=muestras_por_tipo)

        peso = np.clip(peso, 50, None)
        tamano = np.clip(tamano, 2, None)

        for p, t in zip(peso, tamano):
            data.append((float(p), float(t), tipo))

    if len(data) < num_muestras:
        extra = num_muestras - len(data)
        data.extend([data[i % len(data)] for i in range(extra)])

    np.random.shuffle(data)
    data = np.array(data, dtype=object)
    X = data[:, :2].astype(float)
    y = data[:, 2]
    return X, y


def entrenar_modelo(data):
    """
    Entrena un modelo de RandomForest para clasificar frutas.
    Args:
        data (tuple): (X, y) donde X son características y y etiquetas.
    Returns:
        tuple: (clf, label_encoder)
    """
    X, y = data
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Evaluación del modelo en test:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return clf, le


def predecir_fruta(modelo, peso: float, tamano: float) -> str:
    """
    Predice el tipo de fruta para un nuevo ejemplo.
    Args:
        modelo: tupla (clf, label_encoder).
        peso (float): en gramos.
        tamano (float): en cm.
    Returns:
        str: etiqueta predicha.
    """
    clf, le = modelo
    X_new = np.array([[peso, tamano]])
    y_new = clf.predict(X_new)
    return le.inverse_transform(y_new)[0]


if __name__ == "__main__":
    # Generación de datos y entrenamiento
    data = generar_datos_frutas(300)
    modelo = entrenar_modelo(data)

    # Predicción de ejemplos
    ejemplos = [(165, 8), (130, 16), (150, 6)]
    for peso, tamano in ejemplos:
        print(f"Fruta ({peso}g, {tamano}cm) -> {predecir_fruta(modelo, peso, tamano)}")