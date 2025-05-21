import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------------------
# 1. Generar datos sintéticos
# ------------------------------------------
def generar_datos_compras(num_muestras=100):
    np.random.seed(42)
    
    paginas = np.random.randint(1, 21, size=num_muestras)  # 1 a 20 páginas
    tiempo = np.random.uniform(0, 30, size=num_muestras)   # 0 a 30 min

    etiquetas = np.where((paginas > 5) & (tiempo > 10), 1, 0)

    X = np.column_stack((paginas, tiempo))
    y = etiquetas
    return X, y

# ------------------------------------------
# 2. Entrenar modelo
# ------------------------------------------
def entrenar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    return modelo

# ------------------------------------------
# 3. Evaluar modelo
# ------------------------------------------
def evaluar_modelo(modelo, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision:.2f}")
    return precision

# ------------------------------------------
# 4. Predecir con mensaje
# ------------------------------------------
def predecir_compra(modelo, num_paginas_vistas, tiempo_en_sitio):
    entrada = np.array([[num_paginas_vistas, tiempo_en_sitio]])
    prediccion = modelo.predict(entrada)[0]

    if prediccion == 1:
        return "El usuario comprará el producto."
    else:
        return "El usuario no comprará el producto."