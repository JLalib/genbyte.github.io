import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------
# 1. Generar datos sintéticos de emails
# ------------------------------------------
def generar_datos_emails(num_muestras=100):
    np.random.seed(42)
    
    longitud_mensaje = np.random.randint(50, 501, size=num_muestras)
    frecuencia_palabra_clave = np.random.rand(num_muestras)  # entre 0 y 1
    cantidad_enlaces = np.random.randint(0, 11, size=num_muestras)

    # Reglas para etiquetar como spam
    etiquetas = np.where(
        (frecuencia_palabra_clave > 0.6) | 
        (cantidad_enlaces > 5) | 
        (longitud_mensaje < 100), 
        1, 0
    )

    X = np.column_stack((longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces))
    y = etiquetas
    return X, y

# ------------------------------------------
# 2. Entrenar modelo SVM
# ------------------------------------------
def entrenar_modelo_svm(datos, etiquetas):
    modelo = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    modelo.fit(datos, etiquetas)
    return modelo

# ------------------------------------------
# 3. Predecir email con salida correcta
# ------------------------------------------
def predecir_email(modelo, longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces):
    entrada = np.array([[longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces]])
    pred = modelo.predict(entrada)[0]
    return "El email es Spam" if pred == 1 else "El email no es Spam"