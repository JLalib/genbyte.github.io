# Importamos las bibliotecas necesarias
import numpy as np # Para trabajar con arrays numéricos y generación de datos aleatorios
import pandas as pd # Para facilitar la organización de datos, útil para la visualización
import matplotlib.pyplot as plt # Para crear gráficos
from sklearn.model_selection import train_test_split # Para dividir el dataset
from sklearn.svm import SVC # El clasificador de Máquinas de Vectores de Soporte (SVM)
from sklearn.metrics import confusion_matrix, classification_report # Para evaluar el modelo
from sklearn.preprocessing import StandardScaler # Para escalar las características, importante para SVM

# 2. Clase Piece: Representa una única pieza industrial
# Guarda sus características (atributos) y si es "Correcta" o "Defectuosa".
class Piece:
    # El constructor de la clase. Se llama al crear un nuevo objeto Piece.
    def __init__(self, texture, symmetry, edges, center_offset, label):
        self.texture = texture          # Textura de la pieza (flotante entre 0 y 1)
        self.symmetry = symmetry        # Simetría de la pieza (flotante entre 0 y 1)
        self.edges = edges              # Cantidad de bordes (entero)
        self.center_offset = center_offset # Desviación del centro (flotante positivo)
        self.label = label              # Etiqueta: "Correcta" o "Defectuosa"

    # Método para convertir los atributos en un formato vectorial (lista o array)
    # Este formato es el que los modelos de Machine Learning esperan como entrada (X).
    def to_vector(self):
        return [self.texture, self.symmetry, self.edges, self.center_offset]

    # Método especial para representar el objeto como una cadena, útil para imprimirlo
    def __repr__(self):
        return f"Piece(texture={self.texture:.2f}, symmetry={self.symmetry:.2f}, edges={self.edges}, center_offset={self.center_offset:.2f}, label='{self.label}')"

# 3. Clase PieceDatasetGenerator: Genera el conjunto de datos simulado
# Crea un número 'n' de piezas con características y etiquetas según reglas predefinidas.
class PieceDatasetGenerator:
    # El constructor. Recibe 'n', el número de piezas a generar.
    def __init__(self, n=400):
        self.n = n # Guarda el número de piezas

    # Método para generar las piezas y sus etiquetas
    def generate(self):
        np.random.seed(42) # Fija una semilla para que los resultados aleatorios sean los mismos cada vez que se ejecuta
        # Genera arrays de números aleatorios para cada característica
        # Usamos distribución normal con media y desviación estándar especificadas
        # 'size=self.n' asegura que cada array tenga 'n' elementos
        textures = np.random.normal(0.5, 0.15, size=self.n)
        symmetries = np.random.normal(0.6, 0.2, size=self.n)
        # Para los bordes, generamos números, aseguramos que sean no negativos (clip) y los convertimos a enteros (astype).
        edges = np.clip(np.random.normal(50, 15, size=self.n), 0, None).astype(int)
        # Para el offset, generamos números (media 0), tomamos el valor absoluto (np.abs) para que sean positivos.
        offsets = np.abs(np.random.normal(0, 0.2, size=self.n))

        data = [] # Lista para guardar los objetos Piece generados
        # Iteramos sobre las características generadas para crear cada pieza
        for t, s, e, o in zip(textures, symmetries, edges, offsets):
            # Aplicamos las reglas de clasificación para determinar la etiqueta
            # Regla 1: Si simetría < 0.4 Y offset > 0.25 -> Defectuosa
            if s < 0.4 and o > 0.25:
                label = "Defectuosa"
            # Regla 2: Si textura < 0.35 O bordes < 30 O offset > 0.35 -> Defectuosa
            elif t < 0.35 or e < 30 or o > 0.35:
                label = "Defectuosa"
            # Si no cumple ninguna regla anterior -> Correcta
            else:
                label = "Correcta"
            # Creamos un objeto Piece con las características y la etiqueta, y lo añadimos a la lista
            data.append(Piece(t, s, e, o, label))

        return data # Devolvemos la lista de objetos Piece

# 4. Clase PieceClassifier: Implementa y gestiona el modelo SVM
# Se encarga de entrenar el modelo, hacer predicciones y evaluar su rendimiento.
class PieceClassifier:
    # El constructor. Inicializa el modelo SVM y el escalador.
    def __init__(self):
        # Inicializa el modelo SVC (Support Vector Classifier)
        # kernel='rbf': Un tipo común de kernel para SVM no lineal
        # gamma='scale': Determina la influencia de un solo ejemplo de entrenamiento. 'scale' es una opción común que usa 1 / (n_features * X.var())
        # C=1.0: Parámetro de regularización. Controla el compromiso entre margen suave y clasificación correcta de los puntos de entrenamiento.
        self.model = SVC(kernel='rbf', gamma='scale', C=1.0)
        # Inicializa el escalador de características.
        # StandardScaler centra los datos a media 0 y escala a varianza 1.
        # Esto es crucial para SVMs que son sensibles a la escala de las características.
        self.scaler = StandardScaler()

    # Método para entrenar el modelo
    # Recibe una lista de objetos Piece (los datos de entrenamiento)
    def fit(self, pieces):
        # Extrae las características (X) de la lista de piezas
        # Cada p.to_vector() es una lista, np.array() las convierte en un array 2D (samples x features)
        X = np.array([p.to_vector() for p in pieces])
        # Extrae las etiquetas (y) de la lista de piezas
        # np.array() las convierte en un array 1D (labels)
        y = np.array([p.label for p in pieces])

        # Escala las características de los datos de entrenamiento
        # fit_transform: Calcula la media y desviación estándar de X, y luego transforma X.
        # El escalador APRENDE DE LOS DATOS DE ENTRENAMIENTO.
        X_scaled = self.scaler.fit_transform(X)

        # Entrena el modelo SVM con los datos escalados y sus etiquetas
        self.model.fit(X_scaled, y)

    # Método para predecir la clase de una nueva pieza
    # Recibe los 4 atributos de la pieza
    def predict(self, texture, symmetry, edges, offset):
        # Crea un array NumPy 2D con los atributos de la única pieza
        # [[...]] crea una fila con las 4 columnas, formato esperado por transform y predict
        piece_vector = np.array([[texture, symmetry, edges, offset]])
        # Escala el vector de la pieza usando el escalador YA ENTRENADO con los datos de entrenamiento.
        # transform: Aplica la misma transformación (usando la media y varianza APRENDIDAS) a los nuevos datos.
        piece_scaled = self.scaler.transform(piece_vector)
        # Realiza la predicción usando el modelo entrenado
        return self.model.predict(piece_scaled)[0] # [0] para obtener el resultado del array de una sola predicción

    # Método para evaluar el rendimiento del modelo en un conjunto de prueba
    # Recibe una lista de objetos Piece (los datos de prueba)
    def evaluate(self, test_data):
        # Extrae características (X_test) y etiquetas verdaderas (y_true) de los datos de prueba
        X_test = np.array([p.to_vector() for p in test_data])
        y_true = np.array([p.label for p in test_data])

        # Escala las características de los datos de prueba usando el escalador ENTRENADO previamente.
        X_test_scaled = self.scaler.transform(X_test)

        # Realiza predicciones en los datos de prueba escalados
        y_pred = self.model.predict(X_test_scaled)

        # Imprime la matriz de confusión y el informe de clasificación
        print("📊 Matriz de confusión:")
        # Muestra cuántas piezas de cada clase real fueron clasificadas en cada clase predicha.
        # [[Verdaderos Correctos, Falsos Defectuosos], [Falsos Correctos, Verdaderos Defectuosos]]
        print(confusion_matrix(y_true, y_pred))
        print("\n📝 Informe de clasificación:")
        # Proporciona métricas clave (precisión, recall, f1-score, soporte) por clase.
        # target_names: Etiquetas de las clases para el informe.
        print(classification_report(y_true, y_pred, target_names=["Correcta", "Defectuosa"]))


# 5. Clase PieceAnalysisExample: Orquesta todo el proceso
# Aquí se llama a las otras clases para generar datos, entrenar, evaluar y visualizar.
class PieceAnalysisExample:
    # El constructor. Simple en este caso.
    def __init__(self):
        pass # No necesita inicializar nada aquí

    @staticmethod
    def run():
        print("Generando dataset...")
        # Crea un generador y genera las 400 piezas
        generator = PieceDatasetGenerator(n=400)
        pieces = generator.generate()
        print(f"Dataset generado con {len(pieces)} piezas.")

        print("Dividiendo datos en entrenamiento y prueba...")
        # Divide la lista COMPLETA de objetos Piece en conjuntos de entrenamiento y prueba (70/30)
        # random_state: Asegura que la división sea la misma cada vez.
        # stratify: Mantiene la misma proporción de clases en ambos conjuntos.
        # Verificamos si hay al menos 2 muestras de cada clase para estratificar
        labels = [p.label for p in pieces]
        if len(set(labels)) < 2 or min([labels.count(l) for l in set(labels)]) < 2:
             print("Advertencia: No se puede estratificar por falta de muestras por clase. Usando división no estratificada.")
             train, test = train_test_split(pieces, test_size=0.3, random_state=42)
        else:
             train, test = train_test_split(pieces, test_size=0.3, random_state=42, stratify=labels)

        print(f"Tamaño del conjunto de entrenamiento: {len(train)}")
        print(f"Tamaño del conjunto de prueba: {len(test)}")

        print("Inicializando y entrenando clasificador...")
        # Crea una instancia del clasificador
        classifier = PieceClassifier()
        # Entrena el clasificador usando SOLO los datos de entrenamiento
        classifier.fit(train)
        print("Modelo entrenado.")

        print("Evaluando modelo en datos de prueba...")
        # Evalúa el rendimiento del modelo usando SOLO los datos de prueba (datos que el modelo no ha visto)
        classifier.evaluate(test)

        # Sección para probar una pieza personalizada
        print(f"\n🔎 Predicción de pieza personalizada:")
        sample_piece_attrs = (0.45, 0.5, 45, 0.15) # Atributos de la pieza de ejemplo
        # Usa el método predict del clasificador para obtener la predicción
        # El *sample_piece_attrs desempaqueta la tupla en argumentos separados
        prediction = classifier.predict(*sample_piece_attrs)
        print(f"  → Textura: {sample_piece_attrs[0]:.2f}, Simetría: {sample_piece_attrs[1]:.2f}, Bordes: {sample_piece_attrs[2]}, Offset: {sample_piece_attrs[3]:.2f}")
        print(f"  → Clasificación: {prediction}")

        # Sección de visualización de resultados
        print("\nVisualizando resultados (dataset completo)...")
        # Crea un DataFrame de pandas con todos los datos para facilitar la visualización
        df = pd.DataFrame({
            "Textura": [p.texture for p in pieces],
            "Simetría": [p.symmetry for p in pieces],
            "Bordes": [p.edges for p in pieces],
            "Offset": [p.center_offset for p in pieces],
            "Etiqueta": [p.label for p in pieces] # Usamos la etiqueta real para colorear
        })

        # Define un diccionario para mapear etiquetas a colores
        colores = {"Correcta": "green", "Defectuosa": "red"}

        # Crea una figura y ejes para el gráfico
        plt.figure(figsize=(8, 6))

        # Itera sobre las etiquetas y colores definidos
        for label, color in colores.items():
            # Filtra el DataFrame para obtener solo las piezas con la etiqueta actual
            subset = df[df["Etiqueta"] == label]
            # Crea un gráfico de dispersión (scatter plot) para este subconjunto
            # Eje X: Textura, Eje Y: Offset
            # c=color: Usa el color definido para la etiqueta
            # label=label: Añade la etiqueta para la leyenda
            # alpha: Transparencia de los puntos
            plt.scatter(subset["Textura"], subset["Offset"], label=label, c=color, alpha=0.6, edgecolors='w', linewidth=0.5) # edgecolors y linewidth mejoran la visibilidad de los puntos

        # Configura etiquetas de ejes, título y leyenda del gráfico
        plt.xlabel("Nivel de textura (homogeneidad)")
        plt.ylabel("Desviación del centro de masa")
        plt.title("🏭 Clasificación de piezas industriales")
        plt.grid(True, linestyle='--', alpha=0.6) # Añade una cuadrícula al gráfico
        plt.legend(title="Estado de la pieza") # Muestra la leyenda con los colores y etiquetas
        # Establece los límites de los ejes para una mejor visualización de los datos
        plt.xlim(0, 1)
        plt.ylim(0, 0.5)
        # Muestra el gráfico
        plt.show()

# 6. Ejecuta el ejemplo si el script se corre directamente
# Esto es una convención en Python para hacer que el código sea ejecutable.
if __name__ == "__main__":
    # Crea una instancia de la clase principal de ejemplo
    example = PieceAnalysisExample()
    # Ejecuta el proceso completo
    example.run()