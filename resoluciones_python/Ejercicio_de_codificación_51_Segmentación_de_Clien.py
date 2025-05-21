import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import unittest
import sys

class CustomerSegmentationModel:
    def __init__(self, n_customers=500):
        try:
            self.n_customers = int(n_customers)
        except Exception:
            self.n_customers = 500
        self.data = None
        self.kmeans = None
        self.logreg = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = None

    def generate_data(self):
        np.random.seed(42)
        self.data = pd.DataFrame({
            'total_spent': np.random.uniform(100, 5000, self.n_customers),
            'total_purchases': np.random.randint(1, 101, self.n_customers),
            'purchase_frequency': np.random.uniform(1, 30, self.n_customers),
            'will_buy_next_month': np.random.choice([0, 1], size=self.n_customers, p=[0.7, 0.3])
        })

    def segment_customers(self, n_clusters=3):
        if self.data is None:
            self.generate_data()
        features = self.data[['total_spent', 'total_purchases', 'purchase_frequency']]
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data['customer_segment'] = self.kmeans.fit_predict(features)

    def train_model(self):
        if self.data is None:
            raise ValueError("Los datos no han sido generados. Llama primero a generate_data().")
        if 'customer_segment' not in self.data.columns:
            raise ValueError("Los datos no han sido segmentados. Llama primero a segment_customers().")

        X = self.data[['total_spent', 'total_purchases', 'purchase_frequency', 'customer_segment']].values
        y = self.data['will_buy_next_month'].values

        X_train, X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.logreg = LogisticRegression(max_iter=1000)
        self.logreg.fit(X_train, y_train)
        self.y_pred = self.logreg.predict(X_test)

        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        return self.accuracy

    def get_accuracy(self):
        if self.accuracy is None:
            raise ValueError("Primero debes entrenar el modelo con train_model().")
        return self.accuracy

    def get_confusion_matrix(self):
        if self.y_test is None or self.y_pred is None:
            raise ValueError("El modelo no ha sido entrenado todavía.")
        return confusion_matrix(self.y_test, self.y_pred)

# --- Pruebas unitarias ---
class TestCustomerSegmentationModel(unittest.TestCase):
    def setUp(self):
        self.model = CustomerSegmentationModel()
        self.model.generate_data()
        self.model.segment_customers()
        self.model.train_model()

    def test_data_generated(self):
        self.assertIsNotNone(self.model.data)

    def test_customer_segment_column(self):
        self.assertIn('customer_segment', self.model.data.columns)

    def test_model_training(self):
        accuracy = self.model.get_accuracy()
        self.assertGreaterEqual(accuracy, 0.6)

    def test_confusion_matrix_shape(self):
        matriz = self.model.get_confusion_matrix()
        self.assertEqual(matriz.shape, (2, 2))

# --- Ejecución Principal ---
if __name__ == "__main__":
    if 'test' in sys.argv:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
        modelo = CustomerSegmentationModel()
        modelo.generate_data()
        modelo.segment_customers()
        modelo.train_model()
        print("Precisión del modelo:", round(modelo.get_accuracy() * 100, 2), "%")
        print("Matriz de confusión:\n", modelo.get_confusion_matrix())