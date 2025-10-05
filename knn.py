import numpy as np
from collections import Counter, defaultdict


class KnnClassif:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = np.array(y_train)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2)) #uso euclidea y no lo considero como hiperparametro en 5.1

    def _predict(self, x):
        #calculo distancias a todos los puntos
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        #obtengo indices los k vecinos más cercanos
        k_indices = np.argsort(distances)[:self.k]
        #veo etiquetas de esos k vecinos
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        #voto por mayoría
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
class WeightedKnnClassif:
    def __init__(self, k=11, epsilon=1e-5):
        self.k = k
        self.epsilon = epsilon

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict(self, x):
        # 1.calculo distancias a todos los puntos
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]

        # 2.pondero los votos por distancia (pondero con 1/distancia (a menor distancia, peso es mayor) y luego le agrego un minimo valor para que no se rompa en distancia=0 y no --> inf)
        class_weights = defaultdict(float)
        for i in k_indices:
            label = self.y_train[i]
            dist = distances[i]
            weight = 1 / (dist + self.epsilon)
            class_weights[label] += weight

        # 3.elijo clase con mayor peso total
        return max(class_weights, key=class_weights.get)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])