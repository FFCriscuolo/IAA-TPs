from data import X_train, X_test, y_train, y_test
import numpy as np

# Impureza de Gini
def gini_impurity(y):
    if len(y) == 0:
        return 0 # Evita división por cero
    proportions = np.bincount(y) / len(y) # Bincount cuenta ocurrencias de cada clase (0,1)
    return 1 - np.sum(proportions**2)

# Función para imprimir el árbol
def print_tree(node, depth=0):
    indent = "  " * depth
    print(f"{indent}Node d={depth} samples={node.num_samples} gini={node.gini:.4f} pred={node.predicted_class} feat={node.feature_index} thr={node.threshold}")
    if node.left: print_tree(node.left, depth + 1)
    if node.right: print_tree(node.right, depth + 1)

# Función para encontrar el mejor split, devuelve qué feature y threshold es mejor
# para dividir el dataset y reducir la impureza de Gini, junto con las particiones
def best_split(X, y):
    m, n = X.shape # m=número de muestras, n=número de features

    best_gini = 1.0 # Inicializamos con el peor caso
    best_idx, best_thr = None, None 
    best_left, best_right = None, None

    # Probamos cada feature
    for feature_idx in range(n):
        # Ordenamos las muestras según la feature actual
        order = np.argsort(X[:, feature_idx])
        vals = X[order, feature_idx]
        labs = y[order]

        # Probamos cada posible punto de corte, discretizando el espacio
        # según el número de muestras
        for i in range(1, m):
            # Si hay dos valores iguales, no tiene sentido partir ahí,
            # evalúa el siguiente
            if vals[i] == vals[i-1]:
                continue

            y_left = labs[:i]
            y_right = labs[i:]

            # Calculamos la impureza de Gini a ambos lados del split
            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)

            # Impureza ponderada
            gini = (len(y_left) * gini_left + len(y_right) * gini_right) / m

            # Si es el mejor split hasta ahora, lo guardamos
            if gini < best_gini:
                best_gini = gini
                best_idx = feature_idx # Índice de la feature
                best_thr = (vals[i] + vals[i - 1]) / 2 # Punto medio entre dos valores

    return best_idx, best_thr, best_gini

# Clase Nodo
class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

# Construcción recursiva del árbol
class CART:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth # Profundidad máxima del árbol
        self.min_samples_split = min_samples_split # Mínimo de muestras para hacer split
        self.min_samples_leaf = min_samples_leaf # Mínimo de muestras en una hoja después del split
        self.tree = None

    def _grow_tree(self, X, y, depth=0):
        classes, counts = np.unique(y, return_counts=True)
        predicted_class = classes[np.argmax(counts)]
        node = Node(
            gini=gini_impurity(y),
            num_samples=len(y),
            num_samples_per_class=counts.tolist(),
            predicted_class=predicted_class,
        )

        if node.gini == 0 or len(y) < self.min_samples_split: # Si todas las muestras son de la misma clase o
            return node # si no hay suficientes muestras para split
        
        if self.max_depth is not None and depth >= self.max_depth:
            return node # Si hemos alcanzado la profundidad máxima
        
        idx, thr, best_gini = best_split(X, y)
        # Solo split si mejora la impureza del nodo
        if idx is None or best_gini >= node.gini: # ¿Epsilon?
            return node

        # Particionar y crear subnodos
        indices_left = X[:, idx] <= thr
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        if len(y_left) >= self.min_samples_leaf and len(y_right) >= self.min_samples_leaf:
            node.feature_index = idx
            node.threshold = thr
            node.left = self._grow_tree(X_left, y_left, depth + 1)
            node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _predict(self, inputs, node):
        if node.left is None and node.right is None:
            return node.predicted_class
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)

    def predict(self, X):
        X = np.asarray(X)
        return [self._predict(np.array(row).ravel(), self.tree) for row in X]