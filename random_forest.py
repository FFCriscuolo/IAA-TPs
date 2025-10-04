import numpy as np
from collections import Counter
from tree import CART
from data import X_train, y_train

class RF:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.max_features = max_features
        self.bootstrap = bootstrap

    def fit(self, X, y):
        self.trees = []
        m = len(X)
        for _ in range(self.n_estimators):
            # Bootstrap sample
            if self.bootstrap:
                indices = np.random.choice(m, m, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y

            tree = CART(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Votación mayoritaria de todos los árboles
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # predictions tiene shape (n_trees, n_samples)
        # Transponemos para tener (n_samples, n_trees)
        predictions = predictions.T  
        final_preds = []
        for row in predictions:
            # Voto mayoritario
            most_common = Counter(row).most_common(1)[0][0]
            final_preds.append(most_common)
        return np.array(final_preds)

# forest = RF(
#     n_estimators=1,
#     max_depth=5,
#     max_features=30,
#     bootstrap=False
# )

# forest.fit(X_train, y_train)
# y_pred = forest.predict(X_train)
# print("Accuracy:", np.mean(y_pred == y_train))

# tree = CART(
#     max_depth=5,
# )

# tree.fit(X_train, y_train)
# y_pred = tree.predict(X_train)
# print("Accuracy:", np.mean(y_pred == y_train))
