from tree import CART
import numpy as np

def bootsrap(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices] 

def RandomForestList (X, y, n_trees, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=5, bootsrap=False):
    trees_list = []
    X_sample_list = []
    for _ in range(n_trees):
        if bootsrap:
            X_sample, y_sample = bootsrap(X, y)
        else:
            X_sample= X
            y_sample= y
        tree = CART(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        tree.rndm_forest = True
        tree.max_features = max_features
        tree.fit(X_sample, y_sample)
        X_sample_list.append(tree.X_sample)
        trees_list.append(tree)
    return trees_list, X_sample_list

def PredictRandomForest (trees_list, X):
    tree_preds = np.array([tree.predict(X) for tree in trees_list])
    y_pred = np.array([np.bincount(sample_preds).argmax() for sample_preds in tree_preds.T])
    return y_pred