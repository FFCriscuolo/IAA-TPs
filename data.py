from ucimlrepo import fetch_ucirepo
import numpy as np

# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features.values 
y = breast_cancer_wisconsin_diagnostic.data.targets.values.ravel()
# feature_names = breast_cancer_wisconsin_diagnostic.data.features.columns

# # metadata 
# print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# # variable information 
# print(breast_cancer_wisconsin_diagnostic.variables) 

# Etiquetas a 0/1 (benigno=0, maligno=1 por ejemplo)
classes, y = np.unique(y, return_inverse=True)


# Separación en train/test (80/20)
np.random.seed(40)
indices = np.arange(len(X))
np.random.shuffle(indices)

split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# thresholds, classes = zip(*sorted(zip(X_train[:, 0], y_train)))
