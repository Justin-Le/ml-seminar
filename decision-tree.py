print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Load data
iris = load_iris()

# Keep only the first two classes 
# and first two features (first 100 examples)
X = iris.data[:100, :2]
y = iris.target[:100]
n_classes = 2

# Train
clf = DecisionTreeClassifier()
clf.fit(X, y)
print clf.score(X, y)

# Resolution and color-scheme of plot
resolution = 0.02
colors = "bry"

# Plot colored regions
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                     np.arange(y_min, y_max, resolution))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 20))
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

# Plot training data
for i, color in zip(range(n_classes), colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, s=80,
                label=iris.target_names[i], cmap=plt.cm.Paired)

tree.export_graphviz(clf, out_file='tree.dot')   

plt.show()
