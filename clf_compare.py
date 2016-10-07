#!/usr/bin/python
# -*- coding: utf-8 -*-
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


names = ["Logistic Regression", "Linear SVM", "Decision Tree", "Random Forest"]
classifiers = [
    LogisticRegression(),
    SVC(kernel="linear"),
    DecisionTreeClassifier(),
    RandomForestClassifier()]

iris = load_iris()
X = iris.data
y = iris.target
X = X[:100, :]
y = y[:100]
datasets = [(X, y)]
                           
# Corrupt the data with noise
rng = np.random.RandomState(1)
X += rng.normal(size=X.shape)

"""
X, y = make_classification(n_samples=500, n_classes=2, n_clusters_per_class=1,
                           n_features=2, n_redundant=0, n_informative=2,
                           class_sep=20.0, random_state=0)

linearly_separable = (X, y)

# Corrupt the data with noise
rng = np.random.RandomState(1)
X += 80*rng.uniform(size=X.shape)
linearly_separable_noisy = (X, y)

datasets = [linearly_separable,
            linearly_separable_noisy]

datasets = [make_moons(noise=0.3, random_state=2),
            make_circles(noise=0.2, factor=0.5, random_state=3),
            linearly_separable
            ]
"""

# figure = plt.figure(figsize=(27, 9))
i = 1

# Grid resolution
h = .01

for ds in datasets:
    # Split data into training/testing sets
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    # Standardize each set separately
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    # Create grid for plotting
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    """
    # Plot data
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    """

    # Fit models and plot contour maps of decision boundaries
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers), i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        """
        # Decision function: distance from decision boundary to each grid point
        if hasattr(clf, "decision_function"):
            decisions = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            decisions = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Contour map
        decisions = decisions.reshape(xx.shape)
        ax.contourf(xx, yy, decisions, cmap=cm, alpha=0.8)
        """

        # Plot training and testing data
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, alpha=0.8)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)

        # Print prediction accuracy on plots
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.subplots_adjust(left=0.01, right=0.99, 
                       top=0.95, bottom=0.05, 
                       wspace=0.02, hspace=0.2)

plt.show()
