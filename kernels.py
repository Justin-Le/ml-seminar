"""
Demonstrate the difference in predictive performance
of linear and non-linear kernels in 
support vector machines for binary classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

X, Y = datasets.make_moons(n_samples=200, noise=0.2, random_state=1)
X_test, Y_test = datasets.make_moons(n_samples=200, noise=0.2, random_state=2)

# Subplot index
subplot_num = 1

plt.figure(figsize=(20.0, 10.0))

for kernel in ('linear', 'rbf'):
    # Instantiate the SVM
    clf = svm.SVC(kernel=kernel)

    # Fit the model
    clf.fit(X, Y)

    # Predict classes on testing data
    score = 100.*clf.score(X_test, Y_test)

    # Plot testing data
    ax = plt.subplot(1, 2, subplot_num)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=150, marker='o', zorder=10)

    x_min = -2.8
    x_max = 2.8
    y_min = -2.8
    y_max = 2.8

    # Create contour map
    xx, yy = np.mgrid[x_min:x_max:1000j, y_min:y_max:1000j]
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z > 0, cmap=plt.cm.Paired)
    ax.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    # Print prediction accuracy on plot
    ax.text(xx.min() + .3, yy.min() + .3, ('%.1f' % score) + '% accuracy', fontsize=25)

    subplot_num = subplot_num + 1
plt.savefig('kernels.png', bbox_inches='tight')
plt.show()
