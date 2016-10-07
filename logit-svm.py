"""
Compare logistic regression to support vector machine (SVM) for binary classification task.
Demonstrate overfitting in the presence of noise/outliers.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, svm

# Load the Iris dataset
# included with sklearn installation
iris = datasets.load_iris()

# Keep only the examples belonging to
# the first two classes (0 and 1)
X = iris.data[:100] 
y = iris.target[:100]

# Keep only the first two features
# to simplify our visualization
X = X[:, :2] 

# Split the data into training/testing sets
X_train = X[0::2]
y_train = y[0::2]
X_test = X[1::2]
y_test = y[1::2]

# Instantiate logistic regression classifier
logistic = linear_model.LogisticRegression()

# Fit logistic regression
logistic.fit(X_train, y_train)

# Classify the testing set using the fitted model
logistic_score = logistic.score(X_test, y_test)*100

# SVM regularization parameter
C = 0.09 

# Instantiate SVM classifiers with 
# linear, polynomial, and radial basis function kernels
lin_svc = svm.SVC(kernel='linear', C=C)

# Fit SVM
lin_svc.fit(X_train, y_train)

# Classify the testing set using the fitted model
svc_score = lin_svc.score(X_test, y_test)*100

# Range of values for each axis in the plot
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1

# Titles of plots
title = 'Logistic regression vs. linear SVM'

plt.figure(figsize=(20.0, 10.0))
plt.subplot(1, 2, 1)

# Plot decision boundary of logistic regression
w = logistic.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(x_min, x_max) 
yy = a * xx - (logistic.intercept_[0]) / w[1]
plt.plot(xx, yy, linewidth=5, linestyle='--')

# Plot decision boundary of SVM
w = lin_svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(x_min, x_max) 
yy = a * xx - (lin_svc.intercept_[0]) / w[1]
plt.plot(xx, yy, linewidth=5)

plt.scatter(X_test[:25, 0], X_test[:25, 1], marker='o', s=50, cmap=plt.cm.Paired)
plt.scatter(X_test[25:, 0], X_test[25:, 1], marker='x', s=50, cmap=plt.cm.Paired)
plt.scatter(X_train[:25, 0], X_train[:25, 1], marker='o', s=50, alpha=0.4)
plt.scatter(X_train[25:, 0], X_train[25:, 1], marker='x', s=50, alpha=0.4)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title(title)

# Print prediction accuracy on plots
plt.text(x_min + .2, y_max - .2, ('LR score: %0.f' % logistic_score).lstrip('0') + '%',
        size=30, horizontalalignment='left')
plt.text(x_min + .2, y_max - .4, ('SVM score: %0.f' % svc_score).lstrip('0') + '%',
        size=30, horizontalalignment='left')

##################################################
# Create outliers for the 0 class
# to see how they affect the decision boundary
##################################################

for i in range(5):
    X_train[i] = [6.5 + i/40., 3.0 + i/40.]

# Refit
logistic.fit(X_train, y_train)
lin_svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)

# Retest
logistic_score = logistic.score(X_test, y_test)*100
svc_score = lin_svc.score(X_test, y_test)*100

plt.subplot(1, 2, 2)

# Plot decision boundary of logistic regression
w = logistic.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(x_min, x_max) 
yy = a * xx - (logistic.intercept_[0]) / w[1]
plt.plot(xx, yy, linewidth=5, linestyle='--')

# Plot decision boundary of SVM
w = lin_svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(x_min, x_max) 
yy = a * xx - (lin_svc.intercept_[0]) / w[1]
plt.plot(xx, yy, linewidth=5)

plt.scatter(X_test[:25, 0], X_test[:25, 1], marker='o', s=50, cmap=plt.cm.Paired)
plt.scatter(X_test[25:, 0], X_test[25:, 1], marker='x', s=50, cmap=plt.cm.Paired)
plt.scatter(X_train[:25, 0], X_train[:25, 1], marker='o', s=50, alpha=0.4)
plt.scatter(X_train[25:, 0], X_train[25:, 1], marker='x', s=50, alpha=0.4)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title(title + ' (in the presence of outliers)')

# Print prediction accuracy on plots
plt.text(x_min + .2, y_max - .2, ('LR score: %0.f' % logistic_score).lstrip('0') + '%',
        size=30, horizontalalignment='left')
plt.text(x_min + .2, y_max - .4, ('SVM score: %0.f' % svc_score).lstrip('0') + '%',
        size=30, horizontalalignment='left')

plt.savefig('logit-svm.png', bbox_inches='tight')
plt.show()
