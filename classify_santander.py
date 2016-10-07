# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier)

def main():
    # Load data as dataframe
    # The last column is the target
    data = pd.read_csv("./train.csv")
    data_original = data

    n_estimators = 30

    models = [DecisionTreeClassifier(max_depth=None),
              RandomForestClassifier(n_estimators=n_estimators)]

    ########################################
    # Train/test without engineered features
    ########################################

    X = data[['var15', 'var38']].as_matrix()
    y = data['TARGET'].as_matrix()

    # count_targets(y)

    print "#"*80 + "\n"
    print "Features: customer age, customer value\n"
    RANDOM_SEED = 1
    # compare_models(models, X, y, RANDOM_SEED)

    ########################################
    # Test the engineered features
    ########################################

    # Create new feature: 1 if num_var4 >= 3, 0 else
    num_products_geq_3 = []
    for n in data['num_var4']:
        if n >= 3:
            num_products_geq_3 += [1]
        else:
            num_products_geq_3 += [0]

    data['num_products_geq_3'] = num_products_geq_3

    X = data[['var15', 'var38', 'num_products_geq_3']].as_matrix()
    y = data['TARGET'].as_matrix()

    print "#"*80 + "\n"
    print "Features: customer age, customer value, number of products\n"
    # compare_models(models, X, y, RANDOM_SEED)

    # Create new feature: number of zeros in a row
    X = data_original.iloc[:, :-1]
    y = data_original.TARGET
    data['n0'] = (X==0).sum(axis=1)

    X = data[['var15', 'var38', 'num_products_geq_3', 'n0', 'saldo_var30', 'saldo_medio_var5_ult3']].as_matrix()
    y = data['TARGET'].as_matrix()

    print "#"*80 + "\n"
    print "Features: customer age, customer value, number of products, number of zeros\n"
    compare_models(models, X, y, RANDOM_SEED)

def compare_models(models, X, y, RANDOM_SEED = 1):
    """
    Input: list of models to compare; 
           data as numpy array; targets as numpy array;
           seed for shuffling prior to train/test split
    Output: print cross validation score
    """
 
    for model in models:
        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        X_test = X[ : len(X)/2]
        y_test = y[ : len(X)/2]
        X_train = X[len(X_test) + 1 : ]
        y_train = y[len(X_test) + 1 : ]

        # Standardize train/test sets separately
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train = (X_train - mean) / std

        mean = X_test.mean(axis=0)
        std = X_test.std(axis=0)
        X_test = (X_test - mean) / std

        # Train
        clf = clone(model)
        clf = model.fit(X_train, y_train)

        scores = cross_val_score(clf, X_train, y_train, cv=10)
        # scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1')

        model_details = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(len(model.estimators_))

        print model_details
        print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
        print confusion_matrix(y_test, clf.predict(X_test))

def count_targets(y):
    """
    Input: numpy array containing binary targets for classification
    Output: count of each target value
    """

    print "\nNumber of instances per class:"
    print "Class 0: " + y.tolist().count(0)
    print "Class 1: " + y.tolist().count(1)

if __name__ == "__main__":
    main()
