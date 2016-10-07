import numpy as np
from sklearn import linear_model, svm
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import clone
from process_wdbc import *
import pdb

def main():
    data = process_wdbc('wdbc.data')

    # Variable 1 is the target
    y = data[:, 1]
    X = np.delete(data, 1, axis=1)

    # Delete uninformative feature
    X = np.delete(data, 0, axis=1)

    n_estimators = 20
    RANDOM_SEED = 1

    models = [DecisionTreeClassifier(max_depth=None),
              RandomForestClassifier(n_estimators=n_estimators)]

    compare_models(models, X, y, RANDOM_SEED)

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    logistic = linear_model.LogisticRegression()
    logistic.fit(X_train, y_train)
    logistic_predict = logistic.predict(X_test)

    svc = svm.SVC()
    svc.fit(X_train, y_train)
    svc_predict = svc.predict(X_test)

    print logistic.score(X_test, y_test)
    print logistic_predict
    print svc.score(X_test, y_test)
    print svc_predict

    # Count number of predictions for each class
    print logistic_predict.tolist().count(1)
    print svc_predict.tolist().count(1)

    print classification_report(y_test, logistic.predict(X_test))
    # classification_report(y_test, svc.predict(X_test))
    """

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

        scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1')

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
