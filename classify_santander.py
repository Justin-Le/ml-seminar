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
from utils import *

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
    compare_models(models, X, y, RANDOM_SEED)

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
    compare_models(models, X, y, RANDOM_SEED)

    # Create new feature: number of zeros in a row
    X = data_original.iloc[:, :-1]
    y = data_original.TARGET
    data['n0'] = (X==0).sum(axis=1)

    X = data[['var15', 'var38', 'num_products_geq_3', 'n0', 'saldo_var30', 'saldo_medio_var5_ult3']].as_matrix()
    y = data['TARGET'].as_matrix()

    print "#"*80 + "\n"
    print "Features: customer age, customer value, number of products, number of zeros\n"
    compare_models(models, X, y, RANDOM_SEED)

if __name__ == "__main__":
    main()
