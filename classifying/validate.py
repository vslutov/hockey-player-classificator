#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors, cross_validation, svm, ensemble, metrics

from classify import get_ground_truth

def calc_percentage(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)

    return metrics.f1_score(y_test, clf.predict(X_test))

def validation(X, y):
    _X, _y = np.array(X), np.array(y)
    X = []
    y = []
    for i in range(_y.shape[0]):
        if _y[i] != 5:
            X.append(_X[i])
            y.append(_y[i])

    X = np.array(X)
    y = np.array(y)

    clfs = [('KNN', neighbors.KNeighborsClassifier(10, weights='distance', metric='manhattan')),
            ('SVM', svm.LinearSVC(C=10.0)),
            ('RandomForest', ensemble.RandomForestClassifier(random_state=42)),
            ('GradientBoosting', ensemble.GradientBoostingClassifier(random_state=42))]

    ordinata = []

    for label, clf in clfs:


        train_size = int(X.shape[0] * 0.4)

        X_train, X_test, y_train, y_test = \
            cross_validation.train_test_split(X, y, test_size=0.6, random_state=42)
            # X[:train_size], X[train_size:], y[:train_size], y[train_size:]

        ordinata.append(calc_percentage(clf, X_train, X_test, y_train, y_test))
        print(ordinata[-1])

    plt.figure(figsize=(3,3))
    plt.bar(range(len(clfs)), ordinata, align='center')
    plt.xticks(range(len(clfs)), [label for label, clf in clfs])
    plt.ylabel('F1-score')
    plt.title('Compare machine-learning algorithm')
    plt.show()

def validate(hockey_dir):
    features, labels = get_ground_truth(hockey_dir)
    validation(features, labels)
