#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import csv
import itertools

import numpy as np

from matplotlib import pyplot as plt

from skimage import color
from sklearn import neighbors, cross_validation, svm, ensemble, cluster
from sklearn.externals import joblib
import xgboost as xgb
import sys

from markup import get_filepath

def calc_percentage(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    result = clf.predict(X_test)
    result_mask = result == y_test

    return 100 * result_mask.sum() / result.shape[0]

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

    abscissa = np.arange(0.2, 0.95, 0.1)

    for label, clf in [('KNN', neighbors.KNeighborsClassifier(10, weights='distance', metric='manhattan')),
                      ('SVM', svm.LinearSVC(C=10.0)),
                      ('RandomForest', ensemble.RandomForestClassifier(random_state=42)),
                      ('Boosting', ensemble.GradientBoostingClassifier(random_state=42)),
                      ('XGBoost', xgb.XGBClassifier(seed=42))]:

        ordinata = []
        for part in abscissa:
            print(part)
            train_size = int(X.shape[0] * part)

            X_train, X_test, y_train, y_test = \
                X[:train_size], X[train_size:], y[:train_size], y[train_size:]
                # cross_validation.train_test_split(X, y, test_size=1 - part, random_state=42)

            ordinata.append(calc_percentage(clf, X_train, X_test, y_train, y_test))

        plt.plot(abscissa, ordinata, label=label)

    plt.legend(loc='lower left')
    plt.show()

HIST_SIZE = 100

def prepare_buckets(colors):
    buckets = cluster.MiniBatchKMeans(n_clusters=HIST_SIZE, batch_size=1000)
    buckets.fit(colors.reshape((-1, 4))[:10**6, :3])
    return buckets

def extract_feature(img, buckets):
    img = img.reshape((-1, 4))
    def _get_hist(img):
        img = img[img[:,3] != 0][:, :3]
        hist = np.bincount(buckets.predict(img), minlength=HIST_SIZE)
        return hist.astype(np.float32) / hist.sum()
    return np.hstack([_get_hist(img[:img.shape[0] // 2]),
                      _get_hist(img[img.shape[0] // 2:])])

def get_ground_truth(hockey_dir, bins):
    hist_filename = get_filepath(hockey_dir, 'hist.npz')
    if os.path.isfile(hist_filename):
        npzfile = np.load(hist_filename)
        return npzfile['features'], npzfile['labels']
    else:
        gt_filepath = get_filepath(hockey_dir, 'gt.txt')

        colors = []
        labels = []

        with open(gt_filepath, 'r') as gt:
            vals = [[int(elem) for elem in line] for line in csv.reader(gt)]
            keys = [line[0] for line in vals]
            labels = [line[1] for line in vals]

        for i in itertools.count():
            try:
                new_samples = np.load(get_filepath(hockey_dir, 'samples_{i}.npy'.format(i=i)))
                colors.extend(sample.reshape((-1, 4)) for sample in new_samples)
            except FileNotFoundError:
                break

        colors = np.array(list(colors[key] for key in keys), dtype=np.float32)
        buckets = prepare_buckets(colors)
        joblib.dump(buckets, get_filepath(hockey_dir, 'buckets.pkl'))

        features = []
        for img in colors:
            features.append(extract_feature(img, buckets))

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        np.savez(hist_filename, features=features, labels=labels)
        return features, labels

def get_classifier(hockey_dir, X, y):
    clf_filename = get_filepath(hockey_dir, 'player_clf.pkl')
    if os.path.isfile(clf_filename):
        return joblib.load(clf_filename)

    _X, _y = np.array(X), np.array(y)
    X = []
    y = []
    for i in range(_y.shape[0]):
        if _y[i] != 5:
            X.append(_X[i])
            y.append(_y[i])

    X = np.array(X)
    y = np.array(y)

    clf = xgb.XGBClassifier(seed=42)
    clf.fit(X, y)
    joblib.dump(clf, clf_filename)

    return clf

def classify(hockey_dir):
    features, labels = get_ground_truth(hockey_dir, 5)
    clf = get_classifier(hockey_dir, features, labels)
    result = clf.predict(features)
    result_mask = result == labels
    print(100 * result_mask.sum() / result.shape[0])
