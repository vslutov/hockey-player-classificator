#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import csv
import itertools

import numpy as np

from matplotlib import pyplot as plt

from skimage import color
from sklearn import neighbors, cross_validation, svm, ensemble, cluster
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

def hist(sample, bins):
    begin, end = -128, 128.001
    step = (end - begin) / bins
    result = np.zeros((3, bins))

    sample_mask = sample[:, :, 3] > 0.5
    sample[:, :, 0:3] = color.rgb2lab(sample[:, :, 0:3])

    for c in [0, 1, 2]:
        channel = sample[:, :, c]
        for i, start in enumerate(np.linspace(begin, end,
                                              bins, endpoint=False)):
            mask = np.logical_and(channel >= start, channel< start + step)
            val = np.logical_and(mask, sample_mask)
            result[c, i] = np.count_nonzero(val)

    result = result.reshape(-1)
    return result / result.sum()

def get_histograms(hockey_dir, bins):
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

        HIST_SIZE = 100
        HistClf = cluster.MiniBatchKMeans(n_clusters=HIST_SIZE, batch_size=1000)
        HistClf.fit(colors.reshape((-1, 4))[:10**6, :3])

        def get_hist(img):
            img = img[img[:,3] != 0][:, :3]
            hist = np.bincount(HistClf.predict(img), minlength=HIST_SIZE)
            return hist.astype(np.float32) / hist.sum()

        features = []
        for img in colors:
            hist = np.hstack([get_hist(img[:img.shape[0] // 2]),
                              get_hist(img[img.shape[0] // 2:])])
            features.append(hist)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        np.savez(hist_filename, features=features, labels=labels)
        return features, labels

def classify(hockey_dir):
    features, labels = get_histograms(hockey_dir, 5)

    validation(features, labels)
