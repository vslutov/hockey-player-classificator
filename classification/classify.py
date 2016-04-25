#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import csv

import numpy as np

from skimage import color
from sklearn import neighbors

from markup import get_filepath

def validation(X, y):
    _X, _y = np.array(X), np.array(y)
    X = []
    y = []
    for i in range(_y.shape[0]):
        if _y[i] not in [0, 4]:
            X.append(_X[i])
            y.append(_y[i])

    X = np.array(X)
    y = np.array(y)

    train_size = int(X.shape[0] * 0.2)

    X_train, X_test, y_train, y_test = \
        X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        # cross_validation.train_test_split(X, y, test_size=0.8, random_state=42)

    clf = neighbors.KNeighborsClassifier(10, weights='distance', metric='manhattan')
    clf.fit(X_train, y_train)
    result = clf.predict(X_test)
    result_mask = result == y_test

    print(result_mask.sum() / result.shape[0])

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
        image_dir = get_filepath(hockey_dir, 'images')
        gt_filepath = get_filepath(image_dir, 'gt.txt')
        samples = np.load(get_filepath(hockey_dir, 'samples.npy'))

        features = []
        labels = []

        with open(gt_filepath, 'r') as gt:
            for sample_gt in csv.reader(gt):
                sample = samples[int(sample_gt[0])]
                sample_hist = hist(sample, bins=bins)

                features.append(sample_hist)
                labels.append(int(sample_gt[-1]))

        features = np.array(features)
        labels = np.array(labels)
        np.savez(hist_filename, features=features, labels=labels)
        return features, labels

def classify(hockey_dir):
    features, labels = get_histograms(hockey_dir, 5)

    validation(features, labels)
