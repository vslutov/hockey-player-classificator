#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import csv

import numpy as np

from sklearn.externals import joblib
from sklearn import cluster, ensemble

from markup import get_filepath

HIST_SIZE = 100

def prepare_buckets(colors):
    buckets = cluster.MiniBatchKMeans(n_clusters=HIST_SIZE, batch_size=1000)
    buckets.fit(colors.reshape((-1, 4))[:10**6, :3])
    return buckets

def extract_feature(img, buckets):
    img = img.reshape((-1, 4))
    def _get_hist(img):
        img = img[img[:,3] > 0.5][:, :3]
        hist = np.bincount(buckets.predict(img), minlength=HIST_SIZE)
        return hist.astype(np.float32) / hist.sum()
    return np.hstack([_get_hist(img[:img.shape[0] // 2]),
                      _get_hist(img[img.shape[0] // 2:])])

def get_ground_truth(hockey_dir):
    hist_filename = get_filepath(hockey_dir, 'hist.npz')
    if os.path.isfile(hist_filename):
        npzfile = np.load(hist_filename)
        return npzfile['features'], npzfile['labels']

    gt_filepath = get_filepath(hockey_dir, 'gt.txt')

    colors = []
    labels = []

    with open(gt_filepath, 'r') as gt:
        vals = [[int(elem) for elem in line] for line in csv.reader(gt)]
        keys = [line[0] for line in vals]
        labels = [line[1] for line in vals]

    pack_size = np.load(get_filepath(hockey_dir, 'samples_0.npy')).shape[0]
    for i in range(max(keys) // pack_size + 1):
        try:
            new_samples = np.load(get_filepath(hockey_dir, 'samples_{i}.npy'.format(i=i)))
            print("read sample pack", i, "of", max(keys) // pack_size + 1)
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
    # np.savez(hist_filename, features=features, labels=labels)
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

    clf = ensemble.RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, clf_filename)

    return clf

def classify(hockey_dir):
    features, labels = get_ground_truth(hockey_dir)
    clf = get_classifier(hockey_dir, features, labels)
    print("Classificator created!")

    # with open(get_filepath(hockey_dir, 'gt.txt'), 'r') as gt:
    #     vals = [[int(elem) for elem in line] for line in csv.reader(gt)]
    #     keys = [line[0] for line in vals]
    #     labels = [line[1] for line in vals]

    # colors = []
    # for i in itertools.count():
    #     try:
    #         new_samples = np.load(get_filepath(hockey_dir, 'samples_{i}.npy'.format(i=i)))
    #         colors.extend(sample.reshape((-1, 4)) for sample in new_samples)
    #     except FileNotFoundError:
    #         break

    # colors = np.array(list(colors[key] for key in keys), dtype=np.float32)
    # buckets = joblib.load(get_filepath(hockey_dir, 'buckets.pkl'))

    # count = 1

    # start = time.clock()
    # for i in range(count):
    #     features = []
    #     for img in colors:
    #         features.append(extract_feature(img, buckets))

    #     clf.predict(features)
    # elapsed = time.clock() - start

    # print(elapsed / count / len(features))
