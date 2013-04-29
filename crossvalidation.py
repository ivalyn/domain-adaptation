#!/usr/bin/env python
# encoding: utf-8
"""
crossvalidation.py
Created by Oana Sandu

Crossvalidates runs of several domain adaptation methods on given data and returns the results of all the runs
"""

import sys
import random
import functions
from adaptationstrategy import PriorStrategy
from adaptationstrategy import IndomainStrategy
from adaptationstrategy import TransferStrategy
from adaptationstrategy import EnsembleStrategy

def crossvalidate(in_source, in_labelled_target, in_unlabelled_target, adaptation_methods, wordindex, numfolds):
    # performs cross-validation by running an iteration on each of the folds in the target data

    source = functions.format(in_source, wordindex)
    target = functions.format(in_labelled_target, wordindex)
    target_unlabelled = functions.format(in_unlabelled_target, wordindex)
    random.shuffle(target)
    print target, numfolds
    target_folds = makefolds(target, numfolds)
    
    train_target = []
    test_target = []
    results = []
    targetcopy = target_folds[:]
    for i in range(0, numfolds):
        train_target = []
        test_target = targetcopy[i]
        for j in range(0, numfolds):
            if j != i:
                train_target.extend(targetcopy[j])        
        results.extend(iteration(source, train_target, test_target, target_unlabelled, adaptation_methods, wordindex))
    return results

def iteration(in_source, in_traintarget, in_testtarget, target_unlabelled, adaptation_methods, wordindex):
    # given a split of the data, runs all adaptation algorithms specified in adaptation_methods on the data

    train_source = in_source[:]
    train_target = in_traintarget[:]
    test_target = in_testtarget[:]
    allresults = []
    for method in adaptation_methods:
        if method == 'indomain':
            strategy = IndomainStrategy(train_target, test_target)
        elif method == 'transfer':
            strategy = TransferStrategy(train_source, test_target)
        elif method == 'ensemble':
            strategy = EnsembleStrategy(train_source, train_target, test_target)
        elif method == 'pred':
            strategy = PriorStrategy(train_source, train_target, test_target, len(wordindex) + 1)
        allresults.append(strategy.adapt())
    return allresults

def makefolds(data, numfolds):
    # divides the data into numfolds folds, each of which is a list of data points

    folds = []
    for fold_number in range(0, numfolds):
        folds.append([])

    test_ratio = 1.0 / numfolds
    source_division = int(test_ratio * len(data))
    indices = range(0, len(data))
    random.shuffle(indices)
    for idx, i in enumerate(indices):
        fold_number = idx / source_division
        if fold_number >= len(folds):
            fold_number = len(folds) - 1
        folds[fold_number].append(data[i])
    return folds