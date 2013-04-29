#!/usr/bin/env python
# encoding: utf-8
"""
adaptationstrategy.py
Created by Oana Sandu

Several alternate domain adaptation strategies, each of which records the results and running time
"""

import time
import functions
import classify_liblinear
import ensemble
import prior

class AdaptationStrategy(object):
    def __init__(self, training, testing):
        self.training = training
        self.testing = testing
    def adapt(self):
        starttime = time.time()
        self.result = self.compute()
        self.runtime = time.time() - starttime
        return self.description, self.result, self.runtime

class IndomainStrategy( AdaptationStrategy ):
    def __init__(self, training, testing):
        super(IndomainStrategy,self).__init__(training, testing)
        self.description = 'In-domain'
    def compute(self):
        return classify_liblinear.prediction(self.training, self.testing, self.description)

class TransferStrategy( AdaptationStrategy ):
    def __init__(self, training, testing):
        super(TransferStrategy,self).__init__(training, testing)
        self.description = 'Transfer'
    def compute(self):
        return classify_liblinear.prediction(self.training, self.testing, self.description)
        
class EnsembleStrategy( AdaptationStrategy ):
    def __init__(self, train_source, train_target, test_target):
        self.train_source = train_source
        self.train_target = train_target
        self.test_target = test_target
        self.description = 'Ensemble'
    def compute(self):
        return ensemble.run_ensemble(self.train_source, self.train_target, self.test_target, self.description)

class PriorStrategy( AdaptationStrategy ):  
    def __init__(self, train_source, train_target, test_target, number_features):
        self.train_source = train_source
        self.train_target = train_target
        self.test_target = test_target
        self.numfeat = number_features
        self.description = 'Prior'
    def compute(self):
        pos_train_target = prior.getposterior(self.train_source, self.train_target, self.numfeat)
        pos_test_target = prior.getposterior(self.train_source, self.test_target, self.numfeat)
        train_target_formatted = functions.format_noindex(pos_train_target)
        test_target_formatted = functions.format_noindex(pos_test_target)
        return classify_liblinear.prediction(train_target_formatted, test_target_formatted, self.description)