# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 05:00:45 2017

@author: johan
"""
#Lab2

#Lab1

from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X, y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)