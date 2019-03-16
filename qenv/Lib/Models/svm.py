# USAGE: Build SVM for classification

import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

def build_svm(x, y):
    if x is None or y is None:
        return None

    clf = svm.SVC(kernel='rbf', C = 1.0, probability=True)
    clf.fit(x, y)

    return clf

