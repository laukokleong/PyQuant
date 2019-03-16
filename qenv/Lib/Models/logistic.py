# USAGE: Build LOGIT for probability modelling

import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

def build_logit(x, y):
    if x is None or y is None:
        return None

    logreg = linear_model.LogisticRegression(C=1e3, fit_intercept=False)
    logreg.fit(x, y)

    return logreg