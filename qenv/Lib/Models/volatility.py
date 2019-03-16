# USAGE: Build GARCH for volatility modelling

import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch.univariate import arch_model

def build_garch11(series):
    ret = None

    if series is None:
        return ret

    garch11 = arch_model(series, dist='studentst')
    ret = garch11.fit(update_freq=5, disp='off', show_warning=False)

    return ret