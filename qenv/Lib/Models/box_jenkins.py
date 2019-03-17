# USAGE: Using Box Jenkins method to build ARIMA model

import warnings
import math
import operator
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats import diagnostic


CONST_INVALID_AIC = 1e6
CONST_SIGNIFICANCE = 0.05

def build_arima(series, exovar=None, diff=0, max_p=5, max_q=5):
    warnings.filterwarnings('ignore')

    ret = None

    # ******** IDENTIFICATION ********
    qstat = diagnostic.acorr_ljungbox(series, lags=max(max_p, max_q))
    #print(qstat[1])

    # look for autocorrelation i.e. reject null hypothesis
    isWhiteNoise = False
    for pval in qstat[1]:
        if pval >= CONST_SIGNIFICANCE:
            isWhiteNoise = True
            break

    if isWhiteNoise:
        return ret

    # ******** ESTIMATION ********
    aic_matrix = {}
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            # p and q cannot be both zero
            if p == 0 and q == 0:
                aic_matrix[p, q] = CONST_INVALID_AIC
                continue

            model_fit = __arima__(series, exovar=exovar, order=(p, diff, q))
            if model_fit != None:
                aic_matrix[p, q] = model_fit.aic
                #print('p=' + str(p) + ' ' + 'q=' + str(q))
                #print(model_fit.arparams)
                for param in model_fit.arparams:
                    if math.isnan(param):
                        aic_matrix[p, q] = CONST_INVALID_AIC
                        break
            else:
                aic_matrix[p, q] = CONST_INVALID_AIC

    # '******** DIAGNOSTIC ********
    while len(aic_matrix):
        # find the minimum aic in matrix
        m = min(aic_matrix.items(), key=operator.itemgetter(1))[0]
        if aic_matrix[m] == CONST_INVALID_AIC:
            aic_matrix.pop(m)
            continue

        # re-estimate model
        model_fit = __arima__(series, exovar=exovar, order=(m[0], diff, m[1]))
        if model_fit is None:
            aic_matrix.pop(m)
            continue

        # ljung box test on residual
        residual = model_fit.resid
        qstat = diagnostic.acorr_ljungbox(residual, lags=max(m[0], m[1]))
        #print(qstat[1])

        # look for white noise i.e. fail to reject null hypothesis
        isWhiteNoise = True
        for pval in qstat[1]:
            if pval < CONST_SIGNIFICANCE:
                isWhiteNoise = False
                break

        # best model found, break from loop
        if isWhiteNoise:
            ret = model_fit
            break

        # not the best model, remove it
        aic_matrix.pop(m)

    #print(ret.pvalues)
    #print(ret.arparams)
    #for x in range(0, len(ret.pvalues)):
    #    if ret.pvalues[x] > CONST_SIGNIFICANCE:
    #        ret.arparams[x] = 0

    return ret

def forecast(model, exog):
    ret = None

    try:
        for i in range(len(exog)):
            if math.isnan(model.pvalues[i]) or model.pvalues[i] > CONST_SIGNIFICANCE:
                exog[i] = 0

        print(exog)
        ret = model.forecast(exog=exog)
    except:
        pass


    return ret

def __arima__(series, exovar=None, order=(1, 0, 1)):
    ret = None

    try:
        model = ARIMA(series, exog=exovar, order=order)
        ret = model.fit(disp=-1e6, trend='nc')
        #print(ret.summary())
    except Exception as e:
        #print('ERROR: build ARIMA failed! Pass...')
        #print(e)
        pass

    return ret