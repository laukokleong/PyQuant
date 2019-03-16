import warnings
warnings.simplefilter('ignore', category=FutureWarning)

import math
import datetime
import csv
import numpy as np
import pandas as pd
import multiprocessing as mp
import Lib.Utils.common as common
import Lib.Utils.indicators as indicators
import Lib.Models.box_jenkins as bj
import Lib.Models.volatility as vol
import Lib.Models.logistic as logit
import Lib.Models.svm as vm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from arch.univariate import arch_model
from sklearn.linear_model import LogisticRegression
from sklearn import svm


is_backtest = False
suffix_weekly = '10080'
suffix_daily = '1440'
datafile_suffix = suffix_daily

currency_pair = 'GBPUSD'
path_raw = 'C:\\Users\\jerry\\Documents\\Projects\\Research\\Data\\' + currency_pair + '.csv'
path_variables = 'C:\\Users\\jerry\\Documents\\Projects\\Research\\Data\\temp.csv'
path_results = 'C:\\Users\\jerry\\Documents\\Projects\\Research\\Data\\' + currency_pair + '_btest.csv'
path_forecast = 'C:\\Users\\jerry\\Documents\\Projects\\Research\\Data\\forecast.csv'

res_df = pd.DataFrame(
        columns=['DATE', 'CLOSE', 'VAR_DLOGRET1', 'ARIMA_DLOGRET1', 'VAR_LOGRET1', 'ARIMA_LOGRET1', 'VAR_SMA10_BIN_L1',
                 'GARCH_VOLATILITY', 'GARCH_1STDDEV', 'GARCH_2STDDEV', 'LOGIT_PROB1', 'SVM_PROB1'])

CONST_MAX_P = 8
CONST_MAX_Q = 8

def Main():
    # load dataset

    # display first few rows
    #print(series['close'])
    # line plot of dataset
    #series.plot()
    #pyplot.show()

    if is_backtest == False:
        now = datetime.datetime.now()
        str_date = now.strftime('%Y.%m.%d')
        str_time = now.strftime('%H:%M')

        forecasts = {}
        forecasts['DATE'] = str_date
        forecasts['TIME'] = str_time
        fx_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD']

        for fxp in fx_pairs:
            fx_filepath = 'C:\\Users\\jerry\\Documents\\Projects\\Research\\Data\\' + fxp + datafile_suffix + '.csv'
            print(fx_filepath)

            raw = pd.read_csv(fx_filepath, header=0)
            raw_len = len(raw)

            # add dummy current forecast period entry
            raw.loc[raw_len] = [str_date,str_time,raw.loc[raw_len - 1]['open'],raw.loc[raw_len - 1]['high'], \
                       raw.loc[raw_len - 1]['low'],raw.loc[raw_len - 1]['close'],999]
            print(raw)

            data = VariableCalculator(raw, path_variables)

            # forecast t=500
            dat_len = len(data)
            fc = BackTest(data, dat_len - 501, dat_len, 1, '')

            forecasts[fxp] = fc

        print('------------- FORECAST RESULTS --------------')
        print(forecasts)
        print('\n')

        # save to file
        fc_data = pd.read_csv(path_forecast, header=0)
        fc_data.loc[len(fc_data)] = [str_date,
                                     str_time,
                                     'W' if datafile_suffix == suffix_weekly else 'D',
                                     forecasts['EURUSD'],
                                     forecasts['GBPUSD'],
                                     forecasts['USDJPY'],
                                     forecasts['USDCAD'],
                                     forecasts['AUDUSD']

        ]
        fc_data.to_csv(path_forecast, index=False)
    else:
        raw = pd.read_csv(path_raw, header=0)

        # back test
        print('******** BACK TEST *********')
        BackTest(data, 499, 1000, 9999, path_results)


def BackTest(data=None, start_index=0, end_index=1000, steps=10, resultpath=''):
    warnings.filterwarnings('ignore')

    if data is None:
        return

    series_size = len(data) #len(np.array(data['VAR_DLOGRET1']))
    #if end_index > series_size:
    #    return

    # create result data frame
    #res_df = pd.DataFrame(
    #    columns=['DATE', 'CLOSE', 'VAR_DLOGRET1', 'ARIMA_DLOGRET1', 'VAR_LOGRET1', 'ARIMA_LOGRET1', 'VAR_SMA10_BIN_L1',
    #             'GARCH_VOLATILITY', 'GARCH_1STDDEV', 'GARCH_2STDDEV', 'LOGIT_PROB1', 'SVM_PROB1'])

    #steps = min(steps, series_size - end_index)
    print(steps)    # debug

    for step in range(0, steps):
        # Build Model (full: ARIMA, GARCH and LOGIT)
        final_arima = None
        final_garch11 = None
        final_logit = None
        final_svm = None

        # -------- INITIALIZE VARIABLES --------
        # ARIMA variables
        arima_dts = 'VAR_DLOGRET1'
        arima_exog_train = [#'VAR_SMA10_BIN_L1', 'VAR_SMA20_BIN_L1', 'VAR_SMA50_BIN_L1', \
                            'VAR_SMA20_DEV_L1', #'VAR_SMA20_DEV_L1', #'VAR_SMA50_DEV_L1', \
                            #'VAR_EMA10_BIN_L1', 'VAR_EMA20_BIN_L1', 'VAR_EMA50_BIN_L1', \
                            #'VAR_EMA10_DEV_L1', 'VAR_EMA20_DEV_L1', 'VAR_EMA50_DEV_L1', \
                            #'VAR_SMA10_20_DIV_L1', 'VAR_SMA20_50_DIV_L1', \
                            #'VAR_EMA10_20_DIV_L1', 'VAR_EMA20_50_DIV_L1', \
                            'VAR_RSI14_L1', 'VAR_MACDSIGN_L1', 'VAR_MACDHIST_L1', \
                            #'VAR_IS_UPPER_TAIL_L1', 'VAR_IS_LOWER_TAIL_L1', \
                            #'VAR_IS_SOLID_GREEN_L1', 'VAR_IS_SOLID_RED_L1' \
                            ]
        arima_exog_fcast = [#'VAR_SMA10_BIN', 'VAR_SMA20_BIN', 'VAR_SMA50_BIN', \
                            'VAR_SMA20_DEV', #'VAR_SMA20_DEV', #'VAR_SMA50_DEV', \
                            #'VAR_EMA10_BIN', 'VAR_EMA20_BIN', 'VAR_EMA50_BIN', \
                            #'VAR_EMA10_DEV', 'VAR_EMA20_DEV', 'VAR_EMA50_DEV', \
                            #'VAR_SMA10_20_DIV', 'VAR_SMA20_50_DIV', \
                            #'VAR_EMA10_20_DIV', 'VAR_EMA20_50_DIV', \
                            'VAR_RSI14', 'VAR_MACDSIGN', 'VAR_MACDHIST', \
                            #'VAR_IS_UPPER_TAIL', 'VAR_IS_LOWER_TAIL', \
                            #'VAR_IS_SOLID_GREEN', 'VAR_IS_SOLID_RED' \
                            ]

        # Logit/SVM variables
        logit_dv = 'VAR_ISGAIN1'
        #logit_ev_train = ['VAR_RET1_L1','VAR_LOGRET1_L1','VAR_RSI14_L1','VAR_MACDSIGN_L1','VAR_MACDHIST_L1']
        #logit_ev_fcast = ['VAR_RET1', 'VAR_LOGRET1', 'VAR_RSI14', 'VAR_MACDSIGN', 'VAR_MACDHIST']
        logit_ev_train = ['VAR_RET1_L1', 'VAR_RET1_L2', 'VAR_RET1_L3', 'VAR_RET1_L4', 'VAR_RET1_L5', \
                          'VAR_RET1_L6', 'VAR_RET1_L7', 'VAR_RET1_L8', 'VAR_RET1_L9', 'VAR_RET1_L10', \
                          'VAR_LGC10_L1', 'VAR_RET4_L1', 'VAR_RET12_L1']#, \
                          #'VAR_SMA10_DEV_L1', 'VAR_SMA20_DEV_L1', 'VAR_SMA50_DEV_L1', 'VAR_SMA100_DEV_L1', \
                          #'VAR_SMA10_BIN_L1', 'VAR_SMA20_BIN_L1', 'VAR_SMA50_BIN_L1', 'VAR_SMA100_BIN_L1', \
                          #'VAR_EMA10_DEV_L1', 'VAR_EMA20_DEV_L1', 'VAR_EMA50_DEV_L1', 'VAR_EMA100_DEV_L1', \
                          #'VAR_EMA10_BIN_L1', 'VAR_EMA20_BIN_L1', 'VAR_EMA50_BIN_L1', 'VAR_EMA100_BIN_L1', \
                          #'VAR_RSI14_L1', 'VAR_MACDSIGN_L1', 'VAR_MACDHIST_L1']
        logit_ev_fcast = ['VAR_RET1', 'VAR_RET1_L1', 'VAR_RET1_L2', 'VAR_RET1_L3', 'VAR_RET1_L4', \
                          'VAR_RET1_L5', 'VAR_RET1_L6', 'VAR_RET1_L7', 'VAR_RET1_L8', 'VAR_RET1_L9', \
                          'VAR_LGC10', 'VAR_RET4', 'VAR_RET12']#, \
                          #'VAR_SMA10_DEV', 'VAR_SMA20_DEV', 'VAR_SMA50_DEV', 'VAR_SMA100_DEV', \
                          #'VAR_SMA10_BIN', 'VAR_SMA20_BIN', 'VAR_SMA50_BIN', 'VAR_SMA100_BIN', \
                          #'VAR_EMA10_DEV', 'VAR_EMA20_DEV', 'VAR_EMA50_DEV', 'VAR_EMA100_DEV', \
                          #'VAR_EMA10_BIN', 'VAR_EMA20_BIN', 'VAR_EMA50_BIN', 'VAR_EMA100_BIN', \
                          #'VAR_RSI14', 'VAR_MACDSIGN', 'VAR_MACDHIST']


        # ------- BUILD MODEL --------
        start = start_index + step #step
        end = end_index + step - 1

        final_model = BuildModel(data, arima_dts, arima_exog_train, logit_dv, logit_ev_train, start, end)

        if final_model[0] is not None:
            final_arima = final_model[0]

        if final_model[1] is not None:
            final_garch11 = final_model[1]

        if final_model[2] is not None:
            final_logit = final_model[2]

        if final_model[3] is not None:
            final_svm = final_model[3]

        print('---- DEBUG: model -----')
        print(final_arima.summary())
        #print(final_arima.arparams)
        #print(final_arima.maparams)
        #print(final_arima.pvalues)
        print('\n')

        # -------- FORECAST ---------
        # all models must be valid
        if final_arima != None and final_garch11 != None and final_logit != None and final_svm != None:
            # one step ahead forecasting
            fc_exog = data.loc[end, arima_exog_fcast]
            old_fcast = final_arima.forecast(exog=fc_exog)[0]
            arima_forecast = bj.forecast(final_arima, exog=fc_exog)[0]
            garch_forecast = final_garch11.forecast()
            logit_forecast = final_logit.predict_proba(data[logit_ev_fcast])
            svm_forecast = final_svm.predict_proba(data[logit_ev_fcast])

            print('----- DEBUG: forecast -----')
            print(np.array(data['date'])[end])
            print('A: ' + str(np.array(data['VAR_LOGRET1'])[end]))
            #print('F_o: ' + str(old_fcast))
            print('F: ' + str(indicators.inverse_difference(np.array(data['VAR_LOGRET1'])[:end], arima_forecast[0])))
            #print(np.array(logit_forecast[-1]))
            #print(np.array(svm_forecast[-1]))
            print('\n')

            # forecast result
            fc_dret = arima_forecast[0]
            fc_volatility = np.array(garch_forecast.variance['h.1'])[-1]

            row = []
            row.append(np.array(data['date'])[end])
            row.append(np.array(data['close'])[end])
            row.append(np.array(data['VAR_DLOGRET1'])[end])
            row.append(fc_dret)
            row.append(np.array(data['VAR_LOGRET1'])[end])
            row.append(indicators.inverse_difference(np.array(data['VAR_LOGRET1'])[:end], fc_dret))
            row.append(np.array(data['VAR_SMA10_BIN_L1'])[end])
            row.append(fc_volatility)
            row.append(math.sqrt(fc_volatility))
            row.append(2 * math.sqrt(fc_volatility))
            row.append(np.array(logit_forecast[-1])[1])
            row.append(np.array(svm_forecast[-1])[1])

            # append to result data frame
            res_df.loc[len(res_df)] = row

    if resultpath != '':
        res_df.to_csv(resultpath)

    return indicators.inverse_difference(np.array(data['VAR_LOGRET1'])[:end], fc_dret)

def VariableCalculator(data, path=''):
    if data is None:
        return None

    # raw price
    open = data['open']
    high = data['high']
    low = data['low']
    close = data['close']

    # up down binary
    data['VAR_ISGAIN1'] = pd.Series(indicators.dv_binary_updown(close, 1))

    # current period return
    data['VAR_RET1'] = pd.Series(indicators.ev_roe(close, 1))
    data['VAR_RET4'] = pd.Series(indicators.ev_roe(close, 4))
    data['VAR_RET12'] = pd.Series(indicators.ev_roe(close, 12))
    data['VAR_LOGRET1'] = pd.Series(indicators.ev_roe(close, interval=1, islog=True))
    data['VAR_LOGRET4'] = pd.Series(indicators.ev_roe(close, interval=4, islog=True))
    data['VAR_LOGRET12'] = pd.Series(indicators.ev_roe(close, interval=12, islog=True))
    data['VAR_DLOGRET1'] = pd.Series(indicators.difference(data['VAR_LOGRET1']))
    data['VAR_DLOGRET4'] = pd.Series(indicators.difference(data['VAR_LOGRET4']))
    data['VAR_DLOGRET12'] = pd.Series(indicators.difference(data['VAR_LOGRET12']))

    # last periods gain count
    data['VAR_LGC10'] = pd.Series(indicators.ev_lgc(data['VAR_RET1'], period=10, lag=0))

    # SMA
    sma10 = indicators.ev_sma(close, period=10, lag=0)
    sma20 = indicators.ev_sma(close, period=20, lag=0)
    sma50 = indicators.ev_sma(close, period=50, lag=0)
    sma100 = indicators.ev_sma(close, period=100, lag=0)

    data['VAR_SMA10'] = pd.Series(sma10[0])
    data['VAR_SMA10_DEV'] = pd.Series(sma10[1])
    data['VAR_SMA10_BIN'] = pd.Series(sma10[2])
    data['VAR_SMA20'] = pd.Series(sma20[0])
    data['VAR_SMA20_DEV'] = pd.Series(sma20[1])
    data['VAR_SMA20_BIN'] = pd.Series(sma20[2])
    data['VAR_SMA50'] = pd.Series(sma50[0])
    data['VAR_SMA50_DEV'] = pd.Series(sma50[1])
    data['VAR_SMA50_BIN'] = pd.Series(sma50[2])
    data['VAR_SMA100'] = pd.Series(sma100[0])
    data['VAR_SMA100_DEV'] = pd.Series(sma100[1])
    data['VAR_SMA100_BIN'] = pd.Series(sma100[2])

    # SMA divergence
    data['VAR_SMA10_20_DIV'] = indicators.ev_div(data['VAR_SMA10'], data['VAR_SMA20'], lag=0)
    data['VAR_SMA20_50_DIV'] = indicators.ev_div(data['VAR_SMA20'], data['VAR_SMA50'], lag=0)
    data['VAR_SMA50_100_DIV'] = indicators.ev_div(data['VAR_SMA50'], data['VAR_SMA100'], lag=0)

    # EMA
    ema10 = indicators.ev_ema(close, period=10, lag=0)
    ema20 = indicators.ev_ema(close, period=20, lag=0)
    ema50 = indicators.ev_ema(close, period=50, lag=0)
    ema100 = indicators.ev_ema(close, period=100, lag=0)

    data['VAR_EMA10'] = pd.Series(ema10[0])
    data['VAR_EMA10_DEV'] = pd.Series(ema10[1])
    data['VAR_EMA10_BIN'] = pd.Series(ema10[2])
    data['VAR_EMA20'] = pd.Series(ema20[0])
    data['VAR_EMA20_DEV'] = pd.Series(ema20[1])
    data['VAR_EMA20_BIN'] = pd.Series(ema20[2])
    data['VAR_EMA50'] = pd.Series(ema50[0])
    data['VAR_EMA50_DEV'] = pd.Series(ema50[1])
    data['VAR_EMA50_BIN'] = pd.Series(ema50[2])
    data['VAR_EMA100'] = pd.Series(ema100[0])
    data['VAR_EMA100_DEV'] = pd.Series(ema100[1])
    data['VAR_EMA100_BIN'] = pd.Series(ema100[2])

    # EMA divergence
    data['VAR_EMA10_20_DIV'] = indicators.ev_div(data['VAR_EMA10'], data['VAR_EMA20'], lag=0)
    data['VAR_EMA20_50_DIV'] = indicators.ev_div(data['VAR_EMA20'], data['VAR_EMA50'], lag=0)
    data['VAR_EMA50_100_DIV'] = indicators.ev_div(data['VAR_EMA50'], data['VAR_EMA100'], lag=0)

    # oscillator
    data['VAR_RSI14'] = pd.Series(indicators.ev_rsi(close, period=14, lag=0))
    data['VAR_MACDSIGN'] = pd.Series(indicators.ev_macd(close, lag=0)[1])
    data['VAR_MACDHIST'] = pd.Series(indicators.ev_macd(close, lag=0)[2])

    # candle pattern
    data['VAR_IS_UPPER_TAIL'] = pd.Series(indicators.ev_tail(open, high, low, close, direction=1, lag=0))
    data['VAR_IS_LOWER_TAIL'] = pd.Series(indicators.ev_tail(open, high, low, close, direction=-1, lag=0))
    data['VAR_IS_SOLID_GREEN'] = pd.Series(indicators.ev_sld(open, high, low, close, direction=1, lag=0))
    data['VAR_IS_SOLID_RED'] = pd.Series(indicators.ev_sld(open, high, low, close, direction=-1, lag=0))

    # lagged return
    data['VAR_RET1_L1'] = pd.Series(indicators.ev_roe(close, interval=1, lag=1))
    data['VAR_RET1_L2'] = pd.Series(indicators.ev_roe(close, interval=1, lag=2))
    data['VAR_RET1_L3'] = pd.Series(indicators.ev_roe(close, interval=1, lag=3))
    data['VAR_RET1_L4'] = pd.Series(indicators.ev_roe(close, interval=1, lag=4))
    data['VAR_RET1_L5'] = pd.Series(indicators.ev_roe(close, interval=1, lag=5))
    data['VAR_RET1_L6'] = pd.Series(indicators.ev_roe(close, interval=1, lag=6))
    data['VAR_RET1_L7'] = pd.Series(indicators.ev_roe(close, interval=1, lag=7))
    data['VAR_RET1_L8'] = pd.Series(indicators.ev_roe(close, interval=1, lag=8))
    data['VAR_RET1_L9'] = pd.Series(indicators.ev_roe(close, interval=1, lag=9))
    data['VAR_RET1_L10'] = pd.Series(indicators.ev_roe(close, interval=1, lag=10))
    data['VAR_RET4_L1'] = pd.Series(indicators.ev_roe(close, interval=4, lag=1))
    data['VAR_RET12_L1'] = pd.Series(indicators.ev_roe(close, interval=12, lag=1))
    data['VAR_LOGRET1_L1'] = pd.Series(indicators.ev_roe(close, interval=1, lag=1, islog=True))
    data['VAR_LOGRET4_L1'] = pd.Series(indicators.ev_roe(close, interval=4, lag=1, islog=True))
    data['VAR_LOGRET12_L1'] = pd.Series(indicators.ev_roe(close, interval=12, lag=1, islog=True))
    data['VAR_DLOGRET1_L1'] = pd.Series(indicators.difference(data['VAR_LOGRET1_L1']))
    data['VAR_DLOGRET4_L1'] = pd.Series(indicators.difference(data['VAR_LOGRET4_L1']))
    data['VAR_DLOGRET12_L1'] = pd.Series(indicators.difference(data['VAR_LOGRET12_L1']))

    # lagged last periods gain count
    data['VAR_LGC10_L1'] = pd.Series(indicators.ev_lgc(data['VAR_RET1'], period=10, lag=1))

    # lagged SMA
    sma10_l1 = indicators.ev_sma(close, period=10, lag=1)
    sma20_l1 = indicators.ev_sma(close, period=20, lag=1)
    sma50_l1 = indicators.ev_sma(close, period=50, lag=1)
    sma100_l1 = indicators.ev_sma(close, period=100, lag=1)

    data['VAR_SMA10_L1'] = pd.Series(sma10_l1[0])
    data['VAR_SMA10_DEV_L1'] = pd.Series(sma10_l1[1])
    data['VAR_SMA10_BIN_L1'] = pd.Series(sma10_l1[2])
    data['VAR_SMA20_L1'] = pd.Series(sma20_l1[0])
    data['VAR_SMA20_DEV_L1'] = pd.Series(sma20_l1[1])
    data['VAR_SMA20_BIN_L1'] = pd.Series(sma20_l1[2])
    data['VAR_SMA50_L1'] = pd.Series(sma50_l1[0])
    data['VAR_SMA50_DEV_L1'] = pd.Series(sma50_l1[1])
    data['VAR_SMA50_BIN_L1'] = pd.Series(sma50_l1[2])
    data['VAR_SMA100_L1'] = pd.Series(sma100_l1[0])
    data['VAR_SMA100_DEV_L1'] = pd.Series(sma100_l1[1])
    data['VAR_SMA100_BIN_L1'] = pd.Series(sma100_l1[2])

    # lagged SMA divergence
    data['VAR_SMA10_20_DIV_L1'] = indicators.ev_div(data['VAR_SMA10_L1'], data['VAR_SMA20_L1'], lag=1)
    data['VAR_SMA20_50_DIV_L1'] = indicators.ev_div(data['VAR_SMA20_L1'], data['VAR_SMA50_L1'], lag=1)
    data['VAR_SMA50_100_DIV_L1'] = indicators.ev_div(data['VAR_SMA50_L1'], data['VAR_SMA100_L1'], lag=1)

    # lagged EMA
    ema10_l1 = indicators.ev_ema(close, period=10, lag=1)
    ema20_l1 = indicators.ev_ema(close, period=20, lag=1)
    ema50_l1 = indicators.ev_ema(close, period=50, lag=1)
    ema100_l1 = indicators.ev_ema(close, period=100, lag=1)

    data['VAR_EMA10_L1'] = pd.Series(ema10_l1[0])
    data['VAR_EMA10_DEV_L1'] = pd.Series(ema10_l1[1])
    data['VAR_EMA10_BIN_L1'] = pd.Series(ema10_l1[2])
    data['VAR_EMA20_L1'] = pd.Series(ema20_l1[0])
    data['VAR_EMA20_DEV_L1'] = pd.Series(ema20_l1[1])
    data['VAR_EMA20_BIN_L1'] = pd.Series(ema20_l1[2])
    data['VAR_EMA50_L1'] = pd.Series(ema50_l1[0])
    data['VAR_EMA50_DEV_L1'] = pd.Series(ema50_l1[1])
    data['VAR_EMA50_BIN_L1'] = pd.Series(ema50_l1[2])
    data['VAR_EMA100_L1'] = pd.Series(ema100_l1[0])
    data['VAR_EMA100_DEV_L1'] = pd.Series(ema100_l1[1])
    data['VAR_EMA100_BIN_L1'] = pd.Series(ema100_l1[2])

    # lagged SMA divergence
    data['VAR_EMA10_20_DIV_L1'] = indicators.ev_div(data['VAR_EMA10_L1'], data['VAR_EMA20_L1'], lag=1)
    data['VAR_EMA20_50_DIV_L1'] = indicators.ev_div(data['VAR_EMA20_L1'], data['VAR_EMA50_L1'], lag=1)
    data['VAR_EMA50_100_DIV_L1'] = indicators.ev_div(data['VAR_EMA50_L1'], data['VAR_EMA100_L1'], lag=1)

    #lagged oscillator
    data['VAR_RSI14_L1'] = pd.Series(indicators.ev_rsi(close, period=14, lag=1))
    data['VAR_MACDSIGN_L1'] = pd.Series(indicators.ev_macd(close, lag=1)[1])
    data['VAR_MACDHIST_L1'] = pd.Series(indicators.ev_macd(close, lag=1)[2])

    # lagged candle pattern
    data['VAR_IS_UPPER_TAIL_L1'] = pd.Series(indicators.ev_tail(open, high, low, close, direction=1, lag=1))
    data['VAR_IS_LOWER_TAIL_L1'] = pd.Series(indicators.ev_tail(open, high, low, close, direction=-1, lag=1))
    data['VAR_IS_SOLID_GREEN_L1'] = pd.Series(indicators.ev_sld(open, high, low, close, direction=1, lag=1))
    data['VAR_IS_SOLID_RED_L1'] = pd.Series(indicators.ev_sld(open, high, low, close, direction=-1, lag=1))

    # trim NaN
    data = data[np.isfinite(data['VAR_SMA100_L1'])]

    # save to file
    if path != '':
        data.to_csv(path)

    return data

def BuildModel(data, arima_dts, arima_exog, logit_dv_col, logit_ev_cols, start, end):
    if data is None:
        return None

    # create time series from column for arima
    dts = np.array(data[arima_dts])[start:end]

    # create exogenous from columns for arima
    exovars=None
    if arima_exog is not None:
        exovars = np.array(data[arima_exog])[start:end]

    # create dependent variable for logit
    y = data[logit_dv_col][start:end]

    # create explanatory variable for logit
    x = data[logit_ev_cols][start:end]

    # ARIMA(p,q)
    # Parallel map
    #setup = dts, exovars, CONST_MAX_P, CONST_MAX_Q
    #po = mp.Pool(processes=2)
    #ret_arima = po.map(build_arima_wrapper, setup)
    #print(len(ret_arima))
    #po.close()
    ret_arima = bj.build_arima(dts, exovar=exovars, max_p=CONST_MAX_P, max_q=CONST_MAX_Q)

    # GARCH(1,1)
    ret_garch11 = vol.build_garch11(dts)

    # LOGIT(y,x)
    ret_logit = logit.build_logit(x, y)

    # SVM
    ret_svm = vm.build_svm(x, y)


    return ret_arima, ret_garch11, ret_logit, ret_svm


def build_arima_wrapper(args):
    bj.build_arima(args[0], exovar=args[1], max_p=args[2], max_q=args[3])

def prepare_fx_data_file():
    pairs_to_fix = ['EURUSD','GBPUSD','USDJPY','USDCAD','AUDUSD']

    for p in pairs_to_fix:
        path_to_fix = 'C:\\Users\\jerry\\Documents\\Projects\\Research\\Data\\' + p + datafile_suffix + '.csv'
        common.insert_csv_header(path_to_fix)

# start work
print(datetime.datetime.time(datetime.datetime.now()))

prepare_fx_data_file()
Main()

print(datetime.datetime.time(datetime.datetime.now()))

#res_df.to_csv(path_results)

def temp():
    end_index = 400
    closef = list()
    for i in range(0,1):
        train_set = dlog[i:end_index]
        #print(train_set)
        print(len(train_set))
        # fit model
        model = ARIMA(train_set, order=(3, 0, 1))
        model_fit = model.fit(disp=0)
        # print summary of fit model
        print(model_fit.summary())

        forecast = model_fit.forecast()[0]
        print(forecast)

        val = inverse_difference(train_set, forecast) + .35
        closef.append(val)

        end_index += 1

    actual = list(close[401:411])

    print(actual)
    print(closef)

    plt.plot(closef)
    plt.plot(actual)
    plt.ylabel('forecast')
    plt.show()