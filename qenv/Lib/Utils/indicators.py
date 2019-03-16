import math
import numpy as np

# create a differenced series
def difference(dataset, interval=1):
	diff = [float('NaN') for i in range(interval)]
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def dv_binary_updown(price, interval=1):
    dvs = list()
    for p in range(0, len(price)):
        if (p - interval >= 0):
            roe = float(price[p]) / float(price[p - interval])
            if roe > 1:
                dvs.append(1)
            else:
                dvs.append(0)
        else:
            dvs.append(float('NaN'))

    return dvs

def ev_roe(price, interval=1, lag=0, islog=False):
    roes = list()
    for p in range(0, len(price)):
        if (p - interval - lag >= 0):
            roe = float(price[p - lag]) / float(price[p - interval - lag])
            if islog == True:
                roe = math.log(roe)
            else:
                roe = roe - 1

            roes.append(roe)
        else:
            roes.append(float('NaN'))

    # shift by lags
    if lag > 0:
        roes.insert(0, roes.pop(lag - 1))

    return roes

# Last gain count
def ev_lgc(r, period=8, lag=0):
    ret = list()

    for i in range(0, len(r)):
        if i-lag-period < 0:
            ret.append(float('NaN'))
        else:
            x = np.array(r[i-lag-period:i-lag])
            gc = 0
            for j in range(0, len(x)):
                if x[j] > 0:
                    gc = gc + 1

            ret.append(gc)

    return ret

# ma divergence
def ev_div(ma1, ma2, lag=0):
    ret = list()

    if len(ma1) != len(ma2):
        return None

    for i in range(0, len(ma1)):
        if (i - lag) < 0:
            ret.append(float('NaN'))
            continue

        j = i - lag
        if math.isnan(ma1[j]) or math.isnan(ma2[j]):
            ret.append(float('NaN'))
            continue

        ret.append(float(ma1[j] - ma2[j]))

    return ret

# is long candle shadow
def ev_tail(open, high, low, close, direction=1, lag=0):
    ret = list()

    if open is None or high is None or low is None or close is None:
        return None

    for i in range(0, len(open)):
        if (i - lag) < 0:
            ret.append(float('NaN'))
            continue

        j = i - lag
        body_size = 0
        tail_size = 0
        if direction == 1:
            # upper tail
            body_size = abs(open[j] - close[j])
            tail_size = high[j] - max(open[j], close[j])
        else:
            # lower tail
            body_size = abs(open[j] - close[j])
            tail_size = min(open[j], close[j]) - low[j]

        # avoid divide by zero
        if body_size == 0:
            body_size = 1e-9

        # if tail more than double the size of body
        if float(tail_size / body_size) > 2:
            ret.append(1)
        else:
            ret.append(0)

    return ret

# is long candle shadow
def ev_sld(open, high, low, close, direction=1, lag=0):
    ret = list()

    if open is None or high is None or low is None or close is None:
        return None

    for i in range(0, len(open)):
        if (i - lag) < 0:
            ret.append(float('NaN'))
            continue

        j = i - lag

        if direction == 1:
            # green
            body_size = close[j] - open[j]
            if body_size > 0:
                ut = high[j] - close[j]
                lt = open[j] - low[j]

                # avoid divide by zero
                if ut == 0:
                    ut = 10e-9
                if lt == 0:
                    lt = 10e-9

                # if body is 3 times the size of both shadows
                if (body_size / ut) > 3 and (body_size / lt) > 3:
                    ret.append(1)
                else:
                    ret.append(0)
            else:
                ret.append(0)
        else:
            # red
            body_size = open[j] - close[j]
            if body_size > 0:
                ut = high[j] - open[j]
                lt = close[j] - low[j]

                # avoid divide by zero
                if ut == 0:
                    ut = 10e-9
                if lt == 0:
                    lt = 10e-9

                # if body is 3 times the size of both shadows
                if (body_size / ut) > 3 and (body_size / lt) > 3:
                    ret.append(1)
                else:
                    ret.append(0)
            else:
                ret.append(0)

    return ret

# RSI calculator
def ev_rsi(price, period=14, interval=1, lag=0):
    ret = list()
    roes = (ev_roe(price, interval=interval))

    for i in range(0, len(roes)):
        if (i - period) < 0 or math.isnan(roes[i]):
            ret.append(float('NaN'))
            continue

        # compute sum of gain loss
        sum_gain = 0
        sum_loss = 0
        for j in range(0, period):
            r = roes[i - j]
            if r > 0:
                sum_gain = sum_gain + r
            else:
                sum_loss = sum_loss + abs(r)

        # compute average gain loss
        avg_gain = sum_gain / period
        avg_loss = sum_loss / period

        # if average is zero, set rsi to 100
        if avg_loss == 0:
            ret.append(100)
            continue

        # compute relative strength
        rs = avg_gain / avg_loss

        # compute RSI
        rsi = 100 - (100 / (1 + rs))

        # append to list
        ret.append(rsi)

    # shift by lags
    if lag > 0:
        ret.insert(0, ret.pop(lag - 1))

    return ret

# Simple Moving Average Calculator
def ev_sma(price, period=10, lag=0):
    ret_sma = list()
    ret_dev = list()
    ret_bin = list()

    for i in range(0, len(price)):
        start = (i - (period - 1)) - lag
        end = (i + 1) - lag

        if start < 0:
            ret_sma.append(float('NaN'))
            ret_dev.append(float('NaN'))
            ret_bin.append(float('NaN'))
            continue

        closes = price[start:end]
        total = sum(closes)
        sma = total / period

        ret_sma.append(sma)
        ret_dev.append(float(price[end - 1]) - sma)

        if float(price[end - 1]) > sma:
            ret_bin.append(1)
        else:
            ret_bin.append(0)

    return ret_sma, ret_dev, ret_bin

def ev_ema(price, period=10, lag=0):
    ret_ema = list()
    ret_dev = list()
    ret_bin = list()

    k = 2 / (period + 1)
    for i in range(0, len(price)):
        start = (i - (period - 1)) - lag
        end = (i + 1) - lag

        if start < 0:
            ret_ema.append(float('NaN'))
            ret_dev.append(float('NaN'))
            ret_bin.append(float('NaN'))
            continue

        if len(ret_ema) != 0:
            if math.isnan(ret_ema[-1]):
                closes = price[start:end]
                total = sum(closes)
                sma = total / period
                ret_ema.append(sma)
                ret_dev.append(float(price[end - 1]) - sma)
                if float(price[end - 1]) > sma:
                    ret_bin.append(1)
                else:
                    ret_bin.append(0)
            else:
                ema = ((price[end - 1] - ret_ema[-1]) * k) + ret_ema[-1]
                ret_ema.append(ema)
                ret_dev.append(float(price[end - 1]) - ema)
                if float(price[end - 1]) > ema:
                    ret_bin.append(1)
                else:
                    ret_bin.append(0)

    return ret_ema, ret_dev, ret_bin

def ev_macd(price, fast_period=12, slow_period=26, macd_period=9, lag=0):
    ret_macd = list()
    ret_sign = list()
    ret_hist = list()

    fast_ema = ev_ema(price, fast_period, lag)[0]
    slow_ema = ev_ema(price, slow_period, lag)[0]

    for i in range(0, len(fast_ema)):
        f = fast_ema[i]
        s = slow_ema[i]
        if math.isnan(f) or math.isnan(s):
            ret_macd.append(float('NaN'))
        else:
            ret_macd.append(f - s)

    ret_sign = ev_ema(ret_macd, macd_period)[0]
    for i in range(0, len(ret_sign)):
        m = ret_macd[i]
        n = ret_sign[i]
        if math.isnan(m) or math.isnan(n):
            ret_hist.append(float('NaN'))
        else:
            ret_hist.append(m - n)

    return ret_macd, ret_sign, ret_hist