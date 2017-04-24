import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.ar_model import AR
import statsmodels.api as sm

from utils import tsplot

plt.style.use('ggplot')

def get_data():
    d1 = date(2014, 10, 5)
    d2 = date(2105, 4, 30)
    delta = d2 - d1
    
    m = np.loadtxt('m.csv')
    rng = pd.date_range('2014/10/5', periods=m.size, freq='H')
    
    data = pd.DataFrame(data=m, index=rng)

    sep = int(m.size * 0.8)
    
    train = data[0:sep]
    test = data[sep:-1]
    
    return data, train, test

def infos(ts):
    rol = ts.rolling(100, min_periods=0)
   
    data = ts.values.reshape(-1)
    m = rol.mean().values.reshape(-1)
    s = rol.std().values.reshape(-1)[1:-1]
    x = np.arange(m.size)

    m_lr = np.poly1d(np.polyfit(x,m,1))(x)
    s_lr = np.poly1d(np.polyfit(x[1:-1],s,1))(x[1:-1])
    
    plt.plot(m, label='100 window mean')
    plt.plot(s, label='100 window std')
    plt.plot(m_lr, label='linear fit window mean')
    plt.plot(s_lr, label='linear fit window std')
    plt.plot(data, alpha=0.3)
    plt.legend()
    plt.title('Some simple first analysis on the time serie')
    plt.show()
    
    result = adfuller(data)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    tsplot(data[::10])
    plt.show()
	
def AR_fit(train, test):
    max_lag = 300 
    idx = range(1, max_lag+1)
    
    res_aic = []
    res_bic = []

    for lag in idx:
        model = AR(train.values, train.index)
        fit = model.fit(lag)
        res_aic.append(fit.aic)
        res_bic.append(fit.bic)

    plt.plot(res_aic)
    plt.plot(res_bic)
    plt.show()

def point_pred(fit, ts, mean, std):
    print(mean, std, '-----')
    params = fit.params
    c = params[0]
    filter_ = params[1::]
    K_known = ts[-169::] 
    X = c + (filter_*K_known).sum() + np.random.normal(mean, std)
    return X

def forecast(fit, ts, n_ahead, mean, std):
    ts = train.values.reshape(-1)
    toz = []
    for i in range(n_ahead):
        pred = point_pred(fit, ts, mean, std)
        ts = np.append(ts, pred)
        toz.append(pred) 
    return toz

def bootstrap(fit, train, test):
    alpha = 0.05
    r0 = 25
    R = int(np.ceil(2*r0/alpha) - 1)
    f = []

    val = train.values.reshape(-1)
    te = test.values.reshape(-1)[0:100]
    mean = 0
    std = 1
    
    comp = te

    for i in range(100):
        pred = forecast(fit, val, 100, mean, std)
        f.append(pred)
        res = te - pred
        plt.plot(res)
        plt.show()
        mean = np.mean(res)
        std = np.std(res)
    
    ff = np.array(f)
    ff.sort(axis=0)
    
    low = r0
    high = R+1-r0
    pred_int = ff[low:high,:]
    print(ff[:,0])
    print(len(f)) 
    plt.plot(te)
    for i in range(15):
        plt.plot(f[i], alpha=0.3)
    plt.show()


def AR_model(train, test):
    model = AR(train.values, train.index)
    fit = model.fit(169)
    
    d1 = test.index[0]
    d2 = test.index[-1]
    
    bootstrap(fit, train, test)
    exit()

    predicted = fit.predict(d1, d2)
    predicted = pd.DataFrame(predicted, test.index)

    diff = test - predicted
    
    plt.plot(test)
    plt.plot(predicted)
    plt.show()
    
    plt.plot(diff)
    plt.show()

    plt.hist(diff, bins=100)
    plt.show()
    tsplot(np.reshape(diff.values.T, (-1)))
    plt.show()


def ARMA_fit(ts):
    lag_acf = acf(ts, nlags=20)
    lag_pacf = pacf(ts, nlags=20, method='ols')
    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()

    plt.show()


def ARMA_model(data, train, test):
    model = ARMA(data.values, dates=data.index, order=(2,9))
    fit = model.fit(mxiter=100)

    rng = pd.date_range('2015/4/30 23:00:00', periods=40, freq='H')

    f, stderr, conf = fit.forecast(40, alpha=0.05)
    f_low = pd.DataFrame(conf[:,0], rng)
    f_high = pd.DataFrame(conf[:,1], rng)
    f = pd.DataFrame(f, rng)
    
    plt.title('Prediction using Recurrent Neural Net')
    plt.plot(data[-100::], label='Data given')
    plt.plot(f, label='Prediction')
    plt.fill_between(x=rng, y1=f_low.values.reshape(-1), y2=f_high.values.reshape(-1), 
                     alpha=0.2, color='b', label='Prediction interval')
    plt.legend()
    plt.show()

def SARIMAX_model(train, test):
    model = SARIMAX(train.values, dates=train.index, order=(9,0,9), enforce_stationarity=False,
                    enforce_invertibility=False, seasonal_order=(9,0,0,0))
    fit = model.fit(mxiter=100)

    d1 = test.index[0]
    d2 = test.index[-1]
    
    f = fit.forecast(39)
    f = pd.DataFrame(f, test[0:39].index)
    #predicted = fit.predict(d1, d2)
    #predicted = pd.DataFrame(predicted, test.index)
    
    plt.plot(test[0:39])
    plt.plot(f)
    plt.show()

data, train, test = get_data()

#infos(data)
#AR_fit(train, test)
ARMA_fit(data)
ARMA_model(data, train, test)

