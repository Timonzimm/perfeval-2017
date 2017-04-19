import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA


def get_data():
    d1 = date(2014, 10, 5)
    d2 = date(2105, 4, 30)
    delta = d2 - d1
    
    m = np.loadtxt('m.csv')
    rng = pd.date_range('2014/10/5', periods=m.size, freq='H')
    
    data = pd.DataFrame(data=m, index=rng)

    return data

def stationarity_test(ts):
    rol = ts.rolling(window=12, center=False, min_periods=1) 
    
    ax = ts.plot()
    rol.mean().plot(ax=ax)
    rol.std().plot(ax=ax)



    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    '''print('Results of Dickey-Fuller Test')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statisric', 'p-value', '#Lags Used', '#Obs'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)'''

#stationarity_test(get_data())

def corr(ts):
    lag_acf = acf(ts, nlags=20)
    lag_pacf = pacf(ts, nlags=20, method='ols')
    
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()

    plt.show()


def AR(ts):
    print(ts)

    model = ARIMA(ts, order=(10,0,0))
    results_AR = model.fit(disp=-1)
    plt.plot(ts[::100])
    plt.plot(results_AR.fittedvalues[::100], color='red')
    plt.title('RSS')
    plt.show()
    

    t = results_AR.predict(ts.index[0], ts.index[50])

    print(t)
    plt.plot(t)
    plt.plot(ts[0:50])
    plt.show()

#corr(get_data())

AR(get_data())

