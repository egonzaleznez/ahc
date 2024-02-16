""" *****************

    Data module

    File name: moduleARMA.py
    Related files: thesis.py
    Created by: Enrique Gonzalez
    Date: 5 ene 2023

    Compilacion:
    Ejecucion:

***************** """

import numpy as np # to use numpy arrays instead of lists
import matplotlib.pyplot as plt # to plot
import pandas as pd # DataFrame (table)
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# --------------------
# compute ARIMA model
def arimaModel(dataSet, use_exog=False):
    """
    Computes ARIMA model
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    yModel = pd.DataFrame()

    forecast = pd.DataFrame()
    dataForecast = pd.DataFrame()
    i=0

    performanceHeaders=['RSS','SSR','TSS','R Square','Time']
    errorHeaders=['Mean','Median', 'Mode', 'SD','MAD','Max','Min','Range']
    modPerformance = pd.DataFrame(index=['ARIMA'], columns=performanceHeaders)
    modPerformance.index.name = "Model Performance"
    regError = pd.DataFrame(index=['ARIMA'], columns=errorHeaders)
    regError.index.name = "Relative Error"

    y=0
    x=1
    mvar = dataSet.columns.tolist()[y]

    # ETS Decomposition
    #result = ets(dataSet.iloc[:,y], 252)

    # Autocorrelation test
    q = autocorrelation(dataSet.iloc[:,y])

    # Augmented Dickey-Fuller test
    target = pd.DataFrame(dataSet.iloc[:,y])
    d, vals, result = adf(target)
    del target

    print("=======================================================================")
    print("                      ARIMA MODEL ANALYSIS")
    print("==========================================================")
    print("                        ADF TEST")
    print("----------------------------------------------------------")
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
	       print('\t%s: %.3f' % (key, value))
    print("\n")

    # compute ARIMA model
    p = 2
    if use_exog ==False:
        model = ARIMA(dataSet.iloc[:,y], order=(p,d,q))
    else:
        model = ARIMA(dataSet.iloc[:,y], order=(p,d,q), exog=dataSet.iloc[:,x])

    # compute regression
    result = model.fit()
    yModel[mvar]=result.fittedvalues

    # compute residuals, relative error and model performance
    regModel, regError, modPerformance = compareMethod(yModel, mvar, dataSet, y, regError, modPerformance, 0, float('nan'))

    print("=======================================================================")
    print("                      MODEL PERFORMANCE")
    print("-----------------------------------------------------------------------")
    print("                        Training")
    print("-----------------------------------------------------------------------")
    print(modPerformance)
    print("=======================================================================")
    print("                      RELATIVE ERROR COMPARISON")
    print("-----------------------------------------------------------------------")
    print("                        Training")
    print("-----------------------------------------------------------------------")
    print(regError)

    # show ARIMA summary
    arimaSummary(yModel, result, mvar)

    while d > 0:
        yModel[mvar]=yModel[mvar].cumsum()+vals.loc[i]
        yModel.iloc[d-1,1]=vals.loc[i]
        i+=1
        d-=1

    plt.figure(figsize=(10, 6))
    plt.plot(dataSet.iloc[:,y], color='navy',label='Raw data')
    plt.plot(yModel[mvar], color='r',label='ARIMA Regression',linestyle="--")
    plt.grid(color='b',linestyle='dashed')
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date (t)')
    plt.ylabel(mvar)
    plt.legend(loc='lower right')
    plt.show()

    # compute forecast
    forecast[mvar] = dataSet[mvar]
    temp = pd.DataFrame(columns=['Date',mvar])
    #temp1['Date'] = [datetime.date(yModel.index[-1]+timedelta(days=1))]
    temp['Date'] = [forecast.index[-1]+timedelta(days=1)]

    if use_exog ==False:
        temp[mvar] = result.predict(len(dataSet)+1).values
    else:
        temp[mvar] = result.predict(len(dataSet)+1).values

    temp.set_index(['Date'],inplace=True, drop=True)
    forecast = forecast.append(temp)
    #print(forecast)

    return forecast

# --------------------
# ETS Decomposition (ETS stands for Error, Trend, and Seasonality)
def ets(target, freq):
    """
    Computes ETS Decomposition
    """
    result = seasonal_decompose(target.dropna(), freq=freq) # model ='multiplicative', freq=252)
    result.plot()

    return result

# --------------------
# Autocorrelation and partial autocorrelation test
def autocorrelation(target):
    """
    Computes autocorrelation and partial autocorrelation test
    """
    autocc=pd.DataFrame()

    autocc['acf']= [pd.Series([target.values]).autocorr(lag=lags+1) for lags in range(len(target))]
    q = [autocc.abs().idxmax().values[0] if autocc.abs().max().values != None and autocc.abs().max().values > 0.45 else 0][0]
    #print('q: ', q)
    #print("\n")

    return q

# --------------------
# Augmented Dickey-Fuller test to check if the data is stationary
# The more negative this statistic, the more likely we are to reject the null hypothesis (we have a stationary dataset).
def adf(target):
    vals = pd.Series()
    d=0

    value = criticValue = -1
    while value >= criticValue:
        result = adfuller((target.dropna()).values)
        criticValue = result[4]['5%']  #-1.6
        value = result[0]
        if value >= criticValue:
            vals.index+=1
            vals=pd.concat([pd.Series([target.iloc[d].values[0]]),vals])
            target=target.diff()
            d+=1
            #print('si derive:', d)
    #print("\n")

    return (d, vals, result)

# --------------------
# ARIMA summary method
def arimaSummary(yModel, result, mvar):
    """
    Provide summary of the computed ARIMA model
    """
    print(result.summary())
    print("\n")

    plt.figure()
    plt.plot(yModel[mvar], yModel['residuals_'+mvar], 'o', color='g',label='Residuals')
    plt.grid(color='b',linestyle='dashed')
    plt.axhline(y=0, color='red')
    plt.title("Model Performance", fontweight='bold')
    plt.xlabel('Predictor')
    plt.ylabel('Residuals')
    plt.legend(loc='lower left')
    plt.show()


# --------------------
# create performance tables
def compareMethod(yModel, mvar, dataSet, y, regError, modPerformance, i, name_time):
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    name_res = 'residuals_' + mvar
    name_err = 'Rel Error_' + mvar

    # compute residuals and relative error
    yModel[name_res] = dataSet.iloc[:,y] - yModel[mvar]
    yModel[name_err] = (1-(yModel[mvar]/dataSet.iloc[:,y])).abs()
    regError.iloc[i,0] = yModel[name_err].mean(axis=0)
    regError.iloc[i,1] = yModel[name_err].median(axis=0)
    regError.iloc[i,2] = float('nan') #yModel[name_err].mode()
    regError.iloc[i,3] = yModel[name_err].std(axis=0)
    regError.iloc[i,4] = yModel[name_err].mad(axis=0)
    regError.iloc[i,5] = yModel[name_err].max()
    regError.iloc[i,6] = yModel[name_err].min()
    regError.iloc[i,7] = regError.iloc[i,5] - regError.iloc[i,6]

    # Models Performance
    modPerformance.iloc[i,0] = (yModel[name_res]**2).sum()
    modPerformance.iloc[i,1] = ((yModel[mvar]-dataSet.iloc[:,y].mean(axis=0))**2).sum()
    modPerformance.iloc[i,2] = modPerformance.iloc[i,0]+modPerformance.iloc[i,1]
    modPerformance.iloc[i,3] = 1-modPerformance.iloc[i,0]/modPerformance.iloc[i,2]
    modPerformance.iloc[i,4] = name_time

    return (yModel, regError, modPerformance)
