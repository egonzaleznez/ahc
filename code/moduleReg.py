""" *****************

    Regressions module

    File name: moduleReg.py
    Related files: thesis.py
    Created by: Enrique Gonzalez
    Date: 5 ene 2021

    Compilacion:
    Ejecucion:

***************** """

import numpy as np # to use numpy arrays instead of lists
import matplotlib.pyplot as plt # to plot
import pandas as pd # DataFrame (table)
import time as tm
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.kernel_ridge import KernelRige
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
import scipy.fft as ff
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# my libraries
import moduleMNLR
import moduleData

# --------------------
# compare Models
def allModels(dataSet, dataSet2):
    """
    Compares models performance
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    yModel = pd.DataFrame()
    yModel3 = pd.DataFrame()
    yModel3 = yModel3.reindex_like(dataSet2)

    forecast = pd.DataFrame()
    dataForecast = pd.DataFrame()
    i=0

    methodsUsed=['LSP', 'MNLR', 'ARIMA', 'NN MLP', 'DT', 'SVM', 'RF', 'GPR']
    performanceHeaders=['RSS','SSR','TSS','R Square','Time']
    errorHeaders=['Mean','Median', 'Mode', 'SD','MAD','Max','Min','Range']
    modPerformance = pd.DataFrame(index=methodsUsed, columns=performanceHeaders)
    modPerformance.index.name = "Model Performance"
    regError = pd.DataFrame(index=methodsUsed, columns=errorHeaders)
    regError.index.name = "Relative Error"
    modPerformance3 = pd.DataFrame(index=methodsUsed, columns=performanceHeaders)
    modPerformance3.index.name = "Model Performance"
    regError3 = pd.DataFrame(index=methodsUsed, columns=errorHeaders)
    regError3.index.name = "Relative Error"

    sizeData=len(dataSet)
    sizeData2=len(dataSet2)
    y=0
    x1=1
    x2=2
    x3=3

    start = tm.time()
    # MNLR model
    # find regression matrix and independet vector
    polyCoef = moduleMNLR.fit(dataSet, y, x1, x2, x3)
    yModel = moduleMNLR.predict(yModel, polyCoef, dataSet, y, x1, x2, x3)
    mnlrTime = (tm.time() - start)

    # compute residuals, relative error and model performance
    yModel, regError, modPerformance = compareMethod(yModel, 'mnlr', dataSet, y, regError, modPerformance, 1, mnlrTime)

    # test data
    yModel3 = moduleMNLR.predict(yModel3, polyCoef, dataSet2, y, x1, x2, x3)
    yModel3, regError3, modPerformance3 = compareMethod(yModel3, 'mnlr', dataSet2, y, regError3, modPerformance3, 1, float('nan'))

    start = tm.time()
    # ARIMA model
    # Autocorrelation test
    q = autocorrelation(dataSet.iloc[:,y])

    # Augmented Dickey-Fuller test
    target = pd.DataFrame(dataSet['log_index'])
    d, vals, result = adf(target)
    del target

    # compute ARIMA model
    p = 2
    model = ARIMA(dataSet.iloc[:,y], order=(p,d,q), exog=dataSet.iloc[:,1])

    # compute ARIMA autoregression
    result = model.fit()
    yModel['fitted_arima']=result.fittedvalues
    arimaTime = (tm.time() - start)

    # compute residuals, relative error and model performance
    regModel, regError, modPerformance = compareMethod(yModel, 'arima', dataSet, y, regError, modPerformance, 2, arimaTime)

    # test data
    print("Prueba test ARIMA")
    #print(dataSet.iloc[:,y])
    #print(dataSet.index[0])
    #print(dataSet.index[-1])
    #print(dataSet2.iloc[:,y])
    #print(yModel['fitted_arima']) #mean_squared_error(dataSet2.iloc[:,y], yModel['fitted_arima'])
    #print(result.summary())
    #print(mean_squared_error(dataSet2.iloc[:,y], yModel['fitted_arima']))
    #df = pd.read_html(result.summary().tables[1].as_html(),header=0,index_col=0)[0]
    #print(df['coef'].values)
    #print(df['coef'].values[1])
    print("\n")

    #yModel3['fitted_arima']=result.forecast(dataSet2.iloc[:,y])
    #yModel3, regError3, modPerformance3 = compareMethod(yModel3, 'arima', dataSet2, y, regError3, modPerformance3, 2, float('nan'))

    start = tm.time()
    # LSP regression
    # Least squares polynomial fit
    polyCoefLSP = pd.DataFrame(np.polyfit(range(sizeData), dataSet.iloc[:,y].values, 6))   # coefficients
    regModel = np.poly1d(polyCoefLSP.to_numpy().reshape(1,7)[0])              # equation
    yModel['fitted_lsp'] = regModel(range(sizeData))
    lspTime = (tm.time() - start)

    # compute residuals, relative error and model performance
    yModel, regError, modPerformance = compareMethod(yModel, 'lsp', dataSet, y, regError, modPerformance, 0, lspTime)

    # test data
    yModel3['fitted_lsp'] = regModel(range(sizeData2))
    yModel3, regError3, modPerformance3 = compareMethod(yModel3, 'lsp', dataSet2, y, regError3, modPerformance3, 0, float('nan'))

    start = tm.time()
    # NN MLP regression
    # Neural Networks
    #regr = MLPRegressor(tol=1e-3, max_iter=500, random_state=1).fit(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1), dataSet.iloc[:,y].to_numpy().reshape(sizeData,1))
    regr = MLPRegressor(hidden_layer_sizes=64, solver='sgd', max_iter=400).fit(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1), dataSet.iloc[:,y].to_numpy().reshape(sizeData,1))
    yModel['fitted_nn'] = regr.predict(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1))
    nnTime = (tm.time() - start)

    # compute residuals, relative error and model performance
    yModel, regError, modPerformance = compareMethod(yModel, 'nn', dataSet, y, regError, modPerformance, 3, nnTime)

    # test data
    yModel3['fitted_nn'] = regr.predict(dataSet2.iloc[:,x1].to_numpy().reshape(sizeData2,1))
    yModel3, regError3, modPerformance3 = compareMethod(yModel3, 'nn', dataSet2, y, regError3, modPerformance3, 3, float('nan'))

    start = tm.time()
    # DT regression
    # DecisionTreeRegressor
    regr = DecisionTreeRegressor(max_depth=5)
    regr.fit(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1), dataSet.iloc[:,y].to_numpy().reshape(sizeData,1))
    yModel['fitted_dt'] = regr.predict(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1))
    dtTime = (tm.time() - start)

    # compute residuals, relative error and model performance
    yModel, regError, modPerformance = compareMethod(yModel, 'dt', dataSet, y, regError, modPerformance, 4, dtTime)

    # test data
    yModel3['fitted_dt'] = regr.predict(dataSet2.iloc[:,x1].to_numpy().reshape(sizeData2,1))
    yModel3, regError3, modPerformance3 = compareMethod(yModel3, 'dt', dataSet2, y, regError3, modPerformance3, 4, float('nan'))

    start = tm.time()
    # SVM regression
    # support vector machine
    regr = svm.SVR()
    regr.fit(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1), dataSet.iloc[:,y].to_numpy().reshape(sizeData,1))
    yModel['fitted_svm'] = regr.predict(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1))
    svmTime = (tm.time() - start)

    # compute residuals, relative error and model performance
    yModel, regError, modPerformance = compareMethod(yModel, 'svm', dataSet, y, regError, modPerformance, 5, svmTime)

    # test data
    yModel3['fitted_svm'] = regr.predict(dataSet2.iloc[:,x1].to_numpy().reshape(sizeData2,1))
    yModel3, regError3, modPerformance3 = compareMethod(yModel3, 'svm', dataSet2, y, regError3, modPerformance3, 5, float('nan'))

    start = tm.time()
    # RF regression
    # RandomForestRegressor
    regr = RandomForestRegressor(random_state=1, n_estimators=10)
    regr.fit(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1), dataSet.iloc[:,y].to_numpy().reshape(sizeData,1))
    yModel['fitted_rf'] = regr.predict(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1))
    rfTime = (tm.time() - start)

    # compute residuals, relative error and model performance
    yModel, regError, modPerformance = compareMethod(yModel, 'rf', dataSet, y, regError, modPerformance, 6, rfTime)

    # test data
    yModel3['fitted_rf'] = regr.predict(dataSet2.iloc[:,x1].to_numpy().reshape(sizeData2,1))
    yModel3, regError3, modPerformance3 = compareMethod(yModel3, 'rf', dataSet2, y, regError3, modPerformance3, 6, float('nan'))

    start = tm.time()
    # GPR regression
    # Gaussian Process Regressor
    kernel = DotProduct() + WhiteKernel()
    regr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1), dataSet.iloc[:,y].to_numpy().reshape(sizeData,1))
    yModel['fitted_gpr'] = regr.predict(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1))
    gprTime = (tm.time() - start)

    # compute residuals, relative error and model performance
    yModel, regError, modPerformance = compareMethod(yModel, 'gpr', dataSet, y, regError, modPerformance, 7, gprTime)

    # test data
    yModel3['fitted_gpr'] = regr.predict(dataSet2.iloc[:,x1].to_numpy().reshape(sizeData2,1))
    yModel3, regError3, modPerformance3 = compareMethod(yModel3, 'gpr', dataSet2, y, regError3, modPerformance3, 7, float('nan'))

    regError.loc['AVERAGE'] = regError.mean()
    modPerformance.loc['AVERAGE'] = modPerformance.mean()
    regError3.loc['AVERAGE'] = regError3.mean()
    modPerformance3.loc['AVERAGE'] = modPerformance3.mean()

    #print(yModel)
    #print(yModel3)
    #print("\n")

    #modPerformance.style.applymap(highlight_min)
    print("=======================================================================")
    print("                      MODEL PERFORMANCE COMPARISON")
    print("-----------------------------------------------------------------------")
    print("                      Training")
    print("-----------------------------------------------------------------------")
    print(modPerformance)
    print("-----------------------------------------------------------------------")
    print("                      Testing")
    print("-----------------------------------------------------------------------")
    print(modPerformance3)
    print("=======================================================================")
    print("                      RELATIVE ERROR COMPARISON")
    print("-----------------------------------------------------------------------")
    print("                      Training")
    print("-----------------------------------------------------------------------")
    print(regError)
    print("-----------------------------------------------------------------------")
    print("                      Testing")
    print("-----------------------------------------------------------------------")
    print(regError3)
    print("\n")

    plt.figure(figsize=(14, 6))
    plt.plot(dataSet.iloc[:,y], color='navy',label='Raw data',linestyle="--")
    plt.plot(yModel['fitted_lsp'],color='g',label='LSP',linestyle="--")
    plt.plot(yModel['fitted_mnlr'], color='r',label='MNLR',linestyle="--")
    plt.plot(yModel['fitted_arima'], color='m',label='ARIMA',linestyle="--")
    plt.plot(yModel['fitted_dt'],color='orange',label='DT',linestyle="--")
    plt.plot(yModel['fitted_nn'],color='indigo',label='NN MLP',linestyle="--")
    plt.plot(yModel['fitted_svm'],color='brown',label='SVM',linestyle="--")
    plt.plot(yModel['fitted_rf'],color='violet',label='RF',linestyle="-.")
    plt.plot(yModel['fitted_gpr'],color='c',label='GPR',linestyle="--")
    plt.grid(color='b',linestyle='dashed')
    plt.gcf().autofmt_xdate()
    plt.title('IPC Mexico Index Regression', fontweight='bold')
    plt.xlabel('Date (t)')
    plt.ylabel('IPC Mexico index (log)')
    plt.legend(loc='lower right')
    plt.show()

    fig, axes = plt.subplots(8, 1, figsize=(14, 18))
    fig.subplots_adjust(hspace=0.3)#, wspace=-0.2)

    axes[0].plot(dataSet.iloc[:,y], color='navy')#,label='Raw data')
    axes[0].plot(yModel['fitted_lsp'],color='g',label='LSP')
    axes[0].grid(color='b',linestyle='dashed')
    axes[0].set_title('IPC Mexico Index Regression', fontweight='bold')
    axes[0].legend(loc='lower right')

    axes[1].plot(dataSet.iloc[:,y], color='navy')#,label='Raw data')
    axes[1].plot(yModel['fitted_mnlr'], color='r',label='MNLR')
    axes[1].grid(color='b',linestyle='dashed')
    axes[1].legend(loc='lower right')

    axes[2].plot(dataSet.iloc[:,y], color='navy')#,label='Raw data')
    axes[2].plot(yModel['fitted_arima'], color='m',label='ARIMA')
    axes[2].grid(color='b',linestyle='dashed')
    axes[2].legend(loc='lower right')

    axes[3].plot(dataSet.iloc[:,y], color='navy')#,label='Raw data')
    axes[3].plot(yModel['fitted_nn'],color='indigo',label='NN MLP')
    axes[3].grid(color='b',linestyle='dashed')
    axes[3].set_ylabel('IPC Mexico index (log)')
    axes[3].legend(loc='lower right')

    axes[4].plot(dataSet.iloc[:,y], color='navy')#,label='Raw data')
    axes[4].plot(yModel['fitted_dt'],color='orange',label='DT')
    axes[4].grid(color='b',linestyle='dashed')
    axes[4].legend(loc='lower right')

    axes[5].plot(dataSet.iloc[:,y], color='navy')#,label='Raw data')
    axes[5].plot(yModel['fitted_svm'],color='brown',label='SVM')
    axes[5].grid(color='b',linestyle='dashed')
    axes[5].legend(loc='lower right')

    axes[6].plot(dataSet.iloc[:,y], color='navy')#,label='Raw data')
    axes[6].plot(yModel['fitted_rf'],color='violet',label='RF',linestyle="-.")
    axes[6].grid(color='b',linestyle='dashed')
    axes[6].legend(loc='lower right')

    axes[7].plot(dataSet.iloc[:,y], color='navy')#,label='Raw data')
    axes[7].plot(yModel['fitted_gpr'],color='c',label='GPR')
    axes[7].grid(color='b',linestyle='dashed')
    axes[7].legend(loc='lower right')
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date (t)')
    plt.show()

# --------------------
# create performance tables
def compareMethod(yModel, method, dataSet, y, regError, modPerformance, i, name_time):
    pd.set_option("display.max_rows", 10, "display.max_columns", None, "display.precision", 4)
    name_fit = 'fitted_' + method
    name_res = 'residuals_' + method
    name_err = 'Rel Error_' + method

    # compute residuals and relative error
    yModel[name_res] = dataSet.iloc[:,y] - yModel[name_fit]
    yModel[name_err] = (1-(yModel[name_fit]/dataSet.iloc[:,y])).abs()
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
    modPerformance.iloc[i,1] = ((yModel[name_fit]-dataSet.iloc[:,y].mean(axis=0))**2).sum()
    modPerformance.iloc[i,2] = modPerformance.iloc[i,0]+modPerformance.iloc[i,1]
    modPerformance.iloc[i,3] = 1-modPerformance.iloc[i,0]/modPerformance.iloc[i,2]
    modPerformance.iloc[i,4] = name_time

    return (yModel, regError, modPerformance)

"""
# --------------------
# highlight the minimum in a Series yellow.
def highlight_min(s):
    is_min = s == s.min()
    color = ['red' if v else '' for v in is_min]

    return 'color: %s' % color
"""

# --------------------
# compute ARIMA model
def arimaModel(dataSet):
    """
    Computes ARIMA model
    """
    pd.set_option("display.max_rows", 10, "display.max_columns", None, "display.precision", 4)
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
    modPerformance3 = pd.DataFrame(index=['ARIMA'], columns=performanceHeaders)
    modPerformance3.index.name = "Model Performance"
    regError3 = pd.DataFrame(index=['ARIMA'], columns=errorHeaders)
    regError3.index.name = "Relative Error"

    y=0

    # split data
    [trainSet, testSet] = moduleData.splitDataSet(dataSet, test_size=.02, randSplit=False)
    [features, principalComp] = moduleData.preprocessDataSet(trainSet, pcaprocess=True)

    # ETS Decomposition
    result = ets(principalComp.iloc[:,y], 252)

    # Autocorrelation test
    q = autocorrelation(principalComp.iloc[:,y])

    # Augmented Dickey-Fuller test
    target = pd.DataFrame(principalComp['log_index'])
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
    model = ARIMA(principalComp.iloc[:,y], order=(p,d,q), exog=principalComp.iloc[:,1])

    # compute regression
    result = model.fit()
    yModel['fitted_arima']=result.fittedvalues

    # compute forecast
    forecast['fitted_arima'] = result.forecast(len(testSet), exog=testSet.iloc[:,1])#, dynamic=True)
    #print(f"{testSet.index[0]:%Y-%m-%d}")
    forecast.index=testSet.index

    # compute residuals, relative error and model performance
    regModel, regError, modPerformance = compareMethod(yModel, 'arima', principalComp, y, regError, modPerformance, 0, float('nan'))
    yModel3, regError3, modPerformance3 = compareMethod(forecast, 'arima', testSet, y, regError3, modPerformance3, 0, float('nan'))

    print("=======================================================================")
    print("                      MODEL PERFORMANCE")
    print("-----------------------------------------------------------------------")
    print("                        Training")
    print("-----------------------------------------------------------------------")
    print(modPerformance)
    print("-----------------------------------------------------------------------")
    print("                        Testing")
    print("-----------------------------------------------------------------------")
    print(modPerformance3)
    print("=======================================================================")
    print("                      RELATIVE ERROR COMPARISON")
    print("-----------------------------------------------------------------------")
    print("                        Training")
    print("-----------------------------------------------------------------------")
    print(regError)
    print("-----------------------------------------------------------------------")
    print("                        Testing")
    print("-----------------------------------------------------------------------")
    print(regError3)
    print("\n")

    # show ARIMA summary
    arimaSummary(yModel, result)

    while d > 0:
        yModel['fitted_arima']=yModel['fitted_arima'].cumsum()+vals.loc[i]
        yModel.iloc[d-1,1]=vals.loc[i]
        i+=1
        d-=1

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    fig.subplots_adjust(hspace=0.3)#, wspace=-0.2)

    axes[0].plot(dataSet.iloc[:,y], color='navy',label='Raw data')
    axes[0].grid(color='b',linestyle='dashed')
    axes[0].legend(loc='lower right')

    axes[1].plot(principalComp.index, principalComp.iloc[:,y], color='blue',label='Training Set')
    axes[1].plot(yModel['fitted_arima'], color='r',label='ARIMA Regression',linestyle="--")
    axes[1].plot(testSet.index, testSet.iloc[:,y], color='cyan',label='Testing Set')
    axes[1].plot(forecast.index, forecast.iloc[:,y], color='darkred',label='ARIMA Forecast',linestyle="--")
    axes[1].grid(color='b',linestyle='dashed')
    axes[1].legend(loc='lower right')

    plt.gcf().autofmt_xdate()
    plt.xlabel('Date (t)')
    plt.show()

    arimaPlot2(principalComp, yModel, testSet, forecast, y)

    return dataForecast

# --------------------
# forecast exog variables
def arimaXvars(dataSet,forecastPeriod):
    pd.set_option("display.max_rows", 10, "display.max_columns", None, "display.precision", 4)
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

    # ETS Decomposition
    result = ets(dataSet.iloc[:,y], 252)

    # Autocorrelation test
    q = autocorrelation(dataSet.iloc[:,y])

    # Augmented Dickey-Fuller test
    target = pd.DataFrame(dataSet['log_index'])
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
    model = ARIMA(dataSet.iloc[:,y], order=(p,d,q))

    # compute regression
    result = model.fit()
    yModel['fitted_arima']=result.fittedvalues

    # compute residuals, relative error and model performance
    regModel, regError, modPerformance = compareMethod(yModel, 'arima', dataSet, y, regError, modPerformance, 0, float('nan'))

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
    print("\n")

    # show ARIMA summary
    arimaSummary(yModel, result)

    # compute forecast
    startDay = yModel.index[-1] + pd.DateOffset(days=1)
    forecast['log_index_forecast'] = result.forecast(steps=forecastPeriod)
    forecast['Date'] = pd.date_range(startDay,periods=forecastPeriod,freq='D')
    cols = forecast.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    forecast = forecast[cols]
    forecast.set_index(['Date'],inplace = True, drop=True)
    dataForecast['log_index_forecast']=result.fittedvalues
    dataForecast = pd.concat([dataForecast,forecast])
    dataForecast = dataForecast.dropna()
    dataForecast = dataForecast.rename(columns={'log_index_forecast':'log_index'})

    while d > 0:
        yModel['fitted_arima']=yModel['fitted_arima'].cumsum()+vals.loc[i]
        dataForecast['log_index']=dataForecast['log_index'].cumsum()+vals.loc[i]
        yModel.iloc[d-1,1]=vals.loc[i]
        dataForecast.iloc[d-1,0]=vals.loc[i]
        i+=1
        d-=1

    arimaPlot(yModel, forecast, dataSet, y)

    return dataForecast


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
def arimaSummary(yModel, result):
    """
    Provide summary of the computed ARIMA model
    """
    print(result.summary())
    print("\n")

    plt.figure()
    plt.plot(yModel['fitted_arima'], yModel['residuals_arima'], 'o', color='g',label='Residuals')
    plt.grid(color='b',linestyle='dashed')
    plt.axhline(y=0, color='red')
    plt.title("Model Performance", fontweight='bold')
    plt.xlabel('Predictor')
    plt.ylabel('Residuals')
    plt.legend(loc='lower left')
    plt.show()

# --------------------
# ARIMA plot method
def arimaPlot(yModel, forecast, dataSet, y):
    myLabel = 'ARIMA Model'
    security = 'IPC (log)'

    plt.figure()
    plt.plot(dataSet.iloc[:,y], color='navy',label='Raw data')
    plt.plot(yModel['fitted_arima'], color='r',label='ARIMA Regression')
    plt.plot(forecast['log_index_forecast'], color='orange',label='ARIMA Forecast')
    plt.grid(color='b',linestyle='dashed')
    plt.gcf().autofmt_xdate()
    plt.title(myLabel, fontweight='bold')
    plt.xlabel('Date (t)')
    plt.ylabel(security)
    plt.legend(loc='lower right')
    plt.show()

# --------------------
# ARIMA plot method
def arimaPlot2(trainSet, yModel, testSet, forecast, y):
    myLabel = 'ARIMA Model'
    security = 'IPC (log)'

    plt.figure()
    plt.plot(trainSet.index, trainSet.iloc[:,y], color='blue',label='Training Set')
    plt.plot(yModel['fitted_arima'], color='r',label='ARIMA Regression',linestyle="--")
    plt.plot(testSet.index, testSet.iloc[:,y], color='cyan',label='Testing Set')
    plt.plot(forecast.index, forecast.iloc[:,y], color='darkred',label='ARIMA Forecast',linestyle="--")

    plt.grid(color='b',linestyle='dashed')
    plt.gcf().autofmt_xdate()
    plt.title(myLabel, fontweight='bold')
    plt.xlabel('Date (t)')
    plt.ylabel(security)
    plt.legend(loc='lower right')
    plt.show()

def arimaAON(dataSet):
    """
    Computes ARIMA for new AON model
    """
    yModel = dataSet

    return yModel

def otherModels(dataSet, dataSet2):
    """
    Compares other models performance
    """
    pd.set_option("display.max_rows", 10, "display.max_columns", None, "display.precision", 4)
    yModel = pd.DataFrame()
    yModel3 = pd.DataFrame()
    yModel3 = yModel3.reindex_like(dataSet2)

    forecast = pd.DataFrame()
    dataForecast = pd.DataFrame()
    i=0

    methodsUsed=['MNLR', 'Splines']
    performanceHeaders=['RSS','SSR','TSS','R Square','Time']
    errorHeaders=['Mean','Median', 'Mode', 'SD','MAD','Max','Min','Range']
    modPerformance = pd.DataFrame(index=methodsUsed, columns=performanceHeaders)
    modPerformance.index.name = "Model Performance"
    regError = pd.DataFrame(index=methodsUsed, columns=errorHeaders)
    regError.index.name = "Relative Error"
    modPerformance3 = pd.DataFrame(index=methodsUsed, columns=performanceHeaders)
    modPerformance3.index.name = "Model Performance"
    regError3 = pd.DataFrame(index=methodsUsed, columns=errorHeaders)
    regError3.index.name = "Relative Error"

    sizeData=len(dataSet)
    sizeData2=len(dataSet2)
    y=0
    x1=1
    x2=2
    x3=3

    start = tm.time()
    # MNLR model
    # find regression matrix and independet vector
    polyCoef = moduleMNLR.fit(dataSet, y, x1, x2, x3)
    yModel = moduleMNLR.predict(yModel, polyCoef, dataSet, y, x1, x2, x3)
    mnlrTime = (tm.time() - start)

    # compute residuals, relative error and model performance
    yModel, regError, modPerformance = compareMethod(yModel, 'mnlr', dataSet, y, regError, modPerformance, 0, mnlrTime)

    # test data
    yModel3 = moduleMNLR.predict(yModel3, polyCoef, dataSet2, y, x1, x2, x3)
    yModel3, regError3, modPerformance3 = compareMethod(yModel3, 'mnlr', dataSet2, y, regError3, modPerformance3, 0, float('nan'))

    start = tm.time()
    # 4th degree Splines
    #regr = make_pipeline(PolynomialFeatures(4), Ridge(alpha=1e-3))
    regr = make_pipeline(SplineTransformer(degree=3, n_knots=12), Ridge(alpha=1e-3))
    regr.fit(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1), dataSet.iloc[:,y].to_numpy().reshape(sizeData,1))
    yModel['fitted_splines'] = regr.predict(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1))
    splinesTime = (tm.time() - start)

    # compute residuals, relative error and model performance
    yModel, regError, modPerformance = compareMethod(yModel, 'splines', dataSet, y, regError, modPerformance, 1, splinesTime)

    # test data
    #supertemp = dataSet2.iloc[:,x1].to_numpy().reshape(sizeData2,1)
    yModel3['fitted_splines'] = regr.predict(dataSet2.iloc[:,x1].to_numpy().reshape(sizeData2,1))
    yModel3, regError3, modPerformance3 = compareMethod(yModel3, 'splines', dataSet2, y, regError3, modPerformance3, 1, float('nan'))

    start = tm.time()
    # Inverse Fast Fourier Transform
    yModel['fitted_ifftn'] = ff.ifftn(dataSet.iloc[:,x1].to_numpy().reshape(sizeData,1))
    ifftnTime = (tm.time() - start)

    # compute residuals, relative error and model performance
    #yModel, regError, modPerformance = compareMethod(yModel, 'ifftn', dataSet, y, regError, modPerformance, 2, ifftnTime)

    # test data
    yModel3['fitted_ifftn'] = ff.ifftn(dataSet2.iloc[:,x1].to_numpy().reshape(sizeData2,1))
    #yModel3, regError3, modPerformance3 = compareMethod(yModel3, 'ifftn', dataSet2, y, regError3, modPerformance3, 2, float('nan'))



    #aqui otros métodos


    #modPerformance.style.applymap(highlight_min)
    print("=======================================================================")
    print("                      MODEL PERFORMANCE COMPARISON")
    print("-----------------------------------------------------------------------")
    print("                      Training")
    print("-----------------------------------------------------------------------")
    print(modPerformance)
    print("-----------------------------------------------------------------------")
    print("                      Testing")
    print("-----------------------------------------------------------------------")
    print(modPerformance3)
    print("=======================================================================")
    print("                      RELATIVE ERROR COMPARISON")
    print("-----------------------------------------------------------------------")
    print("                      Training")
    print("-----------------------------------------------------------------------")
    print(regError)
    print("-----------------------------------------------------------------------")
    print("                      Testing")
    print("-----------------------------------------------------------------------")
    print(regError3)
    print("\n")

    plt.figure(figsize=(14, 6))
    plt.plot(dataSet.iloc[:,y], color='navy',label='Raw data',linestyle="--")
    plt.plot(yModel['fitted_mnlr'], color='r',label='MNLR',linestyle="--")
    plt.plot(yModel['fitted_splines'], color='violet',label='Splines',linestyle="--")
    plt.plot(yModel['fitted_ifftn'],color='brown',label='IFFT',linestyle="--")
    plt.grid(color='b',linestyle='dashed')
    plt.gcf().autofmt_xdate()
    plt.title('IPC Mexico Index Regression', fontweight='bold')
    plt.xlabel('Date (t)')
    plt.ylabel('IPC Mexico index (log)')
    plt.legend(loc='lower right')
    plt.show()

    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    fig.subplots_adjust(hspace=0.3)#, wspace=-0.2)

    axes[0].plot(dataSet.iloc[:,y], color='navy')#,label='Raw data')
    axes[0].plot(yModel['fitted_mnlr'], color='r',label='MNLR')
    axes[0].grid(color='b',linestyle='dashed')
    axes[0].set_title('IPC Mexico Index Regression', fontweight='bold')
    axes[0].legend(loc='lower right')

    axes[1].plot(dataSet.iloc[:,y], color='navy')#,label='Raw data')
    axes[1].plot(yModel['fitted_splines'], color='violet',label='Splines')
    axes[1].grid(color='b',linestyle='dashed')
    axes[1].set_ylabel('IPC Mexico index (log)')
    axes[1].legend(loc='lower right')

    axes[2].plot(dataSet.iloc[:,y], color='navy')#,label='Raw data')
    axes[2].plot(yModel['fitted_ifftn'], color='brown',label='IFFT')
    axes[2].grid(color='b',linestyle='dashed')
    axes[2].legend(loc='lower right')

    plt.gcf().autofmt_xdate()
    plt.xlabel('Date (t)')
    plt.show()
