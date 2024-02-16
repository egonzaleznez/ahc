""" *****************

    Financial Analysis Module

    File name: moduleFA.py
    Related files: thesis.py
    Created by: Enrique Gonzalez
    Date: 5 ene 2021

    Compilacion:
    Ejecucion:

***************** """

import numpy as np # to use numpy arrays instead of lists
import matplotlib.pyplot as plt # to plot
import pandas as pd # DataFrame (table)
from scipy.stats import norm

# --------------------
# compute price return
def priceReturn(dataSet):
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6

    # compute return and log index
    dataFA = pd.DataFrame(dataSet['index'])
    dataFA["PriceDiff"] = dataFA["index"]-dataFA["index"].shift(1)
    dataFA['Return'] = dataFA["PriceDiff"]/dataFA["index"]
    dataFA["Return %"] = dataFA['Return']*100
    dataFA["Direction"] = [1 if dataFA['PriceDiff'].loc[ei] > 0 else 0 for ei in dataFA.index ]
    dataFA["log_index"] = dataSet['log_index']

    #print("----------------------------------------------------------")
    #print(dataFA["Return"])
    #print(dataSet['index'].pct_change())
    #print("\n")

    return dataFA

"""
# --------------------
# compute annualized retrun
def annualReturn(dataFA,dataSet):
    density = pd.DataFrame()

    # chance of losing over 5% in a day?
    prob = 5 # probability set to 5%
    n = len(dataFA)
    mu = dataFA['Return'].mean(axis=0)
    sigma = dataFA['Return'].std(axis=0) # Daily StanDev
    ##print(sigma)
    probLosing = norm.cdf(-prob/100,mu,sigma)
    density['x'] = np.arange(dataFA["Return %"].min()-0.01, dataFA["Return %"].max()+0.01, 0.001)
    density['pdf'] = norm.pdf(density['x'], mu, sigma)

    plt.figure()
    dataFA["Return %"].hist(bins=20, color='navy')
    plt.plot(density['x'], density['pdf'], color='red')
    plt.grid(color='b',linestyle='dashed')
    plt.title("Distribution of the IPC Return", fontweight='bold')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.show()

    # compute annualized retrun and sharpe ratio
    annualizedReturn = (mu+1)**252-1  #255
    volatility = (sigma**2*(n/(n-1)))**0.5 # ~ Daily StanDev
    rf = dataSet.mean(axis=0)/100
    ##print("annualizedReturn: ", annualizedReturn)
    ##print("rf: ", rf)
    ##print("\n")
    sharpeRatio = (annualizedReturn-rf)/sigma
    data= {'Daily Return %':[mu*100],'Volatility %':[volatility],"Daily Prob of Losing >5%":[probLosing*100], 'Yearly Return %':[annualizedReturn*100], 'Yearly Sharpe Ratio':[sharpeRatio]}
    dataFA2 = pd.DataFrame(data,index=['IPC'])
    #print("\n")
    print("==========================================================")
    print("                   FINANCIAL ANALYSIS")
    print("----------------------------------------------------------")
    print(dataFA2)
    print("\n")
"""

# --------------------
# compute retrun
def compoundReturn(dataFA,dataSet):
    n = len(dataFA)
    sigma = dataFA['Return'].std() # daily volatility
    volatility = sigma*(252**0.5) # annualized volatility
    rf = (dataSet.mean()/100+1)**(n/252)-1
    #print(rf)

    return_per_day = (dataFA['Return']+1).prod()**(1/n)-1 # day return
    annualized_return = (return_per_day+1)**252-1 # annual return
    #print((dataFA['Return']+1).prod()-1) # compound return
    #print(return_per_day) # day return
    #print((dataFA['Return']+1).prod()**(252/n)-1) # annual return
    #print((annualized_return+1)**(n/252)-1) # compound return
    #print(annualized_return) # annual return
    #print((dataFA['Return']+1).prod())
    #print(volatility)
    sharpeRatio = ((dataFA['Return']+1).prod()-rf)/volatility
    #print(sharpeRatio)
    #print(     (dataFA['Return']+1).prod()**(1/(n/252))-1     ) # annual return
    #print("\n")

    data = {'Return %':[(annualized_return+1)**(n/252)*100], 'Risk Free Rate %':[rf*100], 'Volatility %':[volatility*100], 'Sharpe Ratio':[sharpeRatio]}
    dataFA2 = pd.DataFrame(data,index=['IPC'])
    print("==========================================================")
    print("                   FINANCIAL ANALYSIS")
    print("----------------------------------------------------------")
    print(dataFA2)
    print("\n")

"""
    compound_return_ipc = ((dataFA['Return']+1).prod()*100).round(2)
    print("\n")
    print(compound_return_ipc)
    print(compound_return_ipc/100)
    print(volatility)
    print((compound_return_ipc/100)/volatility)
    print(  (dataFA['Return']+1).prod()**(1/(n/252))-1 )
    print("\n")
"""

# --------------------
# compute Buy & Hold strategy
def tradeStrat(dataSet,fma,sma):
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6

    fmaLabel = 'FMA'+str(fma)
    smaLabel = 'SMA'+str(sma)
    dataSet[fmaLabel] = dataSet["log_index"].rolling(fma).mean() # fast signal
    dataSet[smaLabel] = dataSet["log_index"].rolling(sma).mean() # slow signal
    dataSet = dataSet.dropna()
    dataSet["UD"] = [1 if dataSet[fmaLabel].loc[ei] >= dataSet[smaLabel].loc[ei] else 0 for ei in dataSet.index ]
    dataSet["Trade"] = ["BUY" if dataSet['UD'].loc[ei] == 1 else "SELL" for ei in dataSet.index ]
    dataSet["Position"] = [1 if dataSet['UD'].loc[ei] == 1 else -1 for ei in dataSet.index ]
    dataSet["wealth"] = dataSet["Return %"].cumsum()
    dataSet.to_csv('trade_strat.csv')

    return [dataSet, fmaLabel, smaLabel]
