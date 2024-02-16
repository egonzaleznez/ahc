""" *****************

    Data module

    File name: moduleData.py
    Related files: thesis.py
    Created by: Enrique Gonzalez
    Date: 5 ene 2021

    Compilacion:
    Ejecucion:

***************** """

import numpy as np # to use numpy arrays instead of lists
import matplotlib.pyplot as plt # to plot
import seaborn as sns
import pandas as pd # DataFrame (table)
from scipy.stats import chi2
import sklearn.decomposition as sk # to compute PCA
from sklearn.model_selection import train_test_split

# my libraries
import moduleARMA

# --------------------
# get data from csv file
def getDataSet():
    """
    Read data from csv file
    """
    dataSet = pd.read_csv('ndx.csv', header=0, index_col=0)
    dataSet.index = pd.to_datetime(dataSet.index,format='%d/%m/%y',errors='ignore')

    return dataSet

# --------------------
# smooth data
def normalizeDataSet(dataSet, smooth=True):
    """
    Normalize and smooth data
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    dataSet = dataSet.fillna(method='ffill')
    dataSet = dataSet.dropna()

    # computing range of data for the regressions
    size = int(len(dataSet))
    x = range(size)
    numFeatures = len(dataSet.columns)-1

    if smooth == True:

        for i in range(numFeatures):
            polyCoef = np.polyfit(x, dataSet.iloc[:,i+1], 10)   # coefficients
            regModel = np.poly1d(polyCoef)              # equation
            regSpace = np.linspace(1, size, size)          # generate n numbers between a range
            dataSet.iloc[:,i+1] = regModel(regSpace)
            dataSet.iloc[:,i+1] = dataSet.iloc[:,i+1]-dataSet.iloc[:,i+1].mean(axis=0)

    dataSet["log_index"] = np.log(dataSet['index'])/np.log(3.5) # math.log(number, base) = np.log(number)/np.log(base) 1.02  100000

    return dataSet


# --------------------
# smooth data
def forecastSet(dataSet, days=3):
    """
    Produce a forecast of n days for each variable,
    based on ARIMA method
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    foreSet = pd.DataFrame()
    numFeatures = len(dataSet.columns)

    for col in range(numFeatures):
        #temp.iloc[:,col] = dataSet.iloc[:,col]
        col_name = dataSet.columns.values[col]
        temp = pd.DataFrame()
        temp[col_name] = dataSet[col_name]
        for i in range(days):
            temp = moduleARMA.arimaModel(temp,False)
        foreSet[col_name] = temp[col_name]

    return foreSet

# --------------------
# print data
def printDataSet(dataSet):
    """
    Print data
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    print("\n")
    print(round(dataSet,6))
    print("\n")

# --------------------
# plot data
def plotData(dataSet,dataSet2):
    y = list(dataSet.columns.values)
    y = y.pop(0)
    x = list(dataSet.columns.values)
    x = x.pop(1)
    y_fitted = list(dataSet2.columns.values)
    y_fitted = y_fitted.pop(0)
    myLabel = 'Model Performance'

    plt.figure()
    plt.plot(dataSet[y], color='navy',label='Raw data')
    plt.plot(dataSet2[y_fitted], color='r',label='Model')
    plt.grid(color='b',linestyle='dashed')
    plt.title(myLabel, fontweight='bold')
    plt.xlabel('Time (t)')
    plt.ylabel('Value')
    plt.legend()
    plt.legend(loc='upper left')
    plt.show()

# --------------------
# Plot original data and its distribution
def plotSet(dataSet):
    """
    Plot original data and its distribution
    """

    n = len(dataSet)
    m = dataSet["log_index"].mean()
    sd = dataSet["log_index"].std(axis=0)

    alpha = 5
    df = n-1
    x2_1 = chi2.ppf(1-(alpha/100/2), df)
    ic_1 = (df*sd/x2_1)**(1/2)
    x2_2 = chi2.ppf((alpha/100/2), df)
    ic_2 = (df*sd/x2_2)**(1/2)
    print(sd)
    print("IC 1: ", ic_1**2)
    print("IC 2: ", ic_2**2)
    print("\n")

    fig, axes = plt.subplots(1, 2)# figsize=(14, 18))
    fig.subplots_adjust(wspace=0.2)

    n, bins, patches = axes[0].hist(dataSet["log_index"], bins=20, color='navy', edgecolor='k', orientation='horizontal')
    axes[0].axhline(m, color='r', linestyle='dashed', label='Mean')
    axes[0].axhline(m+sd, color='m', linestyle='dashed', label='Std')
    axes[0].axhline(m-sd, color='m', linestyle='dashed')
    axes[0].grid(color='b',linestyle='dashed')
    axes[0].set_title('Histogram')
    axes[0].set_xlabel('Freq')
    axes[0].set_ylabel('IPC Mexico Index (log)')
    axes[0].set_xlim(n.max()+50, 0)
    axes[0].legend(loc='lower left')

    #print(n)
    #print(n.max())

    axes[1].plot(dataSet["log_index"],color='navy',label='IPC')
    axes[1].axhline(m, color='r', linestyle='dashed', label='Mean')
    axes[1].axhline(m+sd, color='m', linestyle='dashed', label='Std')
    axes[1].axhline(m-sd, color='m', linestyle='dashed')
    axes[1].grid(color='b',linestyle='dashed')
    #axes[1].gcf().autofmt_xdate()
    axes[1].set_title('Value')
    axes[1].set_xlabel('Date (t)')
    axes[1].legend(loc='lower right')

    fig.suptitle('IPC Mexico Index', fontweight='bold')
    plt.show()

# --------------------
# plot Macro Variables
def plotExoVar(dataSet):
    """
    Plot exogenous macroeconomic variables
    """
    numFeatures = len(dataSet.columns)-2

    plt.figure()
    for i in range(numFeatures):
        plt.plot(dataSet.iloc[:,i+1],label=dataSet.columns[i+1],marker=i+2,markevery=150)
    plt.grid(color='b',linestyle='dashed')
    plt.gcf().autofmt_xdate()
    plt.title('Macroeconomic Variables', fontweight='bold')
    plt.xlabel('Date (t)')
    plt.ylabel('Macroeconomic variables (value)')
    plt.legend(loc='upper left')
    plt.show()

# --------------------
# plot Strategy
def plotStrat(dataSet, fmaLabel, smaLabel):
    myLabel = 'Double Crossover Analysis' # 'Security Performance'
    security = 'IPC Mexico Index (log)'

    plt.figure(figsize=(16, 7))
    plt.plot(dataSet["log_index"],color='navy',label='Raw data')
    plt.plot(dataSet[fmaLabel],color='r',label='FMA')
    plt.plot(dataSet[smaLabel],color='g',label='SMA')
    plt.grid(color='b',linestyle='dashed')
    plt.gcf().autofmt_xdate()
    plt.title(myLabel, fontweight='bold')
    plt.xlabel('Date (t)')
    plt.ylabel(security)
    plt.legend(loc='lower right')
    plt.show()

# --------------------
# split data
def splitDataSet(dataSet, test_size=.2, randSplit=True):
    """
    Split data in train and test sets
    """
    features = dataSet
    #features = features.drop(['index'], axis=1)
    features = features.drop(['log_index'], axis=1)
    target = pd.DataFrame()
    #target['index'] = dataSet['index']
    target['log_index'] = dataSet['log_index']

    train, test, ytrain, ytest = train_test_split(features, target, test_size=test_size, shuffle=randSplit)
    #train['index'] = ytrain['index']
    train['log_index'] = ytrain['log_index']
    cols = train.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    train = train[cols]
    train = train.sort_index()
    #test['index'] = ytest['index']
    test['log_index'] = ytest['log_index']
    test = test[cols]
    test = test.sort_index()

    return [train, test]

# --------------------
# compute PCA
def computePCA(dataSet, pcaprocess=True, num_comp=3):
    """
    Compute PCA
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    features = dataSet
    features = features.drop(['log_index'], axis=1)

    # correlation analysis
    correlation=features.corr(method = 'pearson')
    print("==========================================================")
    print("                 CORRELATION ANALYSIS")
    print("----------------------------------------------------------")
    print(correlation)
    print("\n")
    features = features.drop(['index'], axis=1)

    # principal components analysis
    if pcaprocess == True:
        pca = sk.PCA(n_components=num_comp)
        principalComponents = pca.fit_transform(features)
        comp_names = {'P. Comp. {}'.format(i+1) for i in range(num_comp)}     
        principalComp = pd.DataFrame(data = principalComponents.tolist(),
                      columns = comp_names)
        principalComp.set_index(dataSet.index,inplace = True, drop=True)
        principalComp['log_index'] = dataSet['log_index']
        cols = ['log_index'] + sorted(comp_names)
        principalComp = principalComp[cols]
        print("==========================================================")
        print("                 PCA ANALYSIS")
        print("----------------------------------------------------------")
        print(principalComp)
        print("\n")

        # explained variance ratio
        varRatio = (pca.explained_variance_ratio_)*100
        dic = {'P. Comp. {}'.format(i+1): varRatio[i] for i in range(num_comp)}
        expVar = pd.DataFrame(dic, index=['Exp Var Ratio %'])

        # Principal axes in feature space, representing the directions of maximum variance
        # get the index of the most important feature on EACH component
        # LIST COMPREHENSION HERE
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(num_comp)]
        # get the names
        initial_feature_names = features.columns
        most_important_names = [initial_feature_names[most_important[i]] for i in range(num_comp)]
        # complete the dataframe
        # LIST COMPREHENSION HERE AGAIN
        dic = {'P. Comp. {}'.format(i+1): most_important_names[i] for i in range(num_comp)}
        #df = pd.DataFrame(dic.items()).T
        expVar = pd.concat([expVar, pd.DataFrame(dic, index =['Most Imp. F.'])])
        print(expVar)
        print("\n")

        #features['index'] = dataSet['index']
        #features['log_index'] = dataSet['log_index']
        #cols = features.columns.tolist()
        #cols = cols[-1:] + cols[:-1]
        #features = features[cols]

        return principalComp
    else:
        print("CC with index")
        print(abs(correlation.iloc[:,0]).sort_values(ascending=False).iloc[1:])
        print("\n")

        cols = ['log_index']
        cols.extend(abs(correlation.iloc[:,0]).sort_values(ascending=False).iloc[1:].index.values)
        cols.extend(['index'])
        dataSet = dataSet[cols]

        #features['index'] = dataSet['index']
        #features['log_index'] = dataSet['log_index']
        #cols = features.columns.tolist()
        #cols = cols[-1:] + cols[:-1]
        #features = features[cols]

        return dataSet

# --------------------
# get data from csv file
def get_dataMean():
    """
    Read data from csv file
    """

    dataSet = pd.read_csv('mean.csv', header=0, index_col=0)

    return dataSet

# --------------------
# get data from csv file
def get_dataMedian():
    """
    Read data from csv file
    """

    dataSet = pd.read_csv('median.csv', header=0, index_col=0)

    return dataSet

def analyzeData(dataSet):

    print(dataSet)
    print("\n")

    plt.figure()
    plt.plot(dataSet["5 mol, reg 1"],color='navy',label='5 mol, reg 1')
    plt.plot(dataSet["8 mol, reg 1"],color='r',label='8 mol, reg 1')
    plt.plot(dataSet["5 mol, reg 1e-10"],color='g',label='5 mol, reg 1e-10')
    plt.plot(dataSet["8 mol, reg 1e-10"],color='b',label='8 mol, reg 1e-10')
    plt.grid(color='b',linestyle='dashed')
    plt.title('AON Relative Error', fontweight='bold')
    plt.xlabel('Split Set % Size')
    plt.ylabel('Relative Error')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure(figsize=(14, 7))
    fig = sns.heatmap(dataSet, vmin=.0008, vmax=.35,
                cmap='RdYlGn_r', annot=True, linewidth=1)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=0)#, horizontalalignment='right')
    plt.title('AON Relative Error', fontweight='bold')
    plt.xlabel('Num. of Molecules, Reg. Factor')
    plt.ylabel('Split Set % Size')
    plt.show()

# --------------------
# get data from csv file
def getDataResults(data=''):
    """
    Read data from csv file
    """
    
    file_1 = 'read_aon_analysis_' + data + '.csv'
    dataSet_1 = pd.read_csv(file_1, header=0, index_col=0)
    
    file_2 = 'read_GA_analysis_' + data + '.csv'
    dataSet_2 = pd.read_csv(file_2, header=0, index_col=0)

    return (dataSet_1, dataSet_2)

def analyzeDataTopology(dataSet=0, index=0):
        
    #ts = [5, 8, 10, 15, 20, 25]
    #error = [.0009, .00125]
    #max_mols = [2, 4, 8, 12]
    #lam = [1, .95, 1e-10, 0]
    #setup = 1
    
    #for h in range(len(ts)):
    #    for i in range(len(error)):
    #        for j in range(len(max_mols)):
    #            for k in range(len(lam)):
    #                print("setup: " , setup, ts[h], error[i], max_mols[j], lam[k])
    #                #label = setup
    #                #rint(dataSet[dataSet[string]])
    #                setup += 1
    #print("\n")
    
    table = pd.pivot_table(dataSet, values='ts mean', index=['max mols','lam'],
                       columns=['error', 'ts size'], aggfunc=np.mean)
    
    table2 = pd.pivot_table(dataSet, values=['ts size', 'error', 'max mols', 'lam', 'ts mean'], index=['setup'],
                       aggfunc={'ts size': np.min,
                                'error': np.min,
                                'max mols': np.min,
                                'lam': np.min,
                                'ts mean': np.mean})
    table2['index'] = index
    table2 = table2.reset_index().sort_values(by=['ts mean'], ignore_index=True)
    columns=['index', 'setup', 'ts size', 'error', 'max mols', 'lam', 'ts mean']
    table2 = table2[columns]
     
    data_min = round(dataSet['ts mean'].min(),4)
        
    plt.figure(figsize=(14, 7))
    fig = sns.heatmap(table, vmin=data_min/1.3, vmax=data_min*6.25,
                cmap='RdYlGn_r', annot=False, linewidth=1)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=25)
    plt.title('AHC Relative Error', fontweight='bold')
    plt.show()
    
    return table2

def analyzeDataGA(dataSet=0, index=0):
        
    table = pd.pivot_table(dataSet, values='ts mean', index=['mut prob','gen limit'],
                       columns=['pop size', 'ts size'], aggfunc=np.mean)
    
    table2 = pd.pivot_table(dataSet, values=['ts size','pop size','mut prob', 'gen limit', 'ts mean'], index=['setup'],
                       aggfunc={'ts size': np.min,
                                'pop size': np.min,
                                'mut prob': np.min,
                                'gen limit': np.min,
                                'ts mean': np.mean})
    table2['index'] = index
    table2 = table2.reset_index().sort_values(by=['ts mean'], ignore_index=True)
    columns=['index', 'setup', 'ts size', 'pop size', 'mut prob', 'gen limit', 'ts mean']
    table2 = table2[columns]
    
    data_min = round(dataSet['ts mean'].min(),4)
        
    plt.figure(figsize=(14, 7))
    fig = sns.heatmap(table, vmin=data_min/10, vmax=data_min*150,
                cmap='RdYlGn_r', annot=False, linewidth=1)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=25)
    plt.title('GA Relative Error', fontweight='bold')
    plt.show()
    
    return table2