""" *****************

    AHN module

    File name: moduleAHNv5.py
    Related files: thesis.py
    Created by: Enrique Gonzalez
    Date: 13 mar 2023

    Compilacion:
    Ejecucion:

***************** """

# --------------------
# import libraries
import numpy as np # to use numpy arrays instead of lists
import matplotlib.pyplot as plt # to plot
import seaborn as sns
import pandas as pd # DataFrame (table)
import math
#from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# my libraries
import moduleMNLR

# create saturated compound
# --------------------
def createLinearCompound(n=2):
    """
    Create saturated compound
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    compound = pd.DataFrame(index=range(n))
    compound['omega']=2
    compound.loc[compound.index[0], 'omega'] = compound.loc[compound.index[-1], 'omega'] = 3
    compound['B']=1

    return compound

# compound optimization
# --------------------
def behaviorCompound(sigma=0,centers=0,compound=0,lam=1):
    """
    Compound optimization
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    n = len(compound)
    #func=[("H","reg"), ("F","autoreg"), ("Cl","dt"), ("Br","rf"), ("A",np.cbrt), ("B",np.sinh), ("C",np.cosh), ("D",np.exp2)]
    func=[("H","reg"), ("F","autoreg"), ("A",np.cbrt), ("B",np.sinh), ("C",np.cosh), ("D",np.exp2)]
    funcSize = len(func)
    error=99

    features = sigma.columns.tolist()

    compound2 = pd.DataFrame(columns=['Values','Type',"Cluster",features[0],features[1],features[2],features[3]])
    compound2['Values']= [[] for r in range(n)]

    for i in range(n):
        print("sección: ",i)
        aryl = [0]*funcSize
        list_error = [99]*funcSize
        split_i=sigma[sigma['Cluster']==i]

        for fType in range(funcSize):
            aryl, list_error = behaviorType(split_i, aryl, list_error, fType, func, lam)

        print("entropy list: ", *list_error, sep=", ")
        min_entropy = np.argmin(list_error)  #2
        print("Min entropy cond. by atom: ", min_entropy)
        print("Atom Type: ", func[min_entropy][0])
        print("coefficients: ", aryl[min_entropy])
        print(type(aryl[min_entropy]))
        print("\n")
        compound2.at[i,'Values'] = aryl[min_entropy]     
        compound2.loc[i,'Type'] = func[min_entropy][0]
        compound2.loc[i,"Cluster"] = centers.loc[i,"Cluster"]
        compound2.loc[i,features[0]] = centers.loc[i,features[0]]
        compound2.loc[i,features[1]] = centers.loc[i,features[1]]
        compound2.loc[i,features[2]] = centers.loc[i,features[2]]
        compound2.loc[i,features[3]] = centers.loc[i,features[3]]

    print("modelo")
    [yModel2, regError2]=predict(sigma,compound2,func,0,lam)
    error = regError2['Mean'].values # se cicla si le pido un error muy chico
    print("error modelo:", error)
    print("\n")

    return [regError2, compound2, func]

# # Func type = 1
# --------------------
def behaviorType(split_i=0, aryl=0, list_error=0, fType=0, func=0, lam=1):
        """
        Func type
        """
        pd.set_option("display.max_rows", 10, "display.max_columns", None, "display.precision", 4)
        temp = pd.DataFrame()
        temp['Values'] = [[]]

        aryl[fType] = regPoly(split_i, fType, func, lam)
        #temp.loc[0,'Values'].append(aryl[fType])
        temp.loc[0,'Values'] = aryl[fType]
        temp.loc[0,'Type'] = func[fType][0]
        [yModel, regError]=predict(split_i,temp,func,1,lam)
        #if fType ==0:
        #    pass
        #else:
        list_error[fType] = regError['Mean'].values

        return (aryl, list_error)


# create splits
# --------------------
def splitSigma(sigma=0,num_clusters=2):
    """
    Create data splits
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6

    # get features names from data
    # quito/limpio la columa de cluster porque de lo contrario hay "sesgo"
    #if(num_clusters>2):
    #    sigma=sigma.drop(columns=["Cluster"],axis=1)

    KMestimator = KMeans(n_clusters=num_clusters).fit(sigma)
    dataSet=sigma
    dataSet['Cluster']=KMestimator.labels_
    centers=pd.DataFrame(KMestimator.cluster_centers_)
    centers['Cluster']=KMestimator.predict(KMestimator.cluster_centers_)
    try:
        centers=centers.drop(columns=[4],axis=1)
    except:
        pass
    centers.columns = list(dataSet.columns.values)
    #print(centers)
    #print("\n")

    return [dataSet, centers]

# calculate delta
# --------------------
def deltaValue(sigma=0):
    """
    Calculate delta
    """
    # get features names from data
    features = list(sigma.columns.values)
    # calculate delta value
    features = [features.pop(0)]
    ySigma=sigma.loc[:,features].values
    delta=(ySigma**2).sum()

    return delta

# obtain AHN
# --------------------
def fit(sigma=0, maxMol=400, epsilon=.005, lam=1):
    """
    Create AHN
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6

    # Security Checking
    if sigma.empty:
        raise RuntimeError("sigma must be a data frames: one for the predictors and one for the outcome variables.")
    #if sigma ==0:
    #    print("sigma must be a list with 2 data frames: one for the predictors and one for the outcome variables.")
    #    exit(0)
    if maxMol < 2:
        print("Incorrect value for the max number of molecules n, at least two molecules are required.")
        exit(0)
    if 0 >= epsilon or 1 <= epsilon:
        print("The tolerance epsilon must be between 0 and 1.")
        exit(0)
    if 0 > lam or 1 < lam:
        print("Lambda must be between 0 and 1.")
        exit(0)
 
    error=999
    historic_err = pd.DataFrame(columns=['Num Mol', 'E. Mean'])
    historic_err.loc[0,'Num Mol'] = 0
    historic_err.loc[0, 'E. Mean'] = .1
    historic_err.loc[1,'Num Mol'] = 1
    historic_err.loc[1, 'E. Mean'] = .05
    
    n=2
    
    # OJO NO SE CICLA PORQUE behaviorCompound LLEGA A nmax
    while n <= maxMol and error > epsilon:
        print('n: ',n)
        print("\n")
        
        # split the database
        [mySplits, centers] = splitSigma(sigma,n)
        
        # Create a compound vector
        comp=createLinearCompound(n)
        # compound behavior
        [regError, compound, func]=behaviorCompound(mySplits,centers,comp,lam)
        error = regError['Mean'].values
        
        historic_err.loc[n,'Num Mol'] = n
        historic_err.loc[n, 'E. Mean'] = error

        n += 1
        print("\n")

    plt.figure()
    plt.plot(historic_err['Num Mol'], historic_err['E. Mean'], color='darkorange',label='Relative Error', linestyle='--')
    plt.grid(color='b',linestyle='dashed')
    plt.title('Relative Error Development', fontweight='bold')
    plt.xlabel('Num. of Molecules')
    plt.ylabel('Relative Error')
    plt.legend(loc='upper right')
    plt.show()

    return [compound, func]

def summary(yModel=0):
    """
    AHN Performance summary
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    modPerformance = pd.DataFrame(index=['Model Performance'])
    regError = pd.DataFrame(index=['Relative Error'])

    # compute stats relative error
    regError['Mean'] = yModel['Rel Error'].mean(axis=0)
    regError['Median'] = yModel['Rel Error'].median(axis=0)
    regError['Mode'] = yModel['Rel Error'].mode()
    regError['SD'] = yModel['Rel Error'].std(axis=0)
    regError['MAD'] = yModel['Rel Error'].mad(axis=0)
    regError['Max'] = yModel['Rel Error'].max()
    regError['Min'] = yModel['Rel Error'].min()
    regError['Range'] = regError['Max'] - regError['Min']

    # MNLR model Performance
    modPerformance['RSS'] = (yModel['residuals']**2).sum()
    modPerformance['SSR'] = ((yModel['fitted_mnlr']-yModel['y'].mean(axis=0))**2).sum()
    modPerformance['TSS'] = modPerformance['RSS']+modPerformance['SSR']
    modPerformance['R Square'] = 1-modPerformance['RSS']/modPerformance['TSS']
    print("==========================================================")
    print("                 MODEL AON PERFORMANCE")
    print("----------------------------------------------------------")
    print(modPerformance)
    print("----------------------------------------------------------")
    print(regError)
    print("\n")
    
    return [modPerformance, regError]

# --------------------
# plot AHN
def plotAHN(yModel=0, regError=0):
    """
    Plot AHN compound
    """
    myLabel = 'IPC Mexico Index'
    pLabel = 'AHC'

    plt.figure(figsize=(16, 7))
    plt.plot(yModel['y'], color='navy',label='Raw data')
    plt.plot(yModel['fitted_mnlr'], color='r',label=pLabel,linestyle='--')
    plt.grid(color='b',linestyle='dashed')
    plt.gcf().autofmt_xdate()
    plt.title(myLabel, fontweight='bold')
    plt.xlabel('Date (t)')
    plt.ylabel('IPC Mexico Index (log)')
    plt.legend(loc='lower right')
    plt.show()

    # plot residuals
    totalLabel = pLabel + ' Model Performance'

    plt.figure(figsize=(16, 7))
    plt.plot(yModel['fitted_mnlr'], yModel['residuals'],'o', color='g',label='Residuals')
    plt.grid(color='b',linestyle='dashed')
    plt.axhline(y=0, color='red')
    plt.title(totalLabel, fontweight='bold')
    plt.xlabel('Predictor')
    plt.ylabel('Residuals')
    plt.legend(loc='lower right')
    plt.show()

#    # plot model relative rror
#    totalLabel = pLabel + ' Relative Error Performance'

#    m = regError['Mean'].values

#    plt.figure()
#    plt.plot(yModel['Rel Error'], color='orange',label='Relative Error', linestyle='--')
#    plt.grid(color='b',linestyle='dashed')
#    plt.axhline(y=m, color='r', linestyle='dashed', label='Mean')
#    plt.gcf().autofmt_xdate()
#    plt.title(totalLabel, fontweight='bold')
#    plt.xlabel('Date (t)')
#    plt.ylabel('Relative Error')
#    plt.legend(loc='upper right')
#    plt.show()

#    # plot relative error histogram
#    totalLabel = pLabel + ' Relative Error Histogram'

#    sd = yModel['Rel Error'].std(axis=0)
#    q1 = yModel['Rel Error'].quantile(.25)
#    q3 = yModel['Rel Error'].quantile(.75)

#    plt.figure()
#    plt.hist(yModel['Rel Error'], bins=20, color='navy', edgecolor='k')
#    sns.kdeplot(yModel['Rel Error'], color ='cyan', label='Distribution')
#    plt.axvline(m, color='r', linestyle='dashed', linewidth=1, label='Mean')
#    plt.axvline(m+sd, color='m', linestyle='dashed', linewidth=1, label='Std')
#    plt.axvline(m-sd, color='m', linestyle='dashed', linewidth=1)
#    plt.axvline(q1, color='orange', linestyle='dashed', linewidth=1, label='Q1')
#    plt.axvline(q3, color='orange', linestyle='dashed', linewidth=1, label='Q3')
#    plt.grid(color='b',linestyle='dashed')
#    plt.title(totalLabel, fontweight='bold')
#    plt.xlabel('Value')
#    plt.ylabel('Frequency')
#    plt.legend(loc='upper right')
#    plt.show()

    # plot model relative rror
    totalLabel = pLabel + ' Relative Error'

    m = regError['Mean'].values
    sd = yModel['Rel Error'].std(axis=0)
    q1 = yModel['Rel Error'].quantile(.25)
    q3 = yModel['Rel Error'].quantile(.75)

    fig, axes = plt.subplots(1, 2)
    fig.set_figheight(7)
    fig.set_figwidth(16)
    fig.subplots_adjust(wspace=0.2)

    n, bins, patches = axes[0].hist(yModel['Rel Error'], bins=20, color='orange', edgecolor='k', orientation='horizontal')
    #sns.kdeplot(yModel['Rel Error'], color ='cyan', label='Distribution')
    axes[0].axhline(m, color='r', linestyle='dashed', linewidth=1, label='Mean')
    axes[0].axhline(m+sd, color='m', linestyle='dashed', linewidth=1, label='Std')
    axes[0].axhline(m-sd, color='m', linestyle='dashed', linewidth=1)
    axes[0].axhline(q1, color='cyan', linestyle='dashed', linewidth=1, label='Q1')
    axes[0].axhline(q3, color='cyan', linestyle='dashed', linewidth=1, label='Q3')
    axes[0].axhline(color='b',linestyle='dashed')
    axes[0].grid(color='b',linestyle='dashed')
    axes[0].set_title('Histogram')
    axes[0].set_xlabel('Freq')
    axes[0].set_ylabel('Relative Error')
    axes[0].set_xlim(n.max()+50, 0)
    axes[0].legend(loc='upper left')

    axes[1].plot(yModel['Rel Error'], color='orange',label='Relative Error', linestyle='--')
    axes[1].grid(color='b',linestyle='dashed')
    axes[1].axhline(y=m, color='r', linestyle='dashed', label='Mean')
    axes[1].axhline(m+sd, color='m', linestyle='dashed', label='Std')
    axes[1].axhline(m-sd, color='m', linestyle='dashed')
    axes[1].set_title('Value')
    axes[1].set_xlabel('Date (t)')
    axes[1].legend(loc='upper right')

    fig.suptitle(totalLabel, fontweight='bold')
    plt.show()

    totalLabel = pLabel + ' Relative Error'

    plt.figure()
    medianprops = dict(linestyle='-', linewidth=1.5, color='firebrick')
    plt.boxplot(yModel['Rel Error'], notch=True, medianprops=medianprops)
    plt.title(totalLabel, fontweight='bold')
    plt.show()

# obtain regression for behaviorType
# --------------------
def regPoly(split_i=0,fType=1,func=0, lam=1):
    """
    Obtain regression for behaviorType
    """
    pd.set_option("display.max_rows", 10, "display.max_columns", None, "display.precision", 4)
    sizeData=len(split_i)
    y=0
    x1=1
    x2=2
    x3=3
    x4=4

    #print("Super charros")
    dataSet_AR = pd.DataFrame()
    dataSet_AR["log_index"] = split_i["log_index"]
    dataSet_AR["t-1"] = split_i["log_index"].shift(1,fill_value=split_i.iloc[0,0])
    dataSet_AR["t-2"] = split_i["log_index"].shift(2,fill_value=split_i.iloc[0,0])
    dataSet_AR["P. Comp. 1"] = split_i["P. Comp. 1"]
    dataSet_AR["P. Comp. 2"] = split_i["P. Comp. 2"]
    dataSet_AR = dataSet_AR.fillna(0)
    #print(dataSet_AR)
    #print("\n")

    # MNLR model
    if fType == 0:
        polyCoef = moduleMNLR.fitLAM(split_i, y, x1, x2, x3, lam).loc[:,0].values.tolist()
    elif fType == 1:
        # polyCoef used instead of regr for consistency
        polyCoef = moduleMNLR.fit2(dataSet_AR, y, x1, x2, x3, x4, lam).loc[:,0].values.tolist()
    #e0lif fType == 2:
    #    regr = DecisionTreeRegressor(max_depth=5)
    #    regr.fit(dataSet_AR.iloc[:,x1:x4].to_numpy().reshape(sizeData,3), split_i.iloc[:,y].to_numpy().reshape(sizeData,1))
    #    return regr
    #elif fType == 3:
    #    regr = RandomForestRegressor(random_state=1, n_estimators=10)
    #    regr.fit(dataSet_AR.iloc[:,x1:x4].to_numpy().reshape(sizeData,3), split_i.iloc[:,y].to_numpy().reshape(sizeData,1))
    #    return regr
    else:
        polyCoef = moduleMNLR.fitDin(dataSet_AR, y, x1, x2, x3,func[fType][1]).loc[:,0].values.tolist()

    return polyCoef

# obtain regression/prediction
# --------------------
def predict(sigma=0,compound=0,func=0,case=0,lam=1):
    """
    Obtain regression/prediction
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    regError = pd.DataFrame(index=['Relative Error']) #####
    yModel = pd.DataFrame()

    # get features names from data
    #sigma.columns.tolist())
    features = list(sigma.columns.values)[1]

    # obtain regression/prediction
    if case == 0:
        n = len(compound)
        for j in range(n):

            if len(sigma[sigma['Cluster']==j])>0:

                split_j = sigma[sigma['Cluster']==j]
                ySplit = pd.DataFrame()
                ySplit = predPoly(split_j, ySplit, features, func, compound, j, lam)
                ySplit['mi_index'] = list(sigma[sigma['Cluster']==j].index)
                yModel = pd.concat([yModel,ySplit])

    else:
        ySplit = pd.DataFrame()
        ySplit = predPoly(sigma, ySplit, features, func, compound, 0, lam)
        ySplit['mi_index'] = list(sigma.index)
        yModel = pd.concat([yModel,ySplit])

    yModel = yModel.sort_values(by=['mi_index'])
    yModel[""]=range(len(yModel))
    yModel.set_index([''],inplace = True, drop=True)
    yModel = yModel.rename(columns={'mi_index': 'Date'})
    yModel.set_index(['Date'],inplace = True, drop=True)

    # add column original value
    yModel['y']= sigma[list(sigma.columns.values)[0]]
    cols = yModel.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    yModel = yModel[cols]

    # compute residuals and relative error
    yModel['residuals'] = yModel['y'] - yModel['fitted_mnlr']
    yModel['Rel Error'] = (1-(yModel['fitted_mnlr']/yModel['y'])).abs()
    regError['Mean'] = yModel['Rel Error'].mean(axis=0)

    return [yModel, regError]

def predPoly(split_j=0, ySplit=0, features=0, func=0, compound=0, j=0, lam=1):
    """
    Obtain regression for predict
    """
    pd.set_option("display.max_rows", 10, "display.max_columns", None, "display.precision", 4)
    yModel = pd.DataFrame()
    funcSize = len(func)
    sizeData=len(split_j)
    y=0
    x1=1
    x2=2
    x3=3
    x4=4

    #print("Charros viejos y mal olientes")
    #print(split_j.iloc[0,0])
    dataSet_AR = pd.DataFrame()
    dataSet_AR["log_index"] = split_j["log_index"]
    dataSet_AR["t-1"] = split_j["log_index"].shift(1,fill_value=split_j.iloc[0,0])
    dataSet_AR["t-2"] = split_j["log_index"].shift(2,fill_value=split_j.iloc[0,0])
    dataSet_AR["P. Comp. 1"] = split_j["P. Comp. 1"]
    dataSet_AR["P. Comp. 2"] = split_j["P. Comp. 2"]
    #dataSet_AR = dataSet_AR.fillna(0)
    #print(dataSet_AR)
    #dataSet_AR = dataSet_AR.dropna()
    #print(dataSet_AR)
    #print("\n")

    for fFind in range(funcSize):
        if compound.iloc[j,1] == func[fFind][0]:
            fType = fFind
            break

    if compound.iloc[j,1] == "H":
        # fitted_mnlr reg
        ySplit = moduleMNLR.predictLAM(yModel, pd.DataFrame(compound.iloc[j,0]), split_j, y, x1, x2, x3, lam)
    elif compound.iloc[j,1] == "F":
        # fitted_mnlr autoreg
        ySplit = moduleMNLR.predict2(yModel, pd.DataFrame(compound.iloc[j,0]), dataSet_AR, y, x1, x2, x3, x4, lam)
    #elif compound.iloc[j,1] == "Cl":
    #    regr = compound.iloc[j,0]
    #    # fitted_dt
    #    ySplit['fitted_mnlr'] = regr.predict(dataSet_AR.iloc[:,x1:x4].to_numpy().reshape(sizeData,3))
    #elif compound.iloc[j,1] == "Br":
    #    regr = compound.iloc[j,0]
    #    # fitted_rf
    #    ySplit['fitted_mnlr'] = regr.predict(dataSet_AR.iloc[:,x1:x4].to_numpy().reshape(sizeData,3))
    else:
        ySplit = moduleMNLR.predictDin(yModel, pd.DataFrame(compound.iloc[j,0]), dataSet_AR, y, x1, x2, x3,func[fType][1])

    return ySplit

# assign cluster to testSet
# --------------------
def clusterTest(set1=0, set2=0, mols=0, rand=False):
    """
    Assign cluster to testSet
    """
    pd.set_option("display.max_rows", 10, "display.max_columns", None, "display.precision", 4)

    if rand == True:
        for cluster in range(mols):
            temp = set1[set1['Cluster']==cluster]
            min_date = temp.index[0]
            max_date = temp.index[-1]
            for i in range(len(set2)):
                if set2.index[i] >= min_date and set2.index[i] <= max_date:
                    ind = set2.index[i]
                    set2.loc[ind,'Cluster']=cluster
    else:
        set2['Cluster']=set1.iloc[-1,-1]

    return set2

def clusterTest2(set2=0, compound=0):
    pd.set_option("display.max_rows", 10, "display.max_columns", None, "display.precision", 4)
    dataSet = set2
    dataSet["Cluster"] = 0
    features = set2.columns.tolist()
    compSize = len(compound)
    dist_list = [99]*compSize

    n = len(set2)
    for j in range(n):
        a = (set2.iloc[j,0],set2.iloc[j,1],set2.iloc[j,2],set2.iloc[j,3])
        for i in range(compSize):
            b = (compound.loc[i,features[0]],compound.loc[i,features[1]],compound.loc[i,features[2]],compound.loc[i,features[3]])
            dist_list[i] = math.dist(a,b)
        #print("date: ",j)
        #print("Dist. List: ", dist_list)
        min_dist = np.argmin(dist_list)
        #print("Min. Dist: ", min_dist)
        dataSet.iloc[j,4] = min_dist

    return dataSet
