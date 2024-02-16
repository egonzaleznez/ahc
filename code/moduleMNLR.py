""" *****************

    MNLR module

    File name: moduleMNLR.py
    Related files: thesis.py
    Created by: Enrique Gonzalez
    Date: 5 mar 2021

    Compilacion:
    Ejecucion:

***************** """

# --------------------
# import libraries
import numpy as np # to use numpy arrays instead of lists
import matplotlib.pyplot as plt # to plot
import pandas as pd # DataFrame (table)
from scipy.stats import norm

# --------------------
# compute MNLR model with exogenous variables
def mnlrReg(dataSet, dataSet2, num_comp=3):
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    yModel = pd.DataFrame()
    yModel2 = pd.DataFrame()
    yModel_test = pd.DataFrame()
    y=0
    x1=1
    x2=2
    x3=3
    x4=4

    if num_comp == 1:
        polyCoef = fit1PC(dataSet, y, x1)
        yModel = predict1PC(yModel, polyCoef, dataSet, y, x1)
        yModel_test = predict1PC(yModel_test, polyCoef, dataSet2, y, x1)
    elif num_comp == 2:
        polyCoef = fit2PC(dataSet, y, x1, x2)
        yModel['fitted_mnlr'] = polyCoef.iloc[5,0]*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]+polyCoef.iloc[4,0]*dataSet.iloc[:,x2]**2+polyCoef.iloc[3,0]*dataSet.iloc[:,x1]**2+polyCoef.iloc[2,0]*dataSet.iloc[:,x2]+polyCoef.iloc[1,0]*dataSet.iloc[:,x1]+polyCoef.iloc[0,0]
        yModel_test['fitted_mnlr'] = polyCoef.iloc[5,0]*dataSet2.iloc[:,x1]*dataSet2.iloc[:,x2]+polyCoef.iloc[4,0]*dataSet2.iloc[:,x2]**2+polyCoef.iloc[3,0]*dataSet2.iloc[:,x1]**2+polyCoef.iloc[2,0]*dataSet2.iloc[:,x2]+polyCoef.iloc[1,0]*dataSet2.iloc[:,x1]+polyCoef.iloc[0,0]
    else:
        #func=[np.exp, np.sin, np.sinh, np.sinc, np.arcsinh, np.cbrt, np.cos, np.cosh, np.arctan, np.exp2]
        func=[np.real]
        for ftype in range(len(func)):
            #polyCoef = fitDin(dataSet, y, x1, x2, x3, func[ftype])
            #yModel = predictDin(yModel, polyCoef, dataSet, y, x1, x2, x3, func[ftype])
            #yModel_test = predictDin(yModel_test, polyCoef, dataSet2, y, x1, x2, x3, func[ftype])

            polyCoef = fit(dataSet, y, x1, x2, x3)
            yModel = predict(yModel, polyCoef, dataSet, y, x1, x2, x3)
            #yModel_test = predict(yModel_test, polyCoef, dataSet2, y, x1, x2, x3)

            print("Charros viejos y mal olientes")
            dataSet_AR = pd.DataFrame()
            dataSet_AR["log_index"] = dataSet["log_index"]
            dataSet_AR["t-1"] = dataSet["log_index"].shift(1)
            dataSet_AR["t-2"] = dataSet["log_index"].shift(2)
            dataSet_AR["P. Comp. 1"] = dataSet["P. Comp. 1"]
            dataSet_AR["P. Comp. 2"] = dataSet["P. Comp. 2"]
            dataSet_AR = dataSet_AR.fillna(0)
            print(dataSet_AR)
            print("\n")
            polyCoef2 = fit2(dataSet_AR, y, x1, x2, x3, x4, 1)
            yModel2 = predict2(yModel2, polyCoef2, dataSet_AR, y, x1, x2, x3, x4, 1)

            polyCoef3 = fitLAM(dataSet, y, x1, x2, x3, 1e-10)
            yModel_test = predictLAM(yModel_test, polyCoef3, dataSet, y, x1, x2, x3, 1e-10)

    # MNLR summary
            print("Training Set Results")
            summary(yModel, polyCoef, dataSet, y)
            print("\n")
            summary(yModel2, polyCoef2, dataSet_AR, y)
            print("\n")
            print("Test Set Results")
            summary(yModel_test, polyCoef3, dataSet, y)
            print("\n")

    # LSP regression
    # Least squares polynomial fit
            yModel = myLSP(yModel, dataSet, y)
            yModel2 = myLSP(yModel2, dataSet_AR, y)
            yModel_test = myLSP(yModel_test, dataSet, y)

    # MNLR Plot
            mnlrPlot(yModel, dataSet, y)
            mnlrPlot(yModel2, dataSet_AR, y)
            mnlrPlot(yModel_test, dataSet, y)

# --------------------
# MNLR fit method with 2 variables
def fit2PC(dataSet, y, x1, x2):
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    matrixReg = pd.DataFrame()
    vectorReg = pd.DataFrame()
    polyCoef = pd.DataFrame()
    sizeData=len(dataSet)

    # find regression matrix and independet vector
    matrixReg.loc[0,0] = len(dataSet)
    matrixReg.loc[1,0] = matrixReg.loc[0,1] = dataSet.iloc[:,x1].sum()
    matrixReg.loc[2,0] = matrixReg.loc[0,2] = dataSet.iloc[:,x2].sum()
    matrixReg.loc[3,0] = matrixReg.loc[0,3] = matrixReg.loc[1,1] = (dataSet.iloc[:,x1]**2).sum()
    matrixReg.loc[4,0] = matrixReg.loc[0,4] = matrixReg.loc[2,2] = (dataSet.iloc[:,x2]**2).sum()
    matrixReg.loc[5,0] = matrixReg.loc[0,5] = matrixReg.loc[2,1] = matrixReg.loc[1,2] = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()
    matrixReg.loc[3,1] = matrixReg.loc[1,3] = (dataSet.iloc[:,x1]**3).sum()
    matrixReg.loc[4,1] = matrixReg.loc[1,4] = matrixReg.loc[5,2] = matrixReg.loc[2,5] = (dataSet.iloc[:,x1]*(dataSet.iloc[:,x2])**2).sum()
    matrixReg.loc[5,1] = matrixReg.loc[1,5] = matrixReg.loc[3,2] = matrixReg.loc[2,3] = ((dataSet.iloc[:,x1]**2)*dataSet.iloc[:,x2]).sum()
    matrixReg.loc[4,2] = matrixReg.loc[2,4] = (dataSet.iloc[:,x2]**3).sum()
    matrixReg.loc[4,3] = matrixReg.loc[3,4] = matrixReg.loc[5,5] = ((dataSet.iloc[:,x1]**2)*(dataSet.iloc[:,x2]**2)).sum()
    matrixReg.loc[3,3] = (dataSet.iloc[:,x1]**4).sum()
    matrixReg.loc[5,3] = matrixReg.loc[3,5] = ((dataSet.iloc[:,x1]**3)*dataSet.iloc[:,x2]).sum()
    matrixReg.loc[4,4] = (dataSet.iloc[:,x2]**4).sum()
    matrixReg.loc[5,4] = matrixReg.loc[4,5] = (dataSet.iloc[:,x1]*(dataSet.iloc[:,x2]**3)).sum()
    vectorReg.loc[0,0] = dataSet.iloc[:,y].sum()
    vectorReg.loc[1,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]).sum()
    vectorReg.loc[2,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x2]).sum()
    vectorReg.loc[3,0] = (dataSet.iloc[:,y]*(dataSet.iloc[:,x1])**2).sum()
    vectorReg.loc[4,0] = (dataSet.iloc[:,y]*(dataSet.iloc[:,x2])**2).sum()
    vectorReg.loc[5,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()

    # find inverse matrix
    invMatrix = pd.DataFrame(np.linalg.inv(matrixReg))

    # find solution to system
    polyCoef = invMatrix.dot(vectorReg)

    return polyCoef

# --------------------
# MNLR fit method with 1 variable1
def fit1PC(dataSet, y, x1):
    matrixReg = pd.DataFrame()
    vectorReg = pd.DataFrame()
    polyCoef = pd.DataFrame()
    sizeData=len(dataSet)

    # find regression matrix and independet vector
    matrixReg.loc[0,0] = len(dataSet)
    matrixReg.loc[1,0] = matrixReg.loc[0,1] = dataSet.iloc[:,x1].sum()
    matrixReg.loc[2,0] = matrixReg.loc[0,2] = matrixReg.loc[1,1] = (dataSet.iloc[:,x1]**2).sum()
    matrixReg.loc[2,1] = matrixReg.loc[1,2] = (dataSet.iloc[:,x1]**3).sum()
    matrixReg.loc[2,2] = (dataSet.iloc[:,x1]**4).sum()

    vectorReg.loc[0,0] = dataSet.iloc[:,y].sum()
    vectorReg.loc[1,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]).sum()
    vectorReg.loc[2,0] = (dataSet.iloc[:,y]*(dataSet.iloc[:,x1]**2)).sum()

    # find inverse matrix
    invMatrix = pd.DataFrame(np.linalg.inv(matrixReg))

    # find solution to system
    polyCoef = invMatrix.dot(vectorReg)

    return polyCoef

# --------------------
# MNLR fit method with 3 variables
def fit(dataSet, y, x1, x2, x3):
    matrixReg = pd.DataFrame()
    vectorReg = pd.DataFrame()
    polyCoef = pd.DataFrame()
    sizeData=len(dataSet)

    # find regression matrix and independet vector
    sumX1 = dataSet.iloc[:,x1].sum()
    sumX2 = dataSet.iloc[:,x2].sum()
    sumX3 = dataSet.iloc[:,x3].sum()

    sumX12 = (dataSet.iloc[:,x1]**2).sum()
    sumX22 = (dataSet.iloc[:,x2]**2).sum()
    sumX32 = (dataSet.iloc[:,x3]**2).sum()

    sumX13 = (dataSet.iloc[:,x1]**3).sum()
    sumX23 = (dataSet.iloc[:,x2]**3).sum()
    sumX33 = (dataSet.iloc[:,x3]**3).sum()

    sumX14 = (dataSet.iloc[:,x1]**4).sum()
    sumX24 = (dataSet.iloc[:,x2]**4).sum()
    sumX34 = (dataSet.iloc[:,x3]**4).sum()

    sumX1X2 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()
    sumX1X3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x3]).sum()
    sumX2X3 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX1X2X3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()

    sumX12X2 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]).sum()
    sumX12X3 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x3]).sum()
    sumX22X3 = (dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]).sum()

    sumX1X22 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2).sum()
    sumX1X32 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**2).sum()
    sumX2X32 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()

    sumX12X22 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]**2).sum()
    sumX12X32 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x3]**2).sum()
    sumX22X32 = (dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]**2).sum()

    sumX13X2 = (dataSet.iloc[:,x1]**3*dataSet.iloc[:,x2]).sum()
    sumX13X3 = (dataSet.iloc[:,x1]**3*dataSet.iloc[:,x3]).sum()
    sumX23X3 = (dataSet.iloc[:,x2]**3*dataSet.iloc[:,x3]).sum()

    sumX1X23 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**3).sum()
    sumX1X33 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**3).sum()
    sumX2X33 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**3).sum()

    sumX12X2X3 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX1X22X3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]).sum()
    sumX1X2X32 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()

    #sumX12X22X3 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]).sum()
    #sumX12X2X32 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()
    #sumX1X22X32 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]**2).sum()
    #sumX12X22X32 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]**2).sum()

    #sumX13X2X3 = (dataSet.iloc[:,x1]**3*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    #sumX1X23X3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**3*dataSet.iloc[:,x3]).sum()
    #sumX1X2X33 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**3).sum()

    matrixReg.loc[0,0] = sizeData
    matrixReg.loc[0,1] = sumX1
    matrixReg.loc[0,2] = sumX2
    matrixReg.loc[0,3] = sumX3
    matrixReg.loc[0,4] = sumX12
    matrixReg.loc[0,5] = sumX22
    matrixReg.loc[0,6] = sumX32
    matrixReg.loc[0,7] = sumX1X2
    matrixReg.loc[0,8] = sumX1X3
    matrixReg.loc[0,9] = sumX2X3
    #matrixReg.loc[0,10] = sumX1X2X3

    matrixReg.loc[1,0] = sumX1
    matrixReg.loc[1,1] = sumX12
    matrixReg.loc[1,2] = sumX1X2
    matrixReg.loc[1,3] = sumX1X3
    matrixReg.loc[1,4] = sumX13
    matrixReg.loc[1,5] = sumX1X22
    matrixReg.loc[1,6] = sumX1X32
    matrixReg.loc[1,7] = sumX12X2
    matrixReg.loc[1,8] = sumX12X3
    matrixReg.loc[1,9] = sumX1X2X3
    #matrixReg.loc[1,10] = sumX12X2X3

    matrixReg.loc[2,0] = sumX2
    matrixReg.loc[2,1] = sumX1X2
    matrixReg.loc[2,2] = sumX22
    matrixReg.loc[2,3] = sumX2X3
    matrixReg.loc[2,4] = sumX12X2
    matrixReg.loc[2,5] = sumX23
    matrixReg.loc[2,6] = sumX2X32
    matrixReg.loc[2,7] = sumX1X22
    matrixReg.loc[2,8] = sumX1X2X3
    matrixReg.loc[2,9] = sumX22X3
    #matrixReg.loc[2,10] = sumX1X22X3

    matrixReg.loc[3,0] = sumX3
    matrixReg.loc[3,1] = sumX1X3
    matrixReg.loc[3,2] = sumX2X3
    matrixReg.loc[3,3] = sumX32
    matrixReg.loc[3,4] = sumX12X3
    matrixReg.loc[3,5] = sumX22X3
    matrixReg.loc[3,6] = sumX33
    matrixReg.loc[3,7] = sumX1X2
    matrixReg.loc[3,8] = sumX1X32
    matrixReg.loc[3,9] = sumX2X32
    #matrixReg.loc[3,10] = sumX1X2X32

    matrixReg.loc[4,0] = sumX12
    matrixReg.loc[4,1] = sumX13
    matrixReg.loc[4,2] = sumX12X2
    matrixReg.loc[4,3] = sumX12X3
    matrixReg.loc[4,4] = sumX14
    matrixReg.loc[4,5] = sumX12X22
    matrixReg.loc[4,6] = sumX12X32
    matrixReg.loc[4,7] = sumX13X2
    matrixReg.loc[4,8] = sumX13X3
    matrixReg.loc[4,9] = sumX12X2X3
    #matrixReg.loc[4,10] = sumX13X2X3

    matrixReg.loc[5,0] = sumX22
    matrixReg.loc[5,1] = sumX1X22
    matrixReg.loc[5,2] = sumX23
    matrixReg.loc[5,3] = sumX22X3
    matrixReg.loc[5,4] = sumX12X22
    matrixReg.loc[5,5] = sumX24
    matrixReg.loc[5,6] = sumX22X32
    matrixReg.loc[5,7] = sumX1X23
    matrixReg.loc[5,8] = sumX1X22X3
    matrixReg.loc[5,9] = sumX23X3
    #matrixReg.loc[5,10] = sumX1X2X3

    matrixReg.loc[6,0] = sumX32
    matrixReg.loc[6,1] = sumX1X32
    matrixReg.loc[6,2] = sumX2X32
    matrixReg.loc[6,3] = sumX33
    matrixReg.loc[6,4] = sumX12X32
    matrixReg.loc[6,5] = sumX22X32
    matrixReg.loc[6,6] = sumX34
    matrixReg.loc[6,7] = sumX1X2X32
    matrixReg.loc[6,8] = sumX1X33
    matrixReg.loc[6,9] = sumX2X33
    #matrixReg.loc[6,10] = sumX1X2X33

    matrixReg.loc[7,0] = sumX1X2
    matrixReg.loc[7,1] = sumX12X2
    matrixReg.loc[7,2] = sumX1X22
    matrixReg.loc[7,3] = sumX1X2X3
    matrixReg.loc[7,4] = sumX13X2
    matrixReg.loc[7,5] = sumX1X23
    matrixReg.loc[7,6] = sumX1X2X32
    matrixReg.loc[7,7] = sumX12X22
    matrixReg.loc[7,8] = sumX12X2X3
    matrixReg.loc[7,9] = sumX1X22X3
    #matrixReg.loc[7,10] = sumX12X22X3

    matrixReg.loc[8,0] = sumX1X3
    matrixReg.loc[8,1] = sumX12X3
    matrixReg.loc[8,2] = sumX1X2X3
    matrixReg.loc[8,3] = sumX1X32
    matrixReg.loc[8,4] = sumX13X3
    matrixReg.loc[8,5] = sumX1X22X3
    matrixReg.loc[8,6] = sumX1X33
    matrixReg.loc[8,7] = sumX12X2X3
    matrixReg.loc[8,8] = sumX12X32
    matrixReg.loc[8,9] = sumX1X2X32
    #matrixReg.loc[8,10] = sumX12X2X32

    matrixReg.loc[9,0] = sumX2X3
    matrixReg.loc[9,1] = sumX1X2X3
    matrixReg.loc[9,2] = sumX22X3
    matrixReg.loc[9,3] = sumX2X32
    matrixReg.loc[9,4] = sumX12X2X3
    matrixReg.loc[9,5] = sumX23X3
    matrixReg.loc[9,6] = sumX2X33
    matrixReg.loc[9,7] = sumX1X22X3
    matrixReg.loc[9,8] = sumX1X2X32
    matrixReg.loc[9,9] = sumX22X32
    #matrixReg.loc[9,10] = sumX1X22X32

    #matrixReg.loc[10,0] = sumX1X2X3
    #matrixReg.loc[10,1] = sumX12X2X3
    #matrixReg.loc[10,2] = sumX1X22X3
    #matrixReg.loc[10,3] = sumX1X2X32
    #matrixReg.loc[10,4] = sumX13X2X3
    #matrixReg.loc[10,5] = sumX1X23X3
    #matrixReg.loc[10,6] = sumX1X2X33
    #matrixReg.loc[10,7] = sumX12X22X3
    #matrixReg.loc[10,8] = sumX12X2X32
    #matrixReg.loc[10,9] = sumX1X22X32
    #matrixReg.loc[10,10] = sumX12X22X32

    vectorReg.loc[0,0] = dataSet.iloc[:,y].sum()
    vectorReg.loc[1,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]).sum()
    vectorReg.loc[2,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x2]).sum()
    vectorReg.loc[3,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x3]).sum()
    vectorReg.loc[4,0] = (dataSet.iloc[:,y]*(dataSet.iloc[:,x1])**2).sum()
    vectorReg.loc[5,0] = (dataSet.iloc[:,y]*(dataSet.iloc[:,x2])**2).sum()
    vectorReg.loc[6,0] = (dataSet.iloc[:,y]*(dataSet.iloc[:,x3])**2).sum()
    vectorReg.loc[7,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()
    vectorReg.loc[8,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]).sum()
    vectorReg.loc[9,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    #vectorReg.loc[10,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()

    # find inverse matrix
    ####invMatrix = pd.DataFrame(np.linalg.inv(matrixReg))
    invMatrix = pd.DataFrame(np.linalg.pinv(matrixReg))

    # find solution to system
    polyCoef = invMatrix.dot(vectorReg)

    return polyCoef


# --------------------
# MNLR fit method with 3 variables
def fitLAM(dataSet, y, x1, x2, x3, lam):
    matrixReg = pd.DataFrame()
    vectorReg = pd.DataFrame()
    polyCoef = pd.DataFrame()
    sizeData=len(dataSet)

    # find regression matrix and independet vector
    sumX1 = dataSet.iloc[:,x1].sum()
    sumX2 = dataSet.iloc[:,x2].sum()
    sumX3 = dataSet.iloc[:,x3].sum()

    sumX12 = (dataSet.iloc[:,x1]**2).sum()
    sumX22 = (dataSet.iloc[:,x2]**2).sum()
    sumX32 = (dataSet.iloc[:,x3]**2).sum()

    sumX12lam = (lam*dataSet.iloc[:,x1]**2).sum()
    sumX22lam = (lam*dataSet.iloc[:,x2]**2).sum()
    sumX32lam = (lam*dataSet.iloc[:,x3]**2).sum()

    sumX13 = (dataSet.iloc[:,x1]**3).sum()
    sumX23 = (dataSet.iloc[:,x2]**3).sum()
    sumX33 = (dataSet.iloc[:,x3]**3).sum()

    sumX13lam = (lam*dataSet.iloc[:,x1]**3).sum()
    sumX23lam = (lam*dataSet.iloc[:,x2]**3).sum()
    sumX33lam = (lam*dataSet.iloc[:,x3]**3).sum()

    sumX14lam2 = (lam**2*dataSet.iloc[:,x1]**4).sum()
    sumX24lam2 = (lam**2*dataSet.iloc[:,x2]**4).sum()
    sumX34lam2 = (lam**2*dataSet.iloc[:,x3]**4).sum()

    sumX1X2 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()
    sumX1X3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x3]).sum()
    sumX2X3 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()

    sumX1X2lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()
    sumX1X3lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]).sum()
    sumX2X3lam = (lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX1X2X3lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()

    sumX12X2 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]).sum()
    sumX12X3 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x3]).sum()
    sumX22X3 = (dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]).sum()

    sumX12X2lam = (lam*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]).sum()
    sumX12X3lam = (lam*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x3]).sum()
    sumX22X3lam = (lam*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]).sum()

    sumX1X22 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2).sum()
    sumX1X32 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**2).sum()
    sumX2X32 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()

    sumX1X22lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2).sum()
    sumX1X32lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**2).sum()
    sumX2X32lam = (lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()

    sumX1X22lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2).sum()
    sumX1X32lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**2).sum()
    sumX2X32lam2 = (lam**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()

    sumX12X22lam = (lam*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]**2).sum()
    sumX12X32lam = (lam*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x3]**2).sum()
    sumX22X32lam = (lam*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]**2).sum()

    sumX12X22lam2 = (lam**2*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]**2).sum()
    sumX12X32lam2 = (lam**2*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x3]**2).sum()
    sumX22X32lam2 = (lam**2*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]**2).sum()

    sumX13X2 = (dataSet.iloc[:,x1]**3*dataSet.iloc[:,x2]).sum()
    sumX13X3 = (dataSet.iloc[:,x1]**3*dataSet.iloc[:,x3]).sum()
    sumX23X3 = (dataSet.iloc[:,x2]**3*dataSet.iloc[:,x3]).sum()

    sumX13X2lam2 = (lam**2*dataSet.iloc[:,x1]**3*dataSet.iloc[:,x2]).sum()
    sumX13X3lam2 = (lam**2*dataSet.iloc[:,x1]**3*dataSet.iloc[:,x3]).sum()
    sumX23X3lam2 = (lam**2*dataSet.iloc[:,x2]**3*dataSet.iloc[:,x3]).sum()

    sumX1X23 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**3).sum()
    sumX1X33 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**3).sum()
    sumX2X33 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**3).sum()

    sumX1X23lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**3).sum()
    sumX1X33lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**3).sum()
    sumX2X33lam2 = (lam**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**3).sum()

    sumX12X2X3 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX1X22X3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]).sum()
    sumX1X2X32 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()

    sumX12X2X3lam2 = (lam**2*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX1X22X3lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]).sum()
    sumX1X2X32lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()

    matrixReg.loc[0,0] = sizeData
    matrixReg.loc[0,1] = sumX1
    matrixReg.loc[0,2] = sumX2
    matrixReg.loc[0,3] = sumX3
    matrixReg.loc[0,4] = sumX12lam
    matrixReg.loc[0,5] = sumX22lam
    matrixReg.loc[0,6] = sumX32lam
    matrixReg.loc[0,7] = sumX1X2lam
    matrixReg.loc[0,8] = sumX1X3lam
    matrixReg.loc[0,9] = sumX2X3lam

    matrixReg.loc[1,0] = sumX1
    matrixReg.loc[1,1] = sumX12
    matrixReg.loc[1,2] = sumX1X2
    matrixReg.loc[1,3] = sumX1X3
    matrixReg.loc[1,4] = sumX13lam
    matrixReg.loc[1,5] = sumX1X22lam
    matrixReg.loc[1,6] = sumX1X32lam
    matrixReg.loc[1,7] = sumX12X2lam
    matrixReg.loc[1,8] = sumX12X3lam
    matrixReg.loc[1,9] = sumX1X2X3lam

    matrixReg.loc[2,0] = sumX2
    matrixReg.loc[2,1] = sumX1X2
    matrixReg.loc[2,2] = sumX22
    matrixReg.loc[2,3] = sumX2X3
    matrixReg.loc[2,4] = sumX12X2lam
    matrixReg.loc[2,5] = sumX23lam
    matrixReg.loc[2,6] = sumX2X32lam
    matrixReg.loc[2,7] = sumX1X22lam
    matrixReg.loc[2,8] = sumX1X2X3lam
    matrixReg.loc[2,9] = sumX22X3lam

    matrixReg.loc[3,0] = sumX3
    matrixReg.loc[3,1] = sumX1X3
    matrixReg.loc[3,2] = sumX2X3
    matrixReg.loc[3,3] = sumX32
    matrixReg.loc[3,4] = sumX12X3lam
    matrixReg.loc[3,5] = sumX22X3lam
    matrixReg.loc[3,6] = sumX33lam
    matrixReg.loc[3,7] = sumX1X2lam
    matrixReg.loc[3,8] = sumX1X32lam
    matrixReg.loc[3,9] = sumX2X32lam

    matrixReg.loc[4,0] = sumX12lam
    matrixReg.loc[4,1] = sumX13lam
    matrixReg.loc[4,2] = sumX12X2lam
    matrixReg.loc[4,3] = sumX12X3lam
    matrixReg.loc[4,4] = sumX14lam2
    matrixReg.loc[4,5] = sumX12X22lam2
    matrixReg.loc[4,6] = sumX12X32lam2
    matrixReg.loc[4,7] = sumX13X2lam2
    matrixReg.loc[4,8] = sumX13X3lam2
    matrixReg.loc[4,9] = sumX12X2X3lam2

    matrixReg.loc[5,0] = sumX22lam
    matrixReg.loc[5,1] = sumX1X22lam
    matrixReg.loc[5,2] = sumX23lam
    matrixReg.loc[5,3] = sumX22X3lam
    matrixReg.loc[5,4] = sumX12X22lam2
    matrixReg.loc[5,5] = sumX24lam2
    matrixReg.loc[5,6] = sumX22X32lam2
    matrixReg.loc[5,7] = sumX1X23lam2
    matrixReg.loc[5,8] = sumX1X22X3lam2
    matrixReg.loc[5,9] = sumX23X3lam2

    matrixReg.loc[6,0] = sumX32lam
    matrixReg.loc[6,1] = sumX1X32lam
    matrixReg.loc[6,2] = sumX2X32lam
    matrixReg.loc[6,3] = sumX33lam
    matrixReg.loc[6,4] = sumX12X32lam2
    matrixReg.loc[6,5] = sumX22X32lam2
    matrixReg.loc[6,6] = sumX34lam2
    matrixReg.loc[6,7] = sumX1X2X32lam2
    matrixReg.loc[6,8] = sumX1X33lam2
    matrixReg.loc[6,9] = sumX2X33lam2

    matrixReg.loc[7,0] = sumX1X2lam
    matrixReg.loc[7,1] = sumX12X2lam
    matrixReg.loc[7,2] = sumX1X22lam
    matrixReg.loc[7,3] = sumX1X2X3lam
    matrixReg.loc[7,4] = sumX13X2lam2
    matrixReg.loc[7,5] = sumX1X23lam2
    matrixReg.loc[7,6] = sumX1X2X32lam2
    matrixReg.loc[7,7] = sumX12X22lam2
    matrixReg.loc[7,8] = sumX12X2X3lam2
    matrixReg.loc[7,9] = sumX1X22X3lam2

    matrixReg.loc[8,0] = sumX1X3lam
    matrixReg.loc[8,1] = sumX12X3lam
    matrixReg.loc[8,2] = sumX1X2X3lam
    matrixReg.loc[8,3] = sumX1X32lam
    matrixReg.loc[8,4] = sumX13X3lam2
    matrixReg.loc[8,5] = sumX1X22X3lam2
    matrixReg.loc[8,6] = sumX1X33lam2
    matrixReg.loc[8,7] = sumX12X2X3lam2
    matrixReg.loc[8,8] = sumX12X32lam2
    matrixReg.loc[8,9] = sumX1X2X32lam2

    matrixReg.loc[9,0] = sumX2X3lam
    matrixReg.loc[9,1] = sumX1X2X3lam
    matrixReg.loc[9,2] = sumX22X3lam
    matrixReg.loc[9,3] = sumX2X32lam
    matrixReg.loc[9,4] = sumX12X2X3lam2
    matrixReg.loc[9,5] = sumX23X3lam2
    matrixReg.loc[9,6] = sumX2X33lam2
    matrixReg.loc[9,7] = sumX1X22X3lam2
    matrixReg.loc[9,8] = sumX1X2X32lam2
    matrixReg.loc[9,9] = sumX22X32lam2

    vectorReg.loc[0,0] = dataSet.iloc[:,y].sum()
    vectorReg.loc[1,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]).sum()
    vectorReg.loc[2,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x2]).sum()
    vectorReg.loc[3,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x3]).sum()
    vectorReg.loc[4,0] = (lam*dataSet.iloc[:,y]*(dataSet.iloc[:,x1])**2).sum()
    vectorReg.loc[5,0] = (lam*dataSet.iloc[:,y]*(dataSet.iloc[:,x2])**2).sum()
    vectorReg.loc[6,0] = (lam*dataSet.iloc[:,y]*(dataSet.iloc[:,x3])**2).sum()
    vectorReg.loc[7,0] = (lam*dataSet.iloc[:,y]*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()
    vectorReg.loc[8,0] = (lam*dataSet.iloc[:,y]*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]).sum()
    vectorReg.loc[9,0] = (lam*dataSet.iloc[:,y]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()

    # find inverse matrix
    ####invMatrix = pd.DataFrame(np.linalg.inv(matrixReg))
    invMatrix = pd.DataFrame(np.linalg.pinv(matrixReg))

    # find solution to system
    polyCoef = invMatrix.dot(vectorReg)

    return polyCoef




# --------------------
# MNLR fit method with 4 variables
def fit2(dataSet, y, x1, x2, x3, x4, lam):
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    #dataSet["4"] = 0
    #print(dataSet)
    #print("\n")

    matrixReg = pd.DataFrame()
    vectorReg = pd.DataFrame()
    polyCoef = pd.DataFrame()
    sizeData=len(dataSet)

    # find regression matrix and independet vector
    sumX1 = dataSet.iloc[:,x1].sum()
    sumX2 = dataSet.iloc[:,x2].sum()
    sumX3 = dataSet.iloc[:,x3].sum()
    sumX4 = dataSet.iloc[:,x4].sum()

    sumX12 = (dataSet.iloc[:,x1]**2).sum()
    sumX22 = (dataSet.iloc[:,x2]**2).sum()
    sumX32 = (dataSet.iloc[:,x3]**2).sum()
    sumX42 = (dataSet.iloc[:,x4]**2).sum()

    sumX12lam = (lam*dataSet.iloc[:,x1]**2).sum()
    sumX22lam = (lam*dataSet.iloc[:,x2]**2).sum()
    sumX32lam = (lam*dataSet.iloc[:,x3]**2).sum()
    sumX42lam = (lam*dataSet.iloc[:,x4]**2).sum()

    sumX13 = (dataSet.iloc[:,x1]**3).sum()
    sumX23 = (dataSet.iloc[:,x2]**3).sum()
    sumX33 = (dataSet.iloc[:,x3]**3).sum()
    sumX43 = (dataSet.iloc[:,x4]**4).sum()

    sumX13lam = (lam*dataSet.iloc[:,x1]**3).sum()
    sumX23lam = (lam*dataSet.iloc[:,x2]**3).sum()
    sumX33lam = (lam*dataSet.iloc[:,x3]**3).sum()
    sumX43lam = (lam*dataSet.iloc[:,x4]**3).sum()

    sumX14lam2 = (lam**2*dataSet.iloc[:,x1]**4).sum()
    sumX24lam2 = (lam**2*dataSet.iloc[:,x2]**4).sum()
    sumX34lam2 = (lam**2*dataSet.iloc[:,x3]**4).sum()
    sumX44lam2 = (lam**2*dataSet.iloc[:,x4]**4).sum()

    sumX1X2 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()
    sumX1X3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x3]).sum()
    sumX1X4 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x4]).sum()
    sumX2X3 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX2X4 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x4]).sum()
    sumX3X4 = (dataSet.iloc[:,x3]*dataSet.iloc[:,x4]).sum()

    sumX1X2lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()
    sumX1X3lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]).sum()
    sumX1X4lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x4]).sum()
    sumX2X3lam = (lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX2X4lam = (lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x4]).sum()
    sumX3X4lam = (lam*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]).sum()

    sumX1X2X3lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX1X2X4lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x4]).sum()
    sumX1X3X4lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]).sum()
    sumX2X3X4lam = (lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]).sum()

    sumX12X2 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]).sum()
    sumX12X3 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x3]).sum()
    sumX12X4 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x4]).sum()
    sumX22X3 = (dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]).sum()
    sumX22X4 = (dataSet.iloc[:,x2]**2*dataSet.iloc[:,x4]).sum()
    sumX32X4 = (dataSet.iloc[:,x3]**2*dataSet.iloc[:,x4]).sum()

    sumX12X2lam = (lam*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]).sum()
    sumX12X3lam = (lam*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x3]).sum()
    sumX12X4lam = (lam*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x4]).sum()
    sumX22X3lam = (lam*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]).sum()
    sumX22X4lam = (lam*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x4]).sum()
    sumX32X4lam = (lam*dataSet.iloc[:,x3]**2*dataSet.iloc[:,x4]).sum()

    sumX22X4lam2 = (lam**2*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x4]).sum()

    sumX1X22 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2).sum()
    sumX1X32 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**2).sum()
    sumX1X42 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x4]**2).sum()
    sumX2X32 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()
    sumX2X42 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x4]**2).sum()
    sumX3X42 = (dataSet.iloc[:,x3]*dataSet.iloc[:,x4]**2).sum()

    sumX1X22lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2).sum()
    sumX1X32lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**2).sum()
    sumX1X42lam = (lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x4]**2).sum()
    sumX2X32lam = (lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()
    sumX2X42lam = (lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x4]**2).sum()
    sumX3X42lam = (lam*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]**2).sum()

    sumX1X22lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2).sum()
    sumX1X32lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**2).sum()
    sumX1X42lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x4]**2).sum()
    sumX2X32lam2 = (lam**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()
    sumX2X42lam2 = (lam**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x4]**2).sum()
    sumX3X42lam2 = (lam**2*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]**2).sum()

    sumX12X22lam = (lam*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]**2).sum()
    sumX12X32lam = (lam*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x3]**2).sum()
    sumX12X42lam = (lam*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x4]**2).sum()
    sumX22X32lam = (lam*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]**2).sum()
    sumX22X42lam = (lam*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x4]**2).sum()
    sumX32X42lam = (lam*dataSet.iloc[:,x3]**2*dataSet.iloc[:,x4]**2).sum()

    sumX12X22lam2 = (lam**2*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]**2).sum()
    sumX12X32lam2 = (lam**2*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x3]**2).sum()
    sumX12X42lam2 = (lam**2*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x4]**2).sum()
    sumX22X32lam2 = (lam**2*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]**2).sum()
    sumX22X42lam2 = (lam**2*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x4]**2).sum()
    sumX32X42lam2 = (lam**2*dataSet.iloc[:,x3]**2*dataSet.iloc[:,x4]**2).sum()

    sumX13X2 = (dataSet.iloc[:,x1]**3*dataSet.iloc[:,x2]).sum()
    sumX13X3 = (dataSet.iloc[:,x1]**3*dataSet.iloc[:,x3]).sum()
    sumX13X4 = (dataSet.iloc[:,x1]**3*dataSet.iloc[:,x4]).sum()
    sumX23X3 = (dataSet.iloc[:,x2]**3*dataSet.iloc[:,x3]).sum()
    sumX23X4 = (dataSet.iloc[:,x2]**3*dataSet.iloc[:,x4]).sum()
    sumX33X4 = (dataSet.iloc[:,x3]**3*dataSet.iloc[:,x4]).sum()

    sumX13X2lam2 = (lam**2*dataSet.iloc[:,x1]**3*dataSet.iloc[:,x2]).sum()
    sumX13X3lam2 = (lam**2*dataSet.iloc[:,x1]**3*dataSet.iloc[:,x3]).sum()
    sumX13X4lam2 = (lam**2*dataSet.iloc[:,x1]**3*dataSet.iloc[:,x4]).sum()
    sumX23X3lam2 = (lam**2*dataSet.iloc[:,x2]**3*dataSet.iloc[:,x3]).sum()
    sumX23X4lam2 = (lam**2*dataSet.iloc[:,x2]**3*dataSet.iloc[:,x4]).sum()
    sumX33X4lam2 = (lam**2*dataSet.iloc[:,x3]**3*dataSet.iloc[:,x4]).sum()

    sumX1X23 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**3).sum()
    sumX1X33 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**3).sum()
    sumX1X43 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x4]**3).sum()
    sumX2X33 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**3).sum()
    sumX2X43 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x4]**3).sum()
    sumX3X43 = (dataSet.iloc[:,x3]*dataSet.iloc[:,x4]**3).sum()

    sumX1X23lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**3).sum()
    sumX1X33lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**3).sum()
    sumX1X43lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x4]**3).sum()
    sumX2X33lam2 = (lam**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**3).sum()
    sumX2X43lam2 = (lam**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x4]**3).sum()
    sumX3X43lam2 = (lam**2*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]**3).sum()

    sumX12X2X3 = (dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX1X22X3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]).sum()
    sumX1X2X32 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()

    sumX12X2X3lam2 = (lam**2*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX1X22X3lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]).sum()
    sumX1X2X32lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2).sum()

    sumX12X2X4lam2 = (lam**2*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x4]).sum()
    sumX1X22X4lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x4]).sum()
    sumX1X2X42lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x4]**2).sum()

    sumX12X3X4lam2 = (lam**2*dataSet.iloc[:,x1]**2*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]).sum()
    sumX1X32X4lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]**2*dataSet.iloc[:,x4]).sum()
    sumX1X3X42lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]**2).sum()

    sumX22X3X4lam2 = (lam**2*dataSet.iloc[:,x2]**2*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]).sum()
    sumX2X32X4lam2 = (lam**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]**2*dataSet.iloc[:,x4]).sum()
    sumX2X3X42lam2 = (lam**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]**2).sum()

    sumX2X3X4lam2 = (lam**2*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]).sum()

    sumX1X2X3X4lam2 = (lam**2*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]).sum()

    matrixReg.loc[0,0] = sizeData
    matrixReg.loc[0,1] = sumX1
    matrixReg.loc[0,2] = sumX2
    matrixReg.loc[0,3] = sumX3
    matrixReg.loc[0,4] = sumX4
    matrixReg.loc[0,5] = sumX12lam
    matrixReg.loc[0,6] = sumX22lam
    matrixReg.loc[0,7] = sumX32lam
    matrixReg.loc[0,8] = sumX42lam
    matrixReg.loc[0,9] = sumX1X2lam
    matrixReg.loc[0,10] = sumX1X3lam
    matrixReg.loc[0,11] = sumX1X4lam
    matrixReg.loc[0,12] = sumX2X3lam
    matrixReg.loc[0,13] = sumX2X4lam
    matrixReg.loc[0,14] = sumX3X4lam

    matrixReg.loc[1,0] = sumX1
    matrixReg.loc[1,1] = sumX12
    matrixReg.loc[1,2] = sumX1X2
    matrixReg.loc[1,3] = sumX1X3
    matrixReg.loc[1,4] = sumX1X4
    matrixReg.loc[1,5] = sumX13lam
    matrixReg.loc[1,6] = sumX1X22lam
    matrixReg.loc[1,7] = sumX1X32lam
    matrixReg.loc[1,8] = sumX1X42lam
    matrixReg.loc[1,9] = sumX12X2lam
    matrixReg.loc[1,10] = sumX12X3lam
    matrixReg.loc[1,11] = sumX12X4lam
    matrixReg.loc[1,12] = sumX1X2X3lam
    matrixReg.loc[1,13] = sumX1X2X4lam
    matrixReg.loc[1,14] = sumX1X3X4lam

    matrixReg.loc[2,0] = sumX2
    matrixReg.loc[2,1] = sumX1X2
    matrixReg.loc[2,2] = sumX22
    matrixReg.loc[2,3] = sumX2X3
    matrixReg.loc[2,4] = sumX2X4
    matrixReg.loc[2,5] = sumX12X2lam
    matrixReg.loc[2,6] = sumX23lam
    matrixReg.loc[2,7] = sumX2X32lam
    matrixReg.loc[2,8] = sumX2X42lam
    matrixReg.loc[2,9] = sumX1X22lam
    matrixReg.loc[2,10] = sumX1X2X3lam
    matrixReg.loc[2,11] = sumX1X2X4lam
    matrixReg.loc[2,12] = sumX22X3lam
    matrixReg.loc[2,13] = sumX22X4lam
    matrixReg.loc[2,14] = sumX2X3X4lam

    matrixReg.loc[3,0] = sumX3
    matrixReg.loc[3,1] = sumX1X3
    matrixReg.loc[3,2] = sumX2X3
    matrixReg.loc[3,3] = sumX32
    matrixReg.loc[3,4] = sumX3X4
    matrixReg.loc[3,5] = sumX12X3lam
    matrixReg.loc[3,6] = sumX22X3lam
    matrixReg.loc[3,7] = sumX33lam
    matrixReg.loc[3,8] = sumX3X42lam
    matrixReg.loc[3,9] = sumX1X2X3lam
    matrixReg.loc[3,10] = sumX1X32lam
    matrixReg.loc[3,11] = sumX1X3X4lam
    matrixReg.loc[3,12] = sumX2X32lam
    matrixReg.loc[3,13] = sumX2X3X4lam
    matrixReg.loc[3,14] = sumX32X4lam

    matrixReg.loc[4,0] = sumX4
    matrixReg.loc[4,1] = sumX1X4
    matrixReg.loc[4,2] = sumX2X4
    matrixReg.loc[4,3] = sumX3X4
    matrixReg.loc[4,4] = sumX42
    matrixReg.loc[4,5] = sumX12X4lam
    matrixReg.loc[4,6] = sumX22X4lam
    matrixReg.loc[4,7] = sumX32X4lam
    matrixReg.loc[4,8] = sumX43lam
    matrixReg.loc[4,9] = sumX1X2X4lam
    matrixReg.loc[4,10] = sumX1X3X4lam
    matrixReg.loc[4,11] = sumX1X42lam
    matrixReg.loc[4,12] = sumX2X3X4lam
    matrixReg.loc[4,13] = sumX22X42lam
    matrixReg.loc[4,14] = sumX3X42lam

    matrixReg.loc[5,0] = sumX12lam
    matrixReg.loc[5,1] = sumX13lam
    matrixReg.loc[5,2] = sumX12X2lam
    matrixReg.loc[5,3] = sumX12X3lam
    matrixReg.loc[5,4] = sumX12X4lam
    matrixReg.loc[5,5] = sumX14lam2
    matrixReg.loc[5,6] = sumX12X22lam2
    matrixReg.loc[5,7] = sumX12X32lam2
    matrixReg.loc[5,8] = sumX12X42lam2
    matrixReg.loc[5,9] = sumX13X2lam2
    matrixReg.loc[5,10] = sumX13X3lam2
    matrixReg.loc[5,11] = sumX13X4lam2
    matrixReg.loc[5,12] = sumX12X2X3lam2
    matrixReg.loc[5,13] = sumX12X2X4lam2
    matrixReg.loc[5,14] = sumX12X3X4lam2

    matrixReg.loc[6,0] = sumX22lam
    matrixReg.loc[6,1] = sumX1X22lam
    matrixReg.loc[6,2] = sumX23lam
    matrixReg.loc[6,3] = sumX22X3lam
    matrixReg.loc[6,4] = sumX22X4lam
    matrixReg.loc[6,5] = sumX12X22lam2
    matrixReg.loc[6,6] = sumX24lam2
    matrixReg.loc[6,7] = sumX22X32lam2
    matrixReg.loc[6,8] = sumX22X42lam2
    matrixReg.loc[6,9] = sumX1X23lam2
    matrixReg.loc[6,10] = sumX1X22X3lam2
    matrixReg.loc[6,11] = sumX1X22X4lam2
    matrixReg.loc[6,12] = sumX23X3lam2
    matrixReg.loc[6,13] = sumX23X4lam2
    matrixReg.loc[6,14] = sumX22X3X4lam2

    matrixReg.loc[7,0] = sumX32lam
    matrixReg.loc[7,1] = sumX1X32lam
    matrixReg.loc[7,2] = sumX2X32lam
    matrixReg.loc[7,3] = sumX33lam
    matrixReg.loc[7,4] = sumX32X4lam
    matrixReg.loc[7,5] = sumX12X32lam2
    matrixReg.loc[7,6] = sumX22X32lam2
    matrixReg.loc[7,7] = sumX34lam2
    matrixReg.loc[7,8] = sumX32X42lam2
    matrixReg.loc[7,9] = sumX1X2X32lam2
    matrixReg.loc[7,10] = sumX1X33lam2
    matrixReg.loc[7,11] = sumX1X32X4lam2
    matrixReg.loc[7,12] = sumX2X33lam2
    matrixReg.loc[7,13] = sumX2X32X4lam2
    matrixReg.loc[7,14] = sumX33X4lam2

    matrixReg.loc[8,0] = sumX42lam
    matrixReg.loc[8,1] = sumX1X42lam
    matrixReg.loc[8,2] = sumX2X42lam
    matrixReg.loc[8,3] = sumX3X42lam
    matrixReg.loc[8,4] = sumX43lam
    matrixReg.loc[8,5] = sumX12X42lam2
    matrixReg.loc[8,6] = sumX22X42lam2
    matrixReg.loc[8,7] = sumX32X42lam2
    matrixReg.loc[8,8] = sumX44lam2
    matrixReg.loc[8,9] = sumX1X2X42lam2
    matrixReg.loc[8,10] = sumX1X3X42lam2
    matrixReg.loc[8,11] = sumX1X43lam2
    matrixReg.loc[8,12] = sumX2X3X42lam2
    matrixReg.loc[8,13] = sumX2X43lam2
    matrixReg.loc[8,14] = sumX3X43lam2

    matrixReg.loc[9,0] = sumX1X2lam
    matrixReg.loc[9,1] = sumX12X2lam
    matrixReg.loc[9,2] = sumX1X22lam
    matrixReg.loc[9,3] = sumX1X2X3lam
    matrixReg.loc[9,4] = sumX1X2X4lam
    matrixReg.loc[9,5] = sumX13X2lam2
    matrixReg.loc[9,6] = sumX1X23lam2
    matrixReg.loc[9,7] = sumX1X2X32lam2
    matrixReg.loc[9,8] = sumX1X2X42lam2
    matrixReg.loc[9,9] = sumX12X22lam2
    matrixReg.loc[9,10] = sumX12X2X3lam2
    matrixReg.loc[9,11] = sumX12X2X4lam2
    matrixReg.loc[9,12] = sumX1X22X3lam2
    matrixReg.loc[9,13] = sumX22X4lam2
    matrixReg.loc[9,14] = sumX1X2X3X4lam2

    matrixReg.loc[10,0] = sumX1X3lam
    matrixReg.loc[10,1] = sumX12X3lam
    matrixReg.loc[10,2] = sumX1X2X3lam
    matrixReg.loc[10,3] = sumX1X32lam
    matrixReg.loc[10,4] = sumX1X3X4lam
    matrixReg.loc[10,5] = sumX13X3lam2
    matrixReg.loc[10,6] = sumX1X22X3lam2
    matrixReg.loc[10,7] = sumX1X33lam2
    matrixReg.loc[10,8] = sumX1X3X42lam2
    matrixReg.loc[10,9] = sumX12X2X3lam2
    matrixReg.loc[10,10] = sumX12X32lam2
    matrixReg.loc[10,11] = sumX12X3X4lam2
    matrixReg.loc[10,12] = sumX1X2X32lam2
    matrixReg.loc[10,13] = sumX1X2X3X4lam2
    matrixReg.loc[10,14] = sumX1X32X4lam2

    matrixReg.loc[11,0] = sumX1X4lam
    matrixReg.loc[11,1] = sumX12X4lam
    matrixReg.loc[11,2] = sumX1X2X4lam
    matrixReg.loc[11,3] = sumX1X3X4lam
    matrixReg.loc[11,4] = sumX1X42lam
    matrixReg.loc[11,5] = sumX13X4lam2
    matrixReg.loc[11,6] = sumX1X22X4lam2
    matrixReg.loc[11,7] = sumX1X32X4lam2
    matrixReg.loc[11,8] = sumX1X43lam2
    matrixReg.loc[11,9] = sumX12X2X4lam2
    matrixReg.loc[11,10] = sumX12X3X4lam2
    matrixReg.loc[11,11] = sumX12X42lam2
    matrixReg.loc[11,12] = sumX1X2X3X4lam2
    matrixReg.loc[11,13] = sumX1X2X42lam2
    matrixReg.loc[11,14] = sumX1X3X42lam2

    matrixReg.loc[12,0] = sumX2X3lam
    matrixReg.loc[12,1] = sumX1X2X3lam
    matrixReg.loc[12,2] = sumX22X3lam
    matrixReg.loc[12,3] = sumX2X32lam
    matrixReg.loc[12,4] = sumX2X3X4lam
    matrixReg.loc[12,5] = sumX12X2X3lam2
    matrixReg.loc[12,6] = sumX23X3lam2
    matrixReg.loc[12,7] = sumX2X33lam2
    matrixReg.loc[12,8] = sumX2X3X42lam2
    matrixReg.loc[12,9] = sumX1X22X3lam2
    matrixReg.loc[12,10] = sumX1X2X32lam2
    matrixReg.loc[12,11] = sumX1X2X3X4lam2
    matrixReg.loc[12,12] = sumX22X32lam2
    matrixReg.loc[12,13] = sumX22X3X4lam2
    matrixReg.loc[12,14] = sumX2X32X4lam2

    matrixReg.loc[13,0] = sumX2X4lam
    matrixReg.loc[13,1] = sumX1X2X4lam
    matrixReg.loc[13,2] = sumX22X4lam
    matrixReg.loc[13,3] = sumX2X3X4lam
    matrixReg.loc[13,4] = sumX2X42lam
    matrixReg.loc[13,5] = sumX12X2X4lam2
    matrixReg.loc[13,6] = sumX23X4lam2
    matrixReg.loc[13,7] = sumX2X32X4lam2
    matrixReg.loc[13,8] = sumX2X43lam2
    matrixReg.loc[13,9] = sumX1X22X4lam2
    matrixReg.loc[13,10] = sumX1X2X3X4lam2
    matrixReg.loc[13,11] = sumX1X2X42lam2
    matrixReg.loc[13,12] = sumX2X3X4lam2
    matrixReg.loc[13,13] = sumX22X42lam2
    matrixReg.loc[13,14] = sumX2X3X42lam2

    matrixReg.loc[14,0] = sumX3X4lam
    matrixReg.loc[14,1] = sumX1X3X4lam
    matrixReg.loc[14,2] = sumX2X3X4lam
    matrixReg.loc[14,3] = sumX32X4lam
    matrixReg.loc[14,4] = sumX3X42lam
    matrixReg.loc[14,5] = sumX12X3X4lam2
    matrixReg.loc[14,6] = sumX22X3X4lam2
    matrixReg.loc[14,7] = sumX33X4lam2
    matrixReg.loc[14,8] = sumX3X43lam2
    matrixReg.loc[14,9] = sumX1X2X3X4lam2
    matrixReg.loc[14,10] = sumX1X32X4lam2
    matrixReg.loc[14,11] = sumX1X3X42lam2
    matrixReg.loc[14,12] = sumX2X32X4lam2
    matrixReg.loc[14,13] = sumX2X3X42lam2
    matrixReg.loc[14,14] = sumX32X42lam2

    vectorReg.loc[0,0] = dataSet.iloc[:,y].sum()
    vectorReg.loc[1,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]).sum()
    vectorReg.loc[2,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x2]).sum()
    vectorReg.loc[3,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x3]).sum()
    vectorReg.loc[4,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x4]).sum()
    vectorReg.loc[5,0] = (lam*dataSet.iloc[:,y]*(dataSet.iloc[:,x1])**2).sum()
    vectorReg.loc[6,0] = (lam*dataSet.iloc[:,y]*(dataSet.iloc[:,x2])**2).sum()
    vectorReg.loc[7,0] = (lam*dataSet.iloc[:,y]*(dataSet.iloc[:,x3])**2).sum()
    vectorReg.loc[8,0] = (lam*dataSet.iloc[:,y]*(dataSet.iloc[:,x4])**2).sum()
    vectorReg.loc[9,0] = (lam*dataSet.iloc[:,y]*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()
    vectorReg.loc[10,0] = (lam*dataSet.iloc[:,y]*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]).sum()
    vectorReg.loc[11,0] = (lam*dataSet.iloc[:,y]*dataSet.iloc[:,x1]*dataSet.iloc[:,x4]).sum()
    vectorReg.loc[12,0] = (lam*dataSet.iloc[:,y]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    vectorReg.loc[13,0] = (lam*dataSet.iloc[:,y]*dataSet.iloc[:,x2]*dataSet.iloc[:,x4]).sum()
    vectorReg.loc[14,0] = (lam*dataSet.iloc[:,y]*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]).sum()

    # find inverse matrix
    ####invMatrix = pd.DataFrame(np.linalg.inv(matrixReg))
    invMatrix = pd.DataFrame(np.linalg.pinv(matrixReg))

    # find solution to system
    polyCoef = invMatrix.dot(vectorReg)

    return polyCoef





# --------------------
# MNLR fit method
def fitDin(dataSet, y, x1, x2, x3, func):
    matrixReg = pd.DataFrame()
    vectorReg = pd.DataFrame()
    polyCoef = pd.DataFrame()
    sizeData=len(dataSet)

    # find regression matrix and independet vector
    sumX1 = dataSet.iloc[:,x1].sum()
    sumX2 = dataSet.iloc[:,x2].sum()
    sumX3 = dataSet.iloc[:,x3].sum()

    sumEX1 = func(dataSet.iloc[:,x1]).sum()
    sumEX2 = func(dataSet.iloc[:,x2]).sum()
    sumEX3 = func(dataSet.iloc[:,x3]).sum()

    sumX1EX1 = (dataSet.iloc[:,x1]*func(dataSet.iloc[:,x1])).sum()
    sumX2EX2 = (dataSet.iloc[:,x2]*func(dataSet.iloc[:,x2])).sum()
    sumX3EX3 = (dataSet.iloc[:,x3]*func(dataSet.iloc[:,x3])).sum()

    sumE2X1 = (func(dataSet.iloc[:,x1]*2)).sum()
    sumE2X2 = (func(dataSet.iloc[:,x2]*2)).sum()
    sumE2X3 = (func(dataSet.iloc[:,x3]*2)).sum()

    sumX1X2 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()
    sumX1X3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x3]).sum()
    sumX2X3 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX1X2X3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()

    sumEX1X2 = (func(dataSet.iloc[:,x1])*dataSet.iloc[:,x2]).sum()
    sumEX1X3 = (func(dataSet.iloc[:,x1])*dataSet.iloc[:,x3]).sum()
    sumEX2X3 = (func(dataSet.iloc[:,x2])*dataSet.iloc[:,x3]).sum()

    sumX1EX2 = (dataSet.iloc[:,x1]*func(dataSet.iloc[:,x2])).sum()
    sumX1EX3 = (dataSet.iloc[:,x1]*func(dataSet.iloc[:,x3])).sum()
    sumX2EX3 = (dataSet.iloc[:,x2]*func(dataSet.iloc[:,x3])).sum()

    sumEX1EX2 = (func(dataSet.iloc[:,x1])*func(dataSet.iloc[:,x2])).sum()
    sumEX1EX3 = (func(dataSet.iloc[:,x1])*func(dataSet.iloc[:,x3])).sum()
    sumEX2EX3 = (func(dataSet.iloc[:,x2])*func(dataSet.iloc[:,x3])).sum()

    sumX1EX1X2 = (dataSet.iloc[:,x1]*func(dataSet.iloc[:,x1])*dataSet.iloc[:,x2]).sum()
    sumX1EX1X3 = (dataSet.iloc[:,x1]*func(dataSet.iloc[:,x1])*dataSet.iloc[:,x3]).sum()
    sumX2EX2X3 = (dataSet.iloc[:,x2]*func(dataSet.iloc[:,x2])*dataSet.iloc[:,x3]).sum()

    sumX1X2EX2 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*func(dataSet.iloc[:,x2])).sum()
    sumX1X3EX3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x3]*func(dataSet.iloc[:,x3])).sum()
    sumX2X3EX3 = (dataSet.iloc[:,x2]*dataSet.iloc[:,x3]*func(dataSet.iloc[:,x3])).sum()

    sumEX1X2X3 = (func(dataSet.iloc[:,x1])*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()
    sumX1EX2X3 = (dataSet.iloc[:,x1]*func(dataSet.iloc[:,x2])*dataSet.iloc[:,x3]).sum()
    sumX1X2EX3 = (dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*func(dataSet.iloc[:,x3])).sum()

    matrixReg.loc[0,0] = sizeData
    matrixReg.loc[0,1] = sumX1
    matrixReg.loc[0,2] = sumX2
    matrixReg.loc[0,3] = sumX3
    matrixReg.loc[0,4] = sumEX1
    matrixReg.loc[0,5] = sumEX2
    matrixReg.loc[0,6] = sumEX3
    matrixReg.loc[0,7] = sumX1X2
    matrixReg.loc[0,8] = sumX1X3
    matrixReg.loc[0,9] = sumX2X3

    matrixReg.loc[1,0] = sumX1
    matrixReg.loc[1,1] = sumEX1
    matrixReg.loc[1,2] = sumX1X2
    matrixReg.loc[1,3] = sumX1X3
    matrixReg.loc[1,4] = sumX1EX1
    matrixReg.loc[1,5] = sumX1EX2
    matrixReg.loc[1,6] = sumX1EX3
    matrixReg.loc[1,7] = sumEX1X2
    matrixReg.loc[1,8] = sumEX1X3
    matrixReg.loc[1,9] = sumX1X2X3

    matrixReg.loc[2,0] = sumX2
    matrixReg.loc[2,1] = sumX1X2
    matrixReg.loc[2,2] = sumEX2
    matrixReg.loc[2,3] = sumX2X3
    matrixReg.loc[2,4] = sumEX1X2
    matrixReg.loc[2,5] = sumX2EX2
    matrixReg.loc[2,6] = sumX2EX3
    matrixReg.loc[2,7] = sumX1EX2
    matrixReg.loc[2,8] = sumX1X2X3
    matrixReg.loc[2,9] = sumEX2X3

    matrixReg.loc[3,0] = sumX3
    matrixReg.loc[3,1] = sumX1X3
    matrixReg.loc[3,2] = sumX2X3
    matrixReg.loc[3,3] = sumEX3
    matrixReg.loc[3,4] = sumEX1X3
    matrixReg.loc[3,5] = sumEX2X3
    matrixReg.loc[3,6] = sumX3EX3
    matrixReg.loc[3,7] = sumX1X2
    matrixReg.loc[3,8] = sumX1EX3
    matrixReg.loc[3,9] = sumX2EX3

    matrixReg.loc[4,0] = sumEX1
    matrixReg.loc[4,1] = sumX1EX1
    matrixReg.loc[4,2] = sumEX1X2
    matrixReg.loc[4,3] = sumEX1X3
    matrixReg.loc[4,4] = sumE2X1
    matrixReg.loc[4,5] = sumEX1EX2
    matrixReg.loc[4,6] = sumEX1EX3
    matrixReg.loc[4,7] = sumX1EX1X2
    matrixReg.loc[4,8] = sumX1EX1X3
    matrixReg.loc[4,9] = sumEX1X2X3

    matrixReg.loc[5,0] = sumEX2
    matrixReg.loc[5,1] = sumX1EX2
    matrixReg.loc[5,2] = sumX2EX2
    matrixReg.loc[5,3] = sumEX2X3
    matrixReg.loc[5,4] = sumEX1EX2
    matrixReg.loc[5,5] = sumE2X2
    matrixReg.loc[5,6] = sumEX2EX3
    matrixReg.loc[5,7] = sumX1X2EX2
    matrixReg.loc[5,8] = sumX1EX2X3
    matrixReg.loc[5,9] = sumX2EX2X3

    matrixReg.loc[6,0] = sumEX3
    matrixReg.loc[6,1] = sumX1EX3
    matrixReg.loc[6,2] = sumX2EX3
    matrixReg.loc[6,3] = sumX3EX3
    matrixReg.loc[6,4] = sumEX1EX3
    matrixReg.loc[6,5] = sumEX2EX3
    matrixReg.loc[6,6] = sumE2X3
    matrixReg.loc[6,7] = sumX1X2EX3
    matrixReg.loc[6,8] = sumX1X3EX3
    matrixReg.loc[6,9] = sumX2X3EX3

    matrixReg.loc[7,0] = sumX1X2
    matrixReg.loc[7,1] = sumEX1X2
    matrixReg.loc[7,2] = sumX1EX2
    matrixReg.loc[7,3] = sumX1X2X3
    matrixReg.loc[7,4] = sumX1EX1X2
    matrixReg.loc[7,5] = sumX1X2EX2
    matrixReg.loc[7,6] = sumX1X2EX3
    matrixReg.loc[7,7] = sumEX1EX2
    matrixReg.loc[7,8] = sumEX1X2X3
    matrixReg.loc[7,9] = sumX1EX2X3

    matrixReg.loc[8,0] = sumX1X3
    matrixReg.loc[8,1] = sumEX1X3
    matrixReg.loc[8,2] = sumX1X2X3
    matrixReg.loc[8,3] = sumX1EX3
    matrixReg.loc[8,4] = sumX1EX1X3
    matrixReg.loc[8,5] = sumX1EX2X3
    matrixReg.loc[8,6] = sumX1X3EX3
    matrixReg.loc[8,7] = sumEX1X2X3
    matrixReg.loc[8,8] = sumEX1EX3
    matrixReg.loc[8,9] = sumX1X2EX3

    matrixReg.loc[9,0] = sumX2X3
    matrixReg.loc[9,1] = sumX1X2X3
    matrixReg.loc[9,2] = sumEX2X3
    matrixReg.loc[9,3] = sumX2EX3
    matrixReg.loc[9,4] = sumEX1X2X3
    matrixReg.loc[9,5] = sumX2EX2X3
    matrixReg.loc[9,6] = sumX2X3EX3
    matrixReg.loc[9,7] = sumX1EX2X3
    matrixReg.loc[9,8] = sumX1X2EX3
    matrixReg.loc[9,9] = sumEX2EX3

    vectorReg.loc[0,0] = dataSet.iloc[:,y].sum()
    vectorReg.loc[1,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]).sum()
    vectorReg.loc[2,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x2]).sum()
    vectorReg.loc[3,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x3]).sum()
    vectorReg.loc[4,0] = (dataSet.iloc[:,y]*func(dataSet.iloc[:,x1])).sum()
    vectorReg.loc[5,0] = (dataSet.iloc[:,y]*func(dataSet.iloc[:,x2])).sum()
    vectorReg.loc[6,0] = (dataSet.iloc[:,y]*func(dataSet.iloc[:,x3])).sum()
    vectorReg.loc[7,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]).sum()
    vectorReg.loc[8,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]).sum()
    vectorReg.loc[9,0] = (dataSet.iloc[:,y]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]).sum()

    # find inverse matrix
    ####invMatrix = pd.DataFrame(np.linalg.inv(matrixReg))
    invMatrix = pd.DataFrame(np.linalg.pinv(matrixReg))

    # find solution to system
    polyCoef = invMatrix.dot(vectorReg)

    return polyCoef



# --------------------
# MNLR predict method with 1 exogenous variable
def predict1PC(yModel, polyCoef, dataSet, y, x1):
    yModel['fitted_mnlr'] = polyCoef.iloc[2,0]*(dataSet.iloc[:,x1]**2)+polyCoef.iloc[1,0]*dataSet.iloc[:,x1]+polyCoef.iloc[0,0]

    return yModel

# --------------------
# MNLR predict method with 3 exogenous variables
def predict(yModel, polyCoef, dataSet, y, x1, x2, x3):
    yModel['fitted_mnlr'] = (#polyCoef.iloc[10,0]*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[9,0]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[8,0]*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[7,0]*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]
                       +polyCoef.iloc[6,0]*dataSet.iloc[:,x3]**2
                       +polyCoef.iloc[5,0]*dataSet.iloc[:,x2]**2
                       +polyCoef.iloc[4,0]*dataSet.iloc[:,x1]**2
                       +polyCoef.iloc[3,0]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[2,0]*dataSet.iloc[:,x2]
                       +polyCoef.iloc[1,0]*dataSet.iloc[:,x1]
                       +polyCoef.iloc[0,0])

    return yModel

# --------------------
# MNLR predict method
def predictLAM(yModel, polyCoef, dataSet, y, x1, x2, x3, lam):
    yModel['fitted_mnlr'] = (#polyCoef.iloc[10,0]*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[9,0]*lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[8,0]*lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[7,0]*lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]
                       +polyCoef.iloc[6,0]*lam*dataSet.iloc[:,x3]**2
                       +polyCoef.iloc[5,0]*lam*dataSet.iloc[:,x2]**2
                       +polyCoef.iloc[4,0]*lam*dataSet.iloc[:,x1]**2
                       +polyCoef.iloc[3,0]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[2,0]*dataSet.iloc[:,x2]
                       +polyCoef.iloc[1,0]*dataSet.iloc[:,x1]
                       +polyCoef.iloc[0,0])

    return yModel


# --------------------
# MNLR predict method with 4 variables
def predict2(yModel, polyCoef, dataSet, y, x1, x2, x3, x4, lam):
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    #dataSet["4"] = 0
    #print(dataSet)
    #print("\n")

    yModel['fitted_mnlr'] = (polyCoef.iloc[14,0]*lam*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]
                       +polyCoef.iloc[13,0]*lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x4]
                       +polyCoef.iloc[12,0]*lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[11,0]*lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x4]
                       +polyCoef.iloc[10,0]*lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[9,0]*lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]
                       +polyCoef.iloc[8,0]*lam*dataSet.iloc[:,x4]**2
                       +polyCoef.iloc[7,0]*lam*dataSet.iloc[:,x3]**2
                       +polyCoef.iloc[6,0]*lam*dataSet.iloc[:,x2]**2
                       +polyCoef.iloc[5,0]*lam*dataSet.iloc[:,x1]**2
                       +polyCoef.iloc[4,0]*dataSet.iloc[:,x4]
                       +polyCoef.iloc[3,0]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[2,0]*dataSet.iloc[:,x2]
                       +polyCoef.iloc[1,0]*dataSet.iloc[:,x1]
                       +polyCoef.iloc[0,0])

    return yModel


# --------------------
# MNLR predict method
def predictDin(yModel, polyCoef, dataSet, y, x1, x2, x3, func):
    yModel['fitted_mnlr'] = (polyCoef.iloc[9,0]*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[8,0]*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[7,0]*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]
                       +polyCoef.iloc[6,0]*func(dataSet.iloc[:,x3])
                       +polyCoef.iloc[5,0]*func(dataSet.iloc[:,x2])
                       +polyCoef.iloc[4,0]*func(dataSet.iloc[:,x1])
                       +polyCoef.iloc[3,0]*dataSet.iloc[:,x3]
                       +polyCoef.iloc[2,0]*dataSet.iloc[:,x2]
                       +polyCoef.iloc[1,0]*dataSet.iloc[:,x1]
                       +polyCoef.iloc[0,0])
    return yModel


# --------------------
# MNLR summary method
def summary(yModel, polyCoef, dataSet, y):
    pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 4)
    modPerformance = pd.DataFrame(index=['Model Performance'])
    regError = pd.DataFrame(index=['Relative Error'])

    #print(matrixReg.dot(polyCoef))
    print("================================================================")
    print("                   Target function Coefficients")
    print("----------------------------------------------------------------")
    [print("A",i," :", polyCoef.iloc[i,0].round(3)) for i in range(len(polyCoef))]
    print("\n")

    # compute residuals and relative error
    yModel['residuals'] = dataSet.iloc[:,y] - yModel['fitted_mnlr']
    yModel['Rel Error'] = (1-(yModel['fitted_mnlr']/dataSet.iloc[:,y])).abs()
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
    modPerformance['SSR'] = ((yModel['fitted_mnlr']-dataSet.iloc[:,y].mean(axis=0))**2).sum()
    modPerformance['TSS'] = modPerformance['RSS']+modPerformance['SSR']
    modPerformance['R Square'] = 1-modPerformance['RSS']/modPerformance['TSS']
    print("==========================================================")
    print("                 MODEL PERFORMANCE")
    print("----------------------------------------------------------")
    print(modPerformance)
    print("----------------------------------------------------------")
    print(regError)
    print("\n")

# --------------------
# MNLR plot method
def mnlrPlot(yModel, dataSet, y):
    pLabel = 'MNLR'

    #plot regression
    plt.figure()
    plt.plot(dataSet.iloc[:,y], color='navy',label='Raw data',marker=2,markevery=150)
    plt.plot(yModel['fitted_mnlr'], color='r',label='MNLR',marker=3,markevery=150)
    plt.plot(yModel['fitted_lsp'], color='g',label='LSP',marker=4,markevery=150)
    plt.grid(color='b',linestyle='dashed')
    plt.gcf().autofmt_xdate()
    plt.title("IPC Mexico Index Regression", fontweight='bold')
    plt.xlabel('Date (t)')
    plt.ylabel('IPC Mexico index (log)')
    plt.legend(loc='lower right')
    plt.show()

    # plot residuals
    totalLabel = pLabel + ' Model Performance'

    plt.figure()
    plt.plot(yModel['fitted_mnlr'], yModel['residuals'],'o', color='g',label='Residuals')
    plt.grid(color='b',linestyle='dashed')
    plt.axhline(y=0, color='red')
    plt.title(totalLabel, fontweight='bold')
    plt.xlabel('Predictor')
    plt.ylabel('Residuals')
    plt.legend(loc='lower right')
    plt.show()

    # plot model relative rror
    totalLabel = pLabel + ' Relative Error Performance'

    m = yModel['Rel Error'].mean(axis=0)

    plt.figure()
    plt.plot(yModel['Rel Error'], color='orange',label='Relative Error', linestyle='--')
    plt.grid(color='b',linestyle='dashed')
    plt.axhline(y=m, color='r', linestyle='dashed', label='Mean')
    plt.gcf().autofmt_xdate()
    plt.title(totalLabel, fontweight='bold')
    plt.xlabel('Date (t)')
    plt.ylabel('Relative Error')
    plt.legend(loc='upper right')
    plt.show()

    # plot relative rror histogram
    totalLabel = pLabel + ' Relative Error Histogram'

    sd = yModel['Rel Error'].std(axis=0)
    q1 = yModel['Rel Error'].quantile(.25)
    q3 = yModel['Rel Error'].quantile(.75)

    plt.figure()
    plt.hist(yModel['Rel Error'], bins=20, color='navy', edgecolor='k')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin,xmax,100)
    p = norm.pdf(x,m,sd)
    plt.plot(x,p, color ='cyan', label='Distribution')
    plt.axvline(m, color='r', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(m+sd, color='m', linestyle='dashed', linewidth=1, label='Std')
    plt.axvline(m-sd, color='m', linestyle='dashed', linewidth=1)
    plt.axvline(q1, color='orange', linestyle='dashed', linewidth=1, label='Q1')
    plt.axvline(q3, color='orange', linestyle='dashed', linewidth=1, label='Q3')
    plt.grid(color='b',linestyle='dashed')
    plt.title(totalLabel, fontweight='bold')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

    totalLabel = pLabel + ' Relative Error'

    plt.figure()
    plt.boxplot(yModel['Rel Error'], notch=True)
    plt.title(totalLabel, fontweight='bold')
    plt.show()

# --------------------
# my lsp method
def myLSP(yModel, dataSet, y):
    sizeData = len(dataSet)

    polyCoefLSP = pd.DataFrame(np.polyfit(range(sizeData), dataSet.iloc[:,y].values, 6))   # coefficients
    regModel = np.poly1d(polyCoefLSP.to_numpy().reshape(1,7)[0])              # equation
    yModel['fitted_lsp'] = regModel(range(sizeData))

    return yModel
