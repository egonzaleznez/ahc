""" *****************

    Thesis Main module

    File name: thesis.py
    Related files: moduleAON.py, moduleFA, moduleReg, moduleData.py
    Created by: Enrique Gonzalez
    Date: 13 mar 2023

    Compilacion:
    Ejecucion:

***************** """

# --------------------
# import libraries
import warnings
import numpy as np # to use numpy arrays instead of lists
import pandas as pd # DataFrame (table)

# my libraries
import moduleAON
import moduleFA
import moduleMNLR
import moduleData

def main():
    warnings.filterwarnings("ignore")

    # --------------------
    # read data Set
    dataSet = moduleData.getDataSet()
    moduleData.printDataSet(dataSet)

    # --------------------
    # smooth data
    normSet = moduleData.normalizeDataSet(dataSet, smooth=True)
    ##moduleData.printDataSet(normSet)
    moduleData.plotSet(normSet)
    moduleData.plotExoVar(normSet)

    # --------------------
    # finance analysis
    ##dataFA = moduleFA.priceReturn(normSet)
    #moduleFA.compoundReturn(dataFA,normSet['RFR'])
    ##[dataFA, fmaLabel, smaLabel] = moduleFA.tradeStrat(dataFA,10,30)
    ##moduleData.printDataSet(dataFA)
    ##moduleData.plotStrat(dataFA, fmaLabel, smaLabel)

    # --------------------
    # preprocess data
    num_comp = 3
    preprocessData = moduleData.computePCA(normSet, pcaprocess=True, num_comp=num_comp)  #normSet

    # --------------------
    # split data
    rand = False
    ts_size = .15
    [principalComp, principalComp3] = moduleData.splitDataSet(preprocessData, test_size=ts_size, randSplit=rand)
  

    # --------------------
    # test new moduleMNLR
    #moduleMNLR.mnlrReg(principalComp, principalComp3, num_comp=3)

#"""
    # --------------------
    # AON param
    error = [9e-4]#[.0009, .0006]
    max_mols = [12]#[2, 4, 8, 12]
    lam = [1e-10]#[1, .95, 1e-10, 0]
   
    max_repeat = 1#50
    iteration = 1
    setup = 1
    best_individual = 1

    iter_list = []
    setup_list = []
    ts_list = []
    error_list = []
    mols_list = []
    max_mols_list = []
    lam_list = []
    tr_mean_list = []
    tr_median_list = []
    tr_sd_list = []
    tr_mad_list = []
    tr_max_list = []
    tr_min_list = []
    tr_range_list = []
    ts_mean_list = []
    ts_median_list = []
    ts_sd_list = []
    ts_mad_list = []
    ts_max_list = []
    ts_min_list = []
    ts_range_list = []
    best_list = []
    
    for i in range(len(error)):
        for j in range(len(max_mols)):
            for k in range(len(lam)):
                for iter in range(max_repeat):
                    
                    print("==========================================================")
                    print("                 AON MODEL")
                    print("----------------------------------------------------------")
                    print("                Training Set")
                    print("----------------------------------------------------------")
                    print("\n")
                    # create AON model
                    [aon, func] = moduleAON.fit(sigma=principalComp,maxMol=max_mols[j],epsilon=error[i],lam=lam[k])
                    # do prediction with model
                    [yModel, regError] = moduleAON.predict(principalComp,aon,func,0,lam[k])
                    # print AON compound
                    moduleData.printDataSet(aon)
                    # AON summary
                    [modPerformance, regError2] = moduleAON.summary(yModel)
                    
                    iter_list.append(iteration)
                    setup_list.append(setup)
                    ts_list.append(ts_size)
                    error_list.append(error[i])
                    mols_list.append(aon.shape[0])
                    max_mols_list.append(max_mols[j])
                    lam_list.append(lam[k])
                    
                    tr_mean_list.append(regError2.iloc[0,0])
                    tr_median_list.append(regError2.iloc[0,1])
                    tr_sd_list.append(regError2.iloc[0,3])
                    tr_mad_list.append(regError2.iloc[0,4]) 
                    tr_max_list.append(regError2.iloc[0,5]) 
                    tr_min_list.append(regError2.iloc[0,6])
                    tr_range_list.append(regError2.iloc[0,7])                
            
                    # plot original data and AON model
                    ###moduleAON.plotAHN(yModel, regError)
                    temp = pd.DataFrame()
                    temp['log_index']=yModel['fitted_mnlr']
                    print('Temp 1')
                    ###moduleData.plotSet(temp)
                
                    print("----------------------------------------------------------")
                    print("                  Test Set")
                    print("----------------------------------------------------------")
                    print("\n")
                    principalComp3 = moduleAON.clusterTest2(principalComp3, aon)
                    #print(principalComp3)
                    #principalComp3 = moduleAON.clusterTest(principalComp, principalComp3, max_mols[j], rand)
                    # do prediction with model
                    [yModel, regError] = moduleAON.predict(principalComp3,aon,func,0,lam[k])
                    # print AON compound
                    moduleData.printDataSet(aon)
                    # AON summary
                    [modPerformance, regError2] = moduleAON.summary(yModel)
                    
                    ts_mean_list.append(regError2.iloc[0,0])
                    ts_median_list.append(regError2.iloc[0,1])
                    ts_sd_list.append(regError2.iloc[0,3])
                    ts_mad_list.append(regError2.iloc[0,4])      
                    ts_max_list.append(regError2.iloc[0,5])              
                    ts_min_list.append(regError2.iloc[0,6])
                    ts_range_list.append(regError2.iloc[0,7])
                        
                    best_list.append(best_individual)   
                    iteration += 1
                    
                    # plot original data and AON model
                    moduleAON.plotAHN(yModel, regError)
                    temp = pd.DataFrame()
                    temp['log_index']=yModel['fitted_mnlr']
                    print('IC pronostico')
                    moduleData.plotSet(temp)
                    print('IC datos originales')
                    temp = pd.DataFrame()
                    temp['log_index']=yModel['y']
                    ###moduleData.plotSet(temp)
                    
                    vato_loco = {"setup":setup_list,
                                 "ts size":ts_list,
                                 "error": error_list, 
                                 "mols used": mols_list,                 
                                 "max mols": max_mols_list,
                                 "lam": lam_list,
                                 "tr mean": tr_mean_list,
                                 "tr median": tr_median_list,
                                 "tr sd": tr_sd_list,                 
                                 "tr mad": tr_mad_list,
                                 "tr max": tr_max_list,
                                 "tr min": tr_min_list,
                                 "tr range": tr_range_list,                 
                                 "ts mean": ts_mean_list,
                                 "ts median": ts_median_list,
                                 "ts sd": ts_sd_list,
                                 "ts mad": ts_mad_list,
                                 "ts max": ts_max_list,
                                 "ts min": ts_min_list,
                                 "ts range": ts_range_list,
                                 "best individual": best_list}
                    mega_loco = pd.DataFrame(vato_loco, index=iter_list)
                    #mega_loco.to_csv("AON_analysis.csv")
                    mega_loco.to_csv("AON_analysis_art.csv")
                setup +=1

    print(mega_loco)
    print(mega_loco["ts mean"].min())
    print(mega_loco["ts mean"].argmin()+1)
    print("\n")
    print(mega_loco.loc[mega_loco["ts mean"].argmin()+1])
    print("\n")
    
    # --------------------
    # finance analysis
    [tr_set, ts_set] = moduleData.splitDataSet(normSet, test_size=ts_size, randSplit=rand)
    pre_dataFA1 = pd.DataFrame()
    pre_dataFA2 = pd.DataFrame()
    pre_dataFA1["index"] = 3.5**principalComp3["log_index"]
    pre_dataFA1["log_index"] = principalComp3["log_index"]
    pre_dataFA2["index"] = 3.5**yModel["fitted_mnlr"]
    pre_dataFA2["log_index"] = yModel["fitted_mnlr"]
    #print(yModel.rename(columns={"fitted_mnlr":'index'}))
    dataFA1 = moduleFA.priceReturn(pre_dataFA1)
    dataFA2 = moduleFA.priceReturn(pre_dataFA2)
    moduleFA.compoundReturn(dataFA1,ts_set['RFR'])
    moduleFA.compoundReturn(dataFA2,ts_set['RFR'])
    
    #[dataFA1, fmaLabel, smaLabel] = moduleFA.tradeStrat(dataFA1,10,30)
    #moduleData.printDataSet(dataFA1)
    #moduleData.plotStrat(dataFA1, fmaLabel, smaLabel)
       
    [dataFA2, fmaLabel, smaLabel] = moduleFA.tradeStrat(dataFA2,10,30)
    moduleData.printDataSet(dataFA2)
    moduleData.plotStrat(dataFA2, fmaLabel, smaLabel)

#"""
main()
