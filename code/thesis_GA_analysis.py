""" *****************

    Thesis Main module

    File name: thesis_data_top.py
    Related files: moduleData.py
    Created by: Enrique Gonzalez
    Date: 12 abr 2023

    Compilacion:
    Ejecucion:

***************** """

# --------------------
# import libraries
import warnings
import numpy as np # to use numpy arrays instead of lists
import pandas as pd # DataFrame (table)
from scipy.stats import wilcoxon

# my libraries
import moduleData
import moduleGA


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
    # preprocess data
    num_comp = 3
    preprocessData = moduleData.computePCA(normSet, pcaprocess=True, num_comp=num_comp)  #normSet

    # --------------------
    # split data
    rand = False
    ts_size = .15
    [principalComp, principalComp3] = moduleData.splitDataSet(preprocessData, test_size=ts_size, randSplit=rand)
    print(principalComp)
    print("\n")
    
    goper_probability = [.1, .3, .6]
    mut_probability =  [.25, .5, .75]
    pop_size = [650, 800]#[300, 500, 700]
    generation_limit = [25]#[25, 35]
        
    max_repeat = 50
    iteration = 1
    setup = 1

    iter_list = []
    setup_list = []
    ts_list = []
    goper_list = []
    mut_list = []
    pop_list = []
    gen_lim_list = []
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
    

    for k in range(len(generation_limit)):
        for j in range(len(pop_size)): 
            for i in range(len(mut_probability)):
                for h in range(len(goper_probability)):
                    for iter in range(max_repeat):
                    
                        print("iter: ", iteration, ", gen lim: ", generation_limit[k], ", pop: ", pop_size[j], ", mut_prob: ", mut_probability[i], ", goper_probability: ", goper_probability[h])
                        gen,  best_individual, modPerformance, regError, modPerformance2, regError2 = moduleGA.run_evolution(pop_size = pop_size[j], individual_size = 15, generation_limit = generation_limit[k], tournament_size = 4, num_xover = 1, mut_probability = mut_probability[i], goper_probability = goper_probability[h], train_Set = principalComp, test_Set = principalComp3)
                        print("total of generations: ", gen, "best individual: ", best_individual)
                        print("\n")
            
                        iter_list.append(iteration)
                        setup_list.append(setup)
                        ts_list.append(ts_size)
                        goper_list.append(goper_probability[h])
                        mut_list.append(mut_probability[i])
                        pop_list.append(pop_size[j])
                        gen_lim_list.append(generation_limit[k])
                        tr_mean_list.append(regError.iloc[0,0])
                        tr_median_list.append(regError.iloc[0,1])
                        tr_sd_list.append(regError.iloc[0,3])
                        tr_mad_list.append(regError.iloc[0,4])      
                        tr_max_list.append(regError.iloc[0,5])              
                        tr_min_list.append(regError.iloc[0,6])
                        tr_range_list.append(regError.iloc[0,7])                
                        ts_mean_list.append(regError2.iloc[0,0])
                        ts_median_list.append(regError2.iloc[0,1])
                        ts_sd_list.append(regError2.iloc[0,3])
                        ts_mad_list.append(regError2.iloc[0,4])      
                        ts_max_list.append(regError2.iloc[0,5])              
                        ts_min_list.append(regError2.iloc[0,6])
                        ts_range_list.append(regError2.iloc[0,7])
                        best_list.append(best_individual)              
                           
                        iteration += 1
                                            
                        vato_loco = {"setup":setup_list,
                                     "ts size":ts_list,
                                     "pop size": pop_list,
                                     "goper prob": goper_list,
                                     "mut prob": mut_list,
                                     "gen limit": gen_lim_list,
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
                        mega_loco.to_csv("GA_analysis.csv")
                    setup += 1    

    print(mega_loco)
    #print(mega_loco["tr mean"].min())
    #print(mega_loco["tr mean"].argmin())
    print("\n")
    print(mega_loco.loc[mega_loco["tr mean"].argmin()+1])
    
    statistic, p_value = wilcoxon(tr_mean_list)
    print("Wilcoxon signed-rank test statistic:", statistic)
    print("p-value:", p_value)


main()


