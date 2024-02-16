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
import matplotlib.pyplot as plt # to plot
import seaborn as sns
import pandas as pd # DataFrame (table)

from scipy.stats import mannwhitneyu
from scipy import stats
import scikit_posthocs as sp


# my libraries
import moduleData

def main():
    warnings.filterwarnings("ignore")

    # --------------------
    # get results
    temp2 = pd.DataFrame()
    GA_mega_loco = pd.DataFrame(columns=['index','setup','ts mean'])
    
    # ['ipc', 'gspc', 'dax', 'dji', 'ftse', 'n225', 'ndx', 'cac']
    index_list = ['cac']
    for index in index_list:
        print("----------------------------------------------------------")
        print("                 ", index)
        print("----------------------------------------------------------")
        print("\n")
        dataSet_1, dataSet_2 = moduleData.getDataResults(index)

        table_2 = moduleData.analyzeDataGA(dataSet_2, index)
        print(table_2)
        print("\n") 
        
        temp2['index'] = table_2['index']
        temp2['setup'] = table_2['setup']
        temp2['ts mean'] = table_2['ts mean']
        GA_mega_loco=pd.concat([GA_mega_loco,temp2],axis=0)
        
        table_2.to_csv("table_2.csv")
        
    print("----------------------------------------------------------")
    print("                 ", 'Results')
    print("----------------------------------------------------------")

    GA_table = pd.pivot_table(GA_mega_loco, values='ts mean', index=['setup'],
                       columns=['index'], aggfunc=np.min)
   
    print(GA_mega_loco)
    print("\n") 
        
    df_ranked = GA_table.rank(method='dense',ascending=True).mean(axis=1).sort_values()
    print(df_ranked.head(10))
    best_7 = df_ranked.head(7)
    print("\n") 
    
    print(GA_table.head(50))
    print("\n") 
    print(GA_table.loc[1,:])
    print("\n")
    print(stats.friedmanchisquare(GA_table.loc[best_7.index[0],:],GA_table.loc[best_7.index[1],:],GA_table.loc[best_7.index[2],:],
                                  GA_table.loc[3,:],GA_table.loc[4,:],GA_table.loc[5,:],GA_table.loc[6,:]))
    print("\n")
    




main()



# --------------------
# temp
def funcTemp1():
    """
    temp
    """
       
    # --------------------
    # analyze data from topology
    #dataSet = moduleData.get_dataMean()
    #moduleData.analyzeData(dataSet)

    #dataSet = moduleData.get_dataMedian()
    #moduleData.analyzeData(dataSet)


    # --------------------   
    index = 'ipc'
    print("----------------------------------------------------------")
    print("                 ", index)
    print("----------------------------------------------------------")
    print("\n")
    dataSet_1, dataSet_2 = moduleData.getDataResults(index)
        
    table_1 = moduleData.analyzeDataTopology(dataSet_1, index)
    print(table_1)
    print("\n")
    
    table_2 = table_1.iloc[:,2:].sort_values(by=['ts size', 'error', 'max mols', 'lam'], ignore_index=True)
    print(table_2)
    print("\n")
    
    table_2.to_csv("table_chica.csv")
    table_3 = pd.read_csv('table_chica2.csv', header=0, index_col=0)
    print(table_3)
    print("\n")
    
    AHC_table = pd.pivot_table(table_3, values='ts mean', index=['setup'],
                               columns=['ts size'], aggfunc=np.min)
    print(AHC_table)
    print("\n")
    
    df_ranked = AHC_table.rank(method='dense',ascending=True).mean(axis=1).sort_values()
    print(df_ranked.head(10))
    
    print("\n")
    print(stats.friedmanchisquare(AHC_table.loc[30,:],AHC_table.loc[22,:],AHC_table.loc[1,:],AHC_table.loc[2,:],AHC_table.loc[29,:],AHC_table.loc[26,:],AHC_table.loc[25,:]))
    print("\n")
    
    plt.figure(figsize=(14, 7))
    fig = sns.heatmap(AHC_table.T, vmin=0.0004, vmax=0.004,
                cbar_kws = dict(use_gridspec=False,location="bottom"),
                cmap='RdYlGn_r', linewidth=1)
    fig.set_yticklabels(fig.get_yticklabels(), rotation=0)
    plt.title('AHC Relative Error', fontweight='bold')
    plt.xlabel('Setup')
    plt.ylabel('Test set size')
    plt.show()



# --------------------
# temp
def funcTemp2():
    """
    temp
    """
    
    # --------------------
    # get results
    temp = pd.DataFrame()
    temp2 = pd.DataFrame()
    AHC_mega_loco = pd.DataFrame(columns=['index','setup','ts mean'])
    GA_mega_loco = pd.DataFrame(columns=['index','setup','ts mean'])
    
       
    #['ipc', 'gspc', 'dax', 'dji', 'ftse', 'n225', 'ndx', 'cac'] 
    index_list = ['ipc', 'gspc', 'dax', 'dji', 'ftse', 'n225', 'ndx', 'cac']  
    for index in index_list:
        print("----------------------------------------------------------")
        print("                 ", index)
        print("----------------------------------------------------------")
        print("\n")
        dataSet_1, dataSet_2 = moduleData.getDataResults(index)
        
        table_1 = moduleData.analyzeDataTopology(dataSet_1, index)
        print(table_1)
        print("\n")
        
        temp['index'] = table_1['index']
        temp['setup'] = table_1['setup']
        temp['ts mean'] = table_1['ts mean']
        AHC_mega_loco=pd.concat([AHC_mega_loco,temp],axis=0)
        
        table_2 = moduleData.analyzeDataGA(dataSet_2, index)
        print(table_2)
        print("\n") 
        
        temp2['index'] = table_2['index']
        temp2['setup'] = table_2['setup']
        temp2['ts mean'] = table_2['ts mean']
        GA_mega_loco=pd.concat([GA_mega_loco,temp2],axis=0)
        
        table_2.to_csv("table_2.csv")
        
    print("----------------------------------------------------------")
    print("                 ", 'Results')
    print("----------------------------------------------------------")
    AHC_table = pd.pivot_table(AHC_mega_loco, values='ts mean', index=['setup'],
                       columns=['index'], aggfunc=np.min)
    GA_table = pd.pivot_table(GA_mega_loco, values='ts mean', index=['setup'],
                       columns=['index'], aggfunc=np.min)

    print(AHC_mega_loco)
    print("\n")     
    print(GA_mega_loco)
    print("\n") 
    
    statistic, p_value = mannwhitneyu(AHC_table['cac'], GA_table['cac'])
    print("Mann-Whitney U test statistic cac index:", statistic)
    print("p-value:", p_value)
    print("\n") 
    
    df_ranked = GA_table.rank(method='dense',ascending=True).mean(axis=1).sort_values()
    print(df_ranked.head(10))
    print("\n") 
    
    print(GA_table.tail(50))
    print("\n") 
    print(GA_table.loc[1,:])
    print("\n")
    print(stats.friedmanchisquare(GA_table.loc[61,:],GA_table.loc[34,:],GA_table.loc[64,:],GA_table.loc[65,:],GA_table.loc[22,:]))
    print("\n")
    
    df_ranked2 = AHC_table.rank(method='dense',ascending=True).mean(axis=1).sort_values()
    print(df_ranked2.head(10))
    print("\n") 
    
    print(AHC_table.head(50))
    print("\n") 
    print(AHC_table.loc[1,:])
    print("\n")
    print(stats.friedmanchisquare(AHC_table.loc[20,:],AHC_table.loc[4,:],AHC_table.loc[3,:],AHC_table.loc[19,:],AHC_table.loc[35,:]))
    print("\n")



    