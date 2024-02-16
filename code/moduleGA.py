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
import numpy as np # to use numpy arrays instead of lists
import matplotlib.pyplot as plt # to plot
import seaborn as sns
import pandas as pd # DataFrame (table)
from math import floor
from random import sample, randint, random

def generate_data(coeff):
  """
  Creates data
  """
  x = [[random() for j in range(len(coeff))] for i in range(1000)]
  y = evaluate_f(coeff, x)
  
  return np.array(x), np.array(y)

def pre_evaluation(originalSet = 0):
    """
    Pre evaluate f
    """
    dataSet = pd.DataFrame()
    x1=1
    x2=2
    x3=3
    x4=4
    lam = 1e-10
    
    #dataSet["log_index"] = originalSet["log_index"]
    #dataSet['var9'] = lam*originalSet.iloc[:,x2]*originalSet.iloc[:,x3]
    #dataSet['var8'] = lam*originalSet.iloc[:,x1]*originalSet.iloc[:,x3]
    #dataSet['var7'] = lam*originalSet.iloc[:,x1]*originalSet.iloc[:,x2]
    #dataSet['var6'] = lam*originalSet.iloc[:,x3]**2
    #dataSet['var5'] = lam*originalSet.iloc[:,x2]**2
    #dataSet['var4'] = lam*originalSet.iloc[:,x1]**2
    #dataSet['var3'] = originalSet.iloc[:,x3]
    #dataSet['var2'] = originalSet.iloc[:,x2]
    #dataSet['var1'] = originalSet.iloc[:,x1]
    
    dataSet["log_index"] = originalSet["log_index"]
    dataSet['var14'] = lam*originalSet.iloc[:,x3]*originalSet.iloc[:,x4]
    dataSet['var13'] = lam*originalSet.iloc[:,x2]*originalSet.iloc[:,x4]
    dataSet['var12'] = lam*originalSet.iloc[:,x2]*originalSet.iloc[:,x3]
    dataSet['var11'] = lam*originalSet.iloc[:,x1]*originalSet.iloc[:,x4]
    dataSet['var10'] = lam*originalSet.iloc[:,x1]*originalSet.iloc[:,x3]
    dataSet['var9'] = lam*originalSet.iloc[:,x1]*originalSet.iloc[:,x2]
    dataSet['var8'] = lam*originalSet.iloc[:,x4]**2
    dataSet['var7'] = lam*originalSet.iloc[:,x3]**2
    dataSet['var6'] = lam*originalSet.iloc[:,x2]**2
    dataSet['var5'] = lam*originalSet.iloc[:,x1]**2
    dataSet['var4'] = originalSet.iloc[:,x4]
    dataSet['var3'] = originalSet.iloc[:,x3]
    dataSet['var2'] = originalSet.iloc[:,x2]
    dataSet['var1'] = originalSet.iloc[:,x1]
     
    return dataSet

def evaluate_f2(eq_coeff, dataSet):
    """
    Evaluate f
    """
    coeff = pd.DataFrame(eq_coeff)
    yModel = pd.DataFrame()
    lam = 1e-10
    
    #yModel['fitted'] = (coeff.iloc[9,0]*dataSet['var9']
    #                    +coeff.iloc[8,0]*dataSet['var8']
    #                    +coeff.iloc[7,0]*dataSet['var7']
    #                    +coeff.iloc[6,0]*dataSet['var6']
    #                    +coeff.iloc[5,0]*dataSet['var5']
    #                    +coeff.iloc[4,0]*dataSet['var4']
    #                    +coeff.iloc[3,0]*dataSet['var3']
    #                    +coeff.iloc[2,0]*dataSet['var2']
    #                    +coeff.iloc[1,0]*dataSet['var1']
    #                    +coeff.iloc[0,0])
    
    yModel['fitted'] = (coeff.iloc[14,0]*lam*dataSet['var14']
                        +coeff.iloc[13,0]*lam*dataSet['var13']
                        +coeff.iloc[12,0]*lam*dataSet['var12']
                        +coeff.iloc[11,0]*lam*dataSet['var11']
                        +coeff.iloc[10,0]*lam*dataSet['var10']                
                        +coeff.iloc[9,0]*lam*dataSet['var9']
                        +coeff.iloc[8,0]*lam*dataSet['var8']
                        +coeff.iloc[7,0]*lam*dataSet['var7']
                        +coeff.iloc[6,0]*lam*dataSet['var6']
                        +coeff.iloc[5,0]*lam*dataSet['var5']
                        +coeff.iloc[4,0]*dataSet['var4']
                        +coeff.iloc[3,0]*dataSet['var3']
                        +coeff.iloc[2,0]*dataSet['var2']
                        +coeff.iloc[1,0]*dataSet['var1']
                        +coeff.iloc[0,0])

    return yModel

def evaluate_f(eq_coeff, dataSet):
    """
    Evaluate f
    """
    coeff = pd.DataFrame(eq_coeff)
    yModel = pd.DataFrame()
    x1=1
    x2=2
    x3=3
    x4=4
    lam = 1e-10
        
    yModel['fitted'] = (coeff.iloc[14,0]*lam*dataSet.iloc[:,x3]*dataSet.iloc[:,x4]
                        +coeff.iloc[13,0]*lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x4]
                        +coeff.iloc[12,0]*lam*dataSet.iloc[:,x2]*dataSet.iloc[:,x3]
                        +coeff.iloc[11,0]*lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x4]
                        +coeff.iloc[10,0]*lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x3]              
                        +coeff.iloc[9,0]*lam*dataSet.iloc[:,x1]*dataSet.iloc[:,x2]
                        +coeff.iloc[8,0]*lam*dataSet.iloc[:,x4]**2
                        +coeff.iloc[7,0]*lam*dataSet.iloc[:,x3]**2
                        +coeff.iloc[6,0]*lam*dataSet.iloc[:,x2]**2
                        +coeff.iloc[5,0]*lam*dataSet.iloc[:,x1]**2
                        +coeff.iloc[4,0]*dataSet.iloc[:,x4]
                        +coeff.iloc[3,0]*dataSet.iloc[:,x3]
                        +coeff.iloc[2,0]*dataSet.iloc[:,x2]
                        +coeff.iloc[1,0]*dataSet.iloc[:,x1]
                        +coeff.iloc[0,0])

    return yModel

def generate_genome(length: int):
  """
  Creates an individual 
  """
  space = 12
  size = ((1,length))
  genome = np.random.uniform(-space,space,size)[0].tolist()
  return genome

def generate_population(size: int, genome_length: int):
  """
  create population
  """
  return [generate_genome(genome_length) for _ in range(size)]

def get_fitness(individual, dataSet):
  """
  Check fitness
  """
  y_hat = pd.DataFrame()
  
  y_hat = evaluate_f(individual, dataSet)
  y_hat['residuals'] = dataSet['log_index'] - y_hat['fitted']
  error = (1-(y_hat['fitted']/dataSet['log_index'])).abs().mean(axis=0)
    
  return error

def selection(population, dataSet):    
    """
    Performs selection
    """    
    df = pd.DataFrame(population)
    df["val"] = [get_fitness(gene,dataSet) for gene in population]
    best_individual = df[df.val == df.val.min()]

    return best_individual

def tournament(population, tournament_size, dataSet):
  """
  Performs tournament
  """
  sub_population = sample(population, tournament_size)
  best_individual = selection(sub_population, dataSet)
  parent = best_individual.iloc[0:1,0:-1].values[0].tolist()
  
  return parent

def single_point_crossover(genome_a, genome_b):    
    """
    Performs single crossover
    """
    length = len(genome_a)
    if length < 2:
        return genome_a, genome_b
    p = randint(1, length - 1)

    return genome_a[0:p] + genome_b[p:], genome_b[0:p] + genome_a[p:]

def multiple_point_xover(genome_a, genome_b, points):  
    """
    Performs multiple crossover
    """
    length = len(genome_a)
    if points >= length:
      print("the number of xover poinst should be smaller than the length of the individual")
      return

    if length < 2:
        return genome_a, genome_b
    
    x_points = sorted(sample(range(1,length),points))
    new_genome_a = genome_a[0:x_points[0]]
    new_genome_b = genome_b[0:x_points[0]]

    for i in range(points):
        if i % 2 ==0:
            try:
                new_genome_a += genome_b[x_points[i]:x_points[i+1]]
                new_genome_b += genome_a[x_points[i]:x_points[i+1]]
            except:
                 new_genome_a += genome_b[x_points[i]:]
                 new_genome_b += genome_a[x_points[i]:]
        else:
            try:
                 new_genome_a += genome_a[x_points[i]:x_points[i+1]]
                 new_genome_b += genome_b[x_points[i]:x_points[i+1]]
            except:
                 new_genome_a += genome_a[x_points[i]:]
                 new_genome_b += genome_b[x_points[i]:]

    return new_genome_a, new_genome_b

def mutation(genome, genome_length, mut_probability):   
    """
    Performs mutation
    """   
        
    size = (1, genome_length)
    
    new_genome = generate_genome(genome_length)
    
    
    mask_seed = np.random.uniform(0,1,size)[0]
    mask = ((mask_seed>=mut_probability)*1).tolist()
    not_mask = ((mask_seed<mut_probability)*1).tolist()
    
    result = (np.array(genome)*np.array(mask)).tolist()
    result2 = (np.array(new_genome)*np.array(not_mask)).tolist()

    return (np.array(result) + np.array(result2)).tolist()

def dataSub(dataSet = 0):
    """
    Substitute dataSet
    """
    dataSet_AR = pd.DataFrame()
    
    dataSet_AR["log_index"] = dataSet["log_index"]
    dataSet_AR["t-1"] = dataSet["log_index"].shift(1,fill_value=dataSet.iloc[0,0])
    dataSet_AR["t-2"] = dataSet["log_index"].shift(2,fill_value=dataSet.iloc[0,0])
    dataSet_AR["P. Comp. 1"] = dataSet["P. Comp. 1"]
    dataSet_AR["P. Comp. 2"] = dataSet["P. Comp. 2"]
    #dataSet_AR = dataSet_AR.fillna(0)
    #print(dataSet_AR)
    
    return dataSet_AR

def evolutionSummary(dataSet = 0, yModel = 0):
    """
    Results summary

    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 25
    pd.set_option('display.max_columns', 10)
    pd.set_option.precision = 4
    modPerformance = pd.DataFrame(index=['Model Performance'])
    regError = pd.DataFrame(index=['Relative Error'])
    
    # compute residuals and relative error
    yModel['residuals'] = dataSet['log_index'] - yModel['fitted']
    yModel['Rel Error'] = (1-(yModel['fitted']/dataSet['log_index'])).abs()

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
    modPerformance['SSR'] = ((yModel['fitted']-dataSet['log_index'].mean(axis=0))**2).sum()
    modPerformance['TSS'] = modPerformance['RSS']+modPerformance['SSR']
    modPerformance['R Square'] = 1-modPerformance['RSS']/modPerformance['TSS']
    print("==========================================================")
    print("                 MODEL GA PERFORMANCE")
    print("----------------------------------------------------------")
    print(modPerformance)
    print("----------------------------------------------------------")
    print(regError)
    print("\n")
            
    plt.figure(figsize=(15, 6))
    plt.plot(dataSet['log_index'],color='navy',label='Raw data')
    plt.plot(yModel['fitted'], color='r',linestyle='dashed',label="GA prediction")
    plt.grid(color='b',linestyle='dashed')
    plt.gcf().autofmt_xdate()
    plt.title('Index Forecast', fontweight='bold')
    plt.xlabel('Date (t)')
    plt.ylabel('Index (log)')
    plt.legend(loc='lower right')
    plt.show()
    
    return modPerformance, regError
    

def run_evolution(pop_size = 5, individual_size = 10, generation_limit = 100, tournament_size = 3, num_xover = 1, mut_probability = .5, goper_probability = .1, train_Set = 0, test_Set = 0):
    """
    GA Evolution
    """
    #pd.set_option("display.max_rows", 10, "display.max_columns", None,'precision', 6)
    pd.set_option.max_rows = 10
    pd.set_option.precision = 6
    historic_err = pd.DataFrame(columns=['Generation', 'Error'])
    
    # check that parameters are correct
    if pop_size <= tournament_size:
        raise ValueError("Tournament size must be smaller then population size")

    if individual_size <= num_xover:
        raise ValueError("Number of crossover must be smaller then individual size")

    if mut_probability >= 1:
        raise ValueError("Mutation probability must be smaller then 1")
    
    # add columns of historic data of y
    trainSet = dataSub(train_Set)
    testSet = dataSub(test_Set)

    # pre evaluate data set  
    #print("antes del mamadero")
    #print(trainSet)
    #print("\n")     
    #dataSet = pre_evaluation(trainSet)
    #dataSet2 = pre_evaluation(testSet)
    #print("despues del mamadero")
    #print(dataSet)
    #print("\n")
    
    historic_err.loc[0,'Generation'] = 0
    historic_err.loc[0, 'Error'] = 1
    
    total_couples = floor((pop_size-1)/2)
    
    population = generate_population(pop_size, individual_size)
    best_individual = selection(population, trainSet) 
    best_ind = best_individual.iloc[0:1,0:-1].values[0].tolist()

    for gen in range(generation_limit):
        print("generation: ", gen)
        print("best individual: ", best_ind)
        fitness_val = best_individual["val"].values[0]
        print("best individual Rel Error: ", round(fitness_val, 4))
        print('preparando para generar nueva gen')
        #print("\n")
        
        historic_err.loc[gen,'Num Mol'] = gen
        historic_err.loc[gen, 'Error'] = fitness_val

        if fitness_val < 0.001:
          break
        
        next_generation = [best_ind]
        
        for num_couples in range(total_couples):
          #print("tournament size: ", tournament_size)
          parent_a = tournament(population, tournament_size, trainSet)
          parent_b = tournament(population, tournament_size, trainSet)
          #print('ready tournament')
          #print("parents: ", parent_a, parent_b)
          
          probability = goper_probability
          
          if random() < probability or total_couples - num_couples == 1:
              offspring_a = mutation(parent_a, individual_size, mut_probability)
              offspring_b = mutation(parent_b, individual_size, mut_probability)
              #print("desp mut: ", offspring_a,offspring_b)
              #print('ready mut')
          else:
              offspring_a, offspring_b = multiple_point_xover(parent_a, parent_b, num_xover)
              #print("desp cross: ", offspring_a, offspring_b)
              #print('ready cross')
         
          next_generation += [offspring_a, offspring_b] 
          #print("next: ", next_generation)
          #print('ready next')
        
        population = next_generation
        best_individual = selection(population, trainSet) 
        best_ind = best_individual.iloc[0:1,0:-1].values[0].tolist()
        print('lista nueva gen')
        print("\n")

    gen = gen + 1
    print("generation: ", gen)
    print("best individual: ", best_ind)
    fitness_val = best_individual["val"].values[0]
    yModel = evaluate_f(best_ind, trainSet)
    print("final Rel Error: ", round(fitness_val, 4))
    print("\n")    
    
    plt.figure(figsize=(15, 6))
    plt.plot(historic_err['Error'], color='darkorange',label='Relative Error', linestyle='--')
    plt.axhline(historic_err['Error'].min(), color='red',linestyle='dashed')
    plt.grid(color='b',linestyle='dashed')
    plt.title('Relative Error Development', fontweight='bold')
    plt.xlabel('Generations')
    plt.ylabel('Relative Error')
    plt.legend(loc='upper right')
    plt.show()
    
    modPerformance, regError = evolutionSummary(trainSet, yModel)    
          
    yModel2 = evaluate_f(best_ind, testSet)
    modPerformance2, regError2 = evolutionSummary(testSet, yModel2)
  

    return gen, best_ind, modPerformance, regError, modPerformance2, regError2