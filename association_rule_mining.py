#-------------------------------------------------------------------------
# AUTHOR: William Armstrong
# FILENAME: association_rule_mining.py
# SPECIFICATION: Association Rule Mining
# FOR: CS 4200- Assignment #5
# TIME SPENT: 20 minutes
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

import string

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below

all_items = []

encoded_vals = []
for index, row in df.iterrows():
    labels = {}
    for j in range(0,7):
        curItem = row[j]
        if (str(curItem) != "nan"):
            labels[curItem] = 1
            if curItem not in all_items:
                all_items.append(curItem)
    encoded_vals.append(labels)
    
for dic in encoded_vals:
    for item in all_items:
        if item not in dic:
            dic[item] = 0

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

numRules = len(rules)
antec = (rules['antecedents'])
conseq = (rules['consequents'])
ant_supp = (rules['antecedent support'])
con_supp = (rules['consequent support'])
supp = (rules['support'])
conf = (rules['confidence'])
lift = (rules['lift'])
lev = (rules['leverage'])
conv = (rules['conviction'])

#iterate the rules data frame and print the apriori algorithm results by using the following format:

#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
#-->add your python code below

for i in range(0,numRules):
    curAnt = str(antec).splitlines()
    curSeq = str(conseq).splitlines()
    curSup = str(supp).splitlines()
    curConf = str(conf).splitlines()
    print(curAnt[i][6:] + "->" + curSeq[i][6:])
    print("Support: " + curSup[i][6:])
    print("Confidence: " + curConf[i][6:])
    
    consStr = curSeq[i][6:]
    while (consStr[0] == ' ' or consStr[0] == '('):
        consStr = consStr[1:]
    while (consStr[-1] == ')'):
        consStr = consStr[:-1]
        
    supportCount = 0
    for dic in encoded_vals:
        if dic[consStr] == 1:
            supportCount = supportCount + 1
    #print(total)
    
    prior = supportCount/len(encoded_vals)
    curConf = str(conf).splitlines()
    Conf = float(curConf[i][6:])
    
    print("Prior: " + str(prior))
    print("Gain in confidence: " + str(100*(Conf-prior)/prior))
    

#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#-->add your python code below

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()