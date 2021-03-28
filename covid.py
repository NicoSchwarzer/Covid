# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 23:00:56 2021

@author: Nico
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch


df1 = pd.read_csv(r"C:\Users\Nico\Documents\Data\Datasets\covid.csv", sep = ",")

## EDA

df1.describe()
df1.dtypes


df1.head(5)

df1 =  df1.drop("id", axis = 1).copy()

## changing values to dates
df1["entry_date"].head(5)


time_list = ["entry_date", "date_symptoms"]

for a in time_list:
    df1[a] = pd.to_datetime(df1[a], errors = 'coerce')

# NAT for still living 
# creating variable for dead patients

df1['date_died'] = df1['date_died'].astype(object).where(df1['date_died'].notnull(),np.nan)

dead_list2 = []

for i in df1["date_died"]:
    if i == '9999-99-99':
        dead_list2.append(0)
    else:
        dead_list2.append(1)
 
# checking if it worked
print(dead_list2[0:10])

# adding died column to DF
df1.died = dead_list2

df2 = df1.copy()

df2 =  df2.drop("date_died", axis = 1)

df2['died'] = dead_list2
df2.dtypes

## 
df2.to_csv(r'C:\Users\Nico\Documents\Data\covid_df2.csv')
dfx = pd.read_csv(r'C:\Users\Nico\Documents\Data\covid_df2.csv')
##

dfx.dtypes

## Preparing the DF
## turning data into binary data from 1-2 encoding

def into_binary(a):
    dfx[a] = dfx[a]-1    

col_list = list(dfx.columns)
col_l1 = []
col_l2 = []

for a in col_list:
    col_l1.append(max(dfx[a]))
    col_l2.append(min(dfx[a]))
    
check_df = pd.DataFrame(col_list)
check_df["col_l1"] = col_l1
check_df["col_l2"] = col_l2

one_two_vars = ["sex", "patient_type"]

for a in one_two_vars:
    into_binary(a)

# No missing values :D
dfx.isnull().values.any()
dfx.isna().sum()


### some more variable recoding 

df3 = dfx.copy()

# tobacco
df3.loc[df3.tobacco == 1, 'tobacco'] = 0
df3.loc[df3.tobacco == 2, 'tobacco'] = 0
df3.loc[df3.tobacco == 98, 'tobacco'] = 1

# cardiovascular
df3.loc[df3.cardiovascular == 1, 'cardiovascular'] = 0
df3.loc[df3.cardiovascular == 2, 'cardiovascular'] = 0
df3.loc[df3.cardiovascular == 98, 'cardiovascular'] = 1

# obesity
df3.loc[df3.obesity == 1, 'obesity'] = 0
df3.loc[df3.obesity == 2, 'obesity'] = 0
df3.loc[df3.obesity == 98, 'obesity'] = 1

# asthma
df3.loc[df3.asthma == 1, 'asthma'] = 0
df3.loc[df3.asthma == 2, 'asthma'] = 0
df3.loc[df3.asthma == 98, 'asthma'] = 1

# hypertension
df3.loc[df3.hypertension == 1, 'hypertension'] = 0
df3.loc[df3.hypertension == 2, 'hypertension'] = 0
df3.loc[df3.hypertension == 98, 'hypertension'] = 1

# inmsupr 
df3.loc[df3.inmsupr == 1, 'inmsupr'] = 0
df3.loc[df3.inmsupr == 2, 'inmsupr'] = 0
df3.loc[df3.inmsupr == 98, 'inmsupr'] = 1

# pneumonia 
df3.loc[df3.pneumonia == 1, 'pneumonia'] = 0
df3.loc[df3.pneumonia == 2, 'pneumonia'] = 0
df3.loc[df3.pneumonia == 98, 'pneumonia'] = 1

# renal_chronic
df3.loc[df3.renal_chronic == 1, 'renal_chronic'] = 0
df3.loc[df3.renal_chronic == 2, 'renal_chronic'] = 0
df3.loc[df3.renal_chronic == 98, 'renal_chronic'] = 1


### Data Viz ###
dfx.dtypes

# death by tobacco
dfx.tobacco

dfx2 = dfx.copy()

dfx2.loc[dfx2.tobacco == 1, 'Smoked'] = "no"
dfx2.loc[dfx2.tobacco == 2, 'Smoked'] = "sometimes"
dfx2.loc[dfx2.tobacco == 98, 'Smoked'] = "yes"
dfx2.Smoked = dfx2.Smoked.astype("category")

sns.barplot(x = 'Smoked', y = 'died', data = dfx2, ci = 68, capsize = 0.15)
plt.legend()


# cardiovascular 
dfx2.cardiovascular

dfx2.loc[dfx2.cardiovascular == 1, 'Cardio'] = "heathly"
dfx2.loc[dfx2.cardiovascular == 2, 'Cardio'] = "mostly healthy"
dfx2.loc[dfx2.cardiovascular == 98, 'Cardio'] = "unhealthy"
dfx2.Smoked = dfx2.Smoked.astype("category")

sns.barplot(x = 'Cardio', y = 'died', data = dfx2, ci = 68, capsize = 0.15)
plt.legend()

# correlation with age 
sns.jointplot("age", "died", data=df3, kind='reg');

### Standardising the dataset
l = []
for a in check_df[0]:    
    l.append(a)
        
l2 = ['sex', 'patient_type', 'entry_date', 'date_symptoms', 'covid_res', 'died']

#for a in l2:
#    l.remove(a)
    
l = ['intubed', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', 'icu']

df_to_be_stand = df3[l]
df_to_be_stand

df_not_to_be_stand = df3[['sex', 'patient_type','entry_date', 'date_symptoms', 'covid_res', 'died']]

## actually standardising
from sklearn import preprocessing

df_to_be_stand2 = pd.DataFrame(preprocessing.scale(df_to_be_stand))

df_to_be_stand2.columns = ['intubed', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', 'icu']

# putting them together

df3 = df_not_to_be_stand.merge(df_to_be_stand2, left_index=True, right_index=True)
df3

## 
df3.to_csv(r'C:\Users\Nico\Documents\Data\covid_df3.csv')
df3 = pd.read_csv(r'C:\Users\Nico\Documents\Data\covid_df3.csv')
##

## Date entry - date symptoms
df3['entry_date'] = pd.to_datetime(df3['entry_date'])
df3['date_symptoms'] = pd.to_datetime(df3['date_symptoms'])

df3['no_days'] = (df3['entry_date'] - df3['date_symptoms']).abs().dt.days

df3[['no_days', 'entry_date', 'date_symptoms']].head(5)


del df3["entry_date"]
del df3["date_symptoms"]
del df3["Unnamed: 0"]

df3.dtypes

df4 = df3.copy()

## 
df4.to_csv(r'C:\Users\Nico\Documents\Data\covid_df4.csv')
df4 = pd.read_csv(r'C:\Users\Nico\Documents\Data\covid_df4.csv')
##

## DF4 is the fully cleaned and standardized DF
