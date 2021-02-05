# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 15:50:48 2020

@author: Nico
"""


### Covid Data

import pandas as pd
import numpy as np

df1 = pd.read_csv(r"C:\Users\Nico\Documents\Data\Datasets\covid.csv", sep = ",")

df1.head(5)

## EDA

df1.dtypes

del df1["id"]


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
        
dead_list2

df2 = df1
del df2["date_died"]

df2["died"] = dead_list2
df2.dtypes

## 
df2.to_csv(r'C:\Users\Nico\Documents\Data\covid_df2.csv')
dfx = pd.read_csv(r'C:\Users\Nico\Documents\Data\covid_df2.csv')
##

## Preparing the DF
## turning data into binary data

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

### Standardising the dataset
l= []
for a in check_df[0]:    
    l.append(a)
        
l2 = ['sex', 'patient_type', 'entry_date', 'date_symptoms', 'covid_res', 'died']

#for a in l2:
#    l.remove(a)
    
l = ['intubed', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', 'icu']

df_to_be_stand = dfx[l]
df_to_be_stand

df_not_to_be_stand = dfx[['sex', 'patient_type','entry_date', 'date_symptoms', 'covid_res', 'died']]

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

df4 = df3

## 
df4.to_csv(r'C:\Users\Nico\Documents\Data\covid_df4.csv')
df4 = pd.read_csv(r'C:\Users\Nico\Documents\Data\covid_df4.csv')
##

## DF4 is the fully cleaned and standardized DF
