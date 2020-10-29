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

#######################
# Regression analysis #
#######################


# 1 linear regression 

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


X = df4.drop('died', axis = 1)
y = df4['died']


model1 = sm.OLS(y, X)
results = model1.fit()

sm.Logit()
print(results.summary())

# very strong impact factors, exepct for:
#tobacco, renal_chronic, other_disease, 

# LogReg

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

model2 = sm.Logit(y, X)
results = model2.fit()

print(results.summary())

# very strong impact factors, exepct for:
#tobacco, renal_chronic, obesity, other_disease, hypertension

ll = ['tobacco', 'renal_chronic', 'obesity', 'other_disease', 'hypertension' ]

df_small = df4

for a in ll:
    del df_small[a]
    
## DF to be used
df_small.to_csv(r'C:\Users\Nico\Documents\Data\covid_df_small.csv')
df_small = pd.read_csv(r'C:\Users\Nico\Documents\Data\covid_df_small.csv')
df4 = pd.read_csv(r'C:\Users\Nico\Documents\Data\covid_df4.csv')
##

#normal X,y
X = df4.drop('died', axis = 1)
y = df4['died']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

#smaller X,y
X_s = df_small.drop('died', axis = 1)
y_s = df_small['died']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=0.3, random_state=30)

## performing PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=8)


df5 = df4.drop('died', axis = 1)

pc = pca.fit_transform(df5)

df_p1 = pd.DataFrame(data = pc
             , columns = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'])


df_p = df_p1.merge(df4['died'], left_index=True, right_index=True)


df_p.to_csv(r'C:\Users\Nico\Documents\Data\covid_df_p.csv')
df_p = pd.read_csv(r'C:\Users\Nico\Documents\Data\covid_df_p.csv')

del df_p['Unnamed: 0']

# PCA normalizing

df_to_be_stand = df_p[['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']]


df_not_to_be_stand = pd.DataFrame(df_p['died'])

## actually normalize
from sklearn import preprocessing

def norm(df):
    dataN = ( (df - df.min() ) / (df.max() - df.min() ) )
    return dataN

p11 = ( (df_to_be_stand['p1'] - df_to_be_stand['p1'].min() )/ ( df_to_be_stand['p1'].max() - df_to_be_stand['p1'].min() ) )
p12 = ( (df_to_be_stand['p2'] - df_to_be_stand['p2'].min() )/ ( df_to_be_stand['p2'].max() - df_to_be_stand['p2'].min() ) )
p13 = ( (df_to_be_stand['p3'] - df_to_be_stand['p3'].min() )/ ( df_to_be_stand['p3'].max() - df_to_be_stand['p3'].min() ) )
p14 = ( (df_to_be_stand['p4'] - df_to_be_stand['p4'].min() )/ ( df_to_be_stand['p4'].max() - df_to_be_stand['p4'].min() ) )
p15 = ( (df_to_be_stand['p5'] - df_to_be_stand['p5'].min() )/ ( df_to_be_stand['p5'].max() - df_to_be_stand['p5'].min() ) )
p16 = ( (df_to_be_stand['p6'] - df_to_be_stand['p6'].min() )/ ( df_to_be_stand['p6'].max() - df_to_be_stand['p6'].min() ) )
p17 = ( (df_to_be_stand['p7'] - df_to_be_stand['p7'].min() )/ ( df_to_be_stand['p7'].max() - df_to_be_stand['p7'].min() ) )
p18 = ( (df_to_be_stand['p8'] - df_to_be_stand['p8'].min() )/ ( df_to_be_stand['p8'].max() - df_to_be_stand['p8'].min() ) )


p11 = pd.DataFrame(p11)
p11['p12'] = p12
p11['p13'] = p13
p11['p14'] = p14
p11['p15'] = p15
p11['p16'] = p16
p11['p17'] = p17
p11['p18'] = p18



# putting them together

df_pca = df_not_to_be_stand.merge(p11, left_index=True, right_index=True)


df_pca.to_csv(r'C:\Users\Nico\Documents\Data\covid_df_pca.csv')
df_pca = pd.read_csv(r'C:\Users\Nico\Documents\Data\covid_df_pca.csv')

##
del df_pca['Unnamed: 0']


X_p = np.array(df_pca.drop('died', axis = 1))
y_p = df_pca['died'].values


X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.3, random_state=30)


#############
## XGBoost ##
#############

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


##
xg = XGBClassifier()
xg.fit(X_train, y_train)

xg_pred = xg.predict(X_test)
predictions = [round(value) for value in xg_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

## 94%

## reduced dataset

xg_s = XGBClassifier()
xg_s.fit(X_train_s, y_train_s)

xg_pred_s = xg_s.predict(X_test_s)
predictions_s = [round(value) for value in xg_pred_s]

accuracy_s = accuracy_score(y_test_s, predictions_s)
print("Accuracy: %.2f%%" % (accuracy_s * 100.0))

## pca-df

xg_p = XGBClassifier()
xg_p.fit(X_train_p, y_train_p)

xg_pred_p = xg_p.predict(X_test_p)
predictions_p = [round(value) for value in xg_pred_p]

accuracy_p = accuracy_score(y_test_p, predictions_p)
print("Accuracy: %.2f%%" % (accuracy_p * 100.0))



#########
## NN  ##
#########

import torch

def to_torch(x):
    x1 = np.array(x)
    x2 = torch.from_numpy(x1)
    return x2

X_train_p = to_torch(X_train_p)
y_train_p = to_torch(y_train_p)
X_test_p = to_torch(X_test_p)
y_test_p = to_torch(y_test_p)


cols = X_train_p.shape[1]


model = torch.nn.Sequential(
    torch.nn.Linear(cols, 50),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(50, 50),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(50, 1))


lossBCE = torch.nn.BCELoss()

learning_rate = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

l_iterations = []
l_loss = [] 


X_test_p.max()
X_test_p.min()

y_train_p.max()
y_train_p.min()

' Y_pred norm


for i in range(3000):
    y_pred1 = model(X_train_p.float())
    
    # 0-1 normalization for BCE
    y_pred2 = ( (y_pred1 - y_pred1.min() )/ ( y_pred1.max() - y_pred1.min() ) )

    loss = lossBCE(y_pred2, y_train_p.float())
    
    l_iterations.append(i)
    l_loss.append(loss)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


import matplotlib.pyplot as plt

plt.plot(l_iterations, l_loss)

## Evaluating the model

y_pred3 = model(X_test.float())

y_pred3 = ( (y_pred3 - y_pred3.min() )/ ( y_pred3.max() - y_pred3.min() ) )

loss = loss_BCE(y_pred3, y_test)
loss



