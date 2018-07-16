# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 02:27:37 2018

@author: johan
"""

cd "C:\Users\johan\MachineLearning\titanic comp"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sklearn
import re as re

datatest=pd.read_csv('test.csv', header=0, dtype={'Age':np.float32})
datatrain=pd.read_csv('train.csv', header=0, dtype={'Age':np.float32})
alldata=[datatrain, datatest]
for dataset in alldata:
    print(pd.isnull(dataset).sum()>0)

print(datatrain[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean())
print(datatrain[['Sex','Survived']].groupby(['Sex'],as_index=False).mean())

for dataset in alldata:
    dataset['Familysize'] = dataset['SibSp']+dataset['Parch']+1

print(datatrain[['Familysize','Survived']].groupby(['Familysize'],as_index=False).mean())

for dataset in alldata:
    dataset['Alone'] = 0
    dataset.loc[dataset['Familysize']==1,'Alone'] =1

print(datatrain[['Alone','Survived']].groupby(['Alone'],as_index=False).mean())

for dataset in alldata:
    dataset['Embarked']=dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
    
for dataset in alldata:
    dataset['Fare']=dataset['Fare'].fillna(datatrain['Fare'].median())

print(datatrain[['Pclass','Fare']].groupby(['Pclass'],as_index=False).mean())
Faresorted=datatrain['Fare'].sort_values()
Faresorted.plot(use_index=False)

for dataset in alldata:
    aveAge=dataset['Age'].mean()
    nulls=dataset['Age'].isnull().sum()
    print(nulls)
    dataset['Age'][np.isnan(dataset['Age'])]=aveAge    
    dataset['Age']=dataset['Age'].astype(int)
    
datatrain.dtypes

priceClassLower=[0,10,20,80]
AgeLower=[0,16,32,48,64]

datatrain['Title']="Special"
datatest['Title']="Special"

for dataset in alldata:
    ix=dataset["Name"].str.contains(r'Mr\. ')
    dataset['Title'][ix]="Mr"
    ix=dataset["Name"].str.contains(r'Mrs\. ')
    dataset['Title'][ix]="Mrs"
    ix=dataset["Name"].str.contains(r'Miss\. ')
    dataset['Title'][ix]="Miss"
    ix=dataset["Name"].str.contains(r'Master\. ')
    dataset['Title'][ix]="Master"
    ix=dataset["Name"].str.contains(r'Mme\. ')
    dataset['Title'][ix]="Mrs"
    ix=dataset["Name"].str.contains(r'Ms\. ')
    dataset['Title'][ix]="Miss"
    ix=dataset["Name"].str.contains(r'Mlle\. ')
    dataset['Title'][ix]="Miss"

print(datatrain[['Title','Survived']].groupby(['Title'],as_index=False).mean())
test=datatrain['Sex'][datatrain['Title']==1]

for dataset in alldata:
    dataset['Sex']=dataset['Sex'].fillna(0)
    dataset['Sex']=dataset['Sex'].map({'female':0, 'male':1}).astype(int)
    
    titlemapping = {"Special":0, "Miss":1,"Mrs":2,"Master":3,"Mr":4}
    dataset["Title"]=dataset["Title"].map(titlemapping)
    dataset["Title"]=dataset["Title"].fillna(0)
    
    dataset["Embarked"]=dataset["Embarked"].map({"S":0,"C":1,"Q":2})
    
    dataset.loc[dataset["Fare"]<=priceClassLower[1],"Fare"]=0
    dataset.loc[((dataset["Fare"]>priceClassLower[1]) & (dataset["Fare"] <=priceClassLower[2])),"Fare"]=1
    dataset.loc[((dataset["Fare"]>priceClassLower[2]) & (dataset["Fare"] <=priceClassLower[3])),"Fare"]=2
    dataset.loc[dataset["Fare"]>=priceClassLower[3],"Fare"]=3
    
    dataset.loc[dataset["Age"]<=AgeLower[1],"Age"]=0
    dataset.loc[((dataset["Age"]>AgeLower[1]) & (dataset["Age"] <=AgeLower[2])),"Age"]=1
    dataset.loc[((dataset["Age"]>AgeLower[2]) & (dataset["Age"] <=AgeLower[3])),"Age"]=2
    dataset.loc[((dataset["Age"]>AgeLower[3]) & (dataset["Age"] <=AgeLower[4])),"Age"]=3    
    dataset.loc[dataset["Age"]>=AgeLower[4],"Age"]=4    
    
dropelements = ["PassengerId","Name","Ticket","Cabin","SibSp","Parch","Familysize"]
datatrain=datatrain.drop(dropelements, axis=1)
datatest=datatest.drop(dropelements, axis=1)

datatrain["Fare"]=datatrain["Fare"].astype(int)
datatest["Fare"]=datatest["Fare"].astype(int)



def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

#for dataset in full_data:
#    dataset['Title'] = dataset['Name'].apply(get_title)

from sklearn.metrics import accuracy_score
#%% Testing
from sklearn.svm import SVC
classifier=SVC(probability=True)
X=datatrain.iloc[:,1:].as_matrix()
y=datatrain.iloc[:,0:1].as_matrix().ravel()

classifier.fit(X,y)
resultback=classifier.predict(X)
acc = accuracy_score(y, resultback)
print(acc)

result=classifier.predict(datatest.as_matrix())
print(result)



#%% Testing
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
X=datatrain.iloc[:,1:].as_matrix()
y=datatrain.iloc[:,0:1].as_matrix().ravel()

classifier.fit(X,y)
resultback=classifier.predict(X)
acc = accuracy_score(y, resultback)
print(acc)
result=classifier.predict(datatest.as_matrix())
print(result)

df_testadd = pd.DataFrame({'survived': pd.Series(result)})
dftest=pd.read_csv('test.csv', header=0, dtype={'Age':np.float32})
dftest.append(df_testadd)