# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 16:28:42 2017

@author: johan
"""

import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt

plt.style.use('ggplot')

#Lab6
#df = pd.DataFrame(np.random.randn(1000, 5), columns=['a', 'b', 'c', 'd', 'e'])
df=pd.read_csv('Datasets/wheat.data')
df=df.drop(labels=['id'],axis=1)
df.corr()

plt.imshow(df.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df.columns))]
plt.xticks(tick_marks, df.columns, rotation='vertical')
plt.yticks(tick_marks, df.columns)

plt.show()

#Lab5
'''
from pandas.tools.plotting import andrews_curves
df=pd.read_csv('Datasets/wheat.data')
df=df.drop(labels=['id'],axis=1)
plt.figure()
andrews_curves(df,'wheat_type')
plt.show()
'''
#Lab4
'''
from pandas.tools.plotting import parallel_coordinates
df=pd.read_csv('Datasets/wheat.data')
df=df.drop(labels=['id','area','perimeter'],axis=1)
plt.figure()
parallel_coordinates(df,'wheat_type')
plt.show()
'''
#Lab3
'''df=pd.read_csv('Datasets/wheat.data')
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.set_xlabel('Area')
ax.set_ylabel('Perimeter')
ax.set_zlabel('Asymmetry')
ax.scatter(df.area,df.perimeter,df.asymmetry,c='r',marker='.')

ax=fig.add_subplot(111,projection='3d')
ax.set_xlabel('Width')
ax.set_ylabel('Groove')
ax.set_zlabel('Length')
ax.scatter(df.width,df.groove,df.length,c='g',marker='*')

df.plot.scatter(x='compactness',y='width')

plt.show()
'''
#Lab2
'''
df=pd.read_csv('Datasets/wheat.data')
df.plot.scatter(x='area',y='perimeter')
df.plot.scatter(x='groove',y='asymmetry')
df.plot.scatter(x='width',y='compactness')
'''
# General parallel graphs
'''from sklearn.datasets import load_iris
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves
data=load_iris()
df=pd.DataFrame(data.data, columns=data.feature_names)
df['target_names']=[data.target_names[i] for i in data.target]

plt.figure()
#parallel_coordinates(df,'target_names')
andrews_curves(df,'target_names')
plt.show()
'''

#General 3d
'''
from mpl_toolkits.mplot3d import Axes3D
student_dataset=pd.read_csv('Datasets/students.data',index_col=0)
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.set_xlabel('Final Grade')
ax.set_ylabel('First Grade')
ax.set_zlabel('Daily Alcohol')

ax.scatter(student_dataset.G1,student_dataset.G3,student_dataset['Dalc'],c='r',marker='.')
plt.show()
'''
# General histogram
'''plt.style.use('ggplot')
student_dataset=pd.read_csv('Datasets/students.data',index_col=0)

myseries=student_dataset.G3
mydf=student_dataset[['G3','G2','G1']]

myseries.plot.hist(alpha=0.5)
mydf.plot.hist(alpha=0.5)
student_dataset.plot.scatter(x='G1', y='G3')
'''