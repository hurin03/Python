# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:27:54 2017

@author: johan
"""

import pandas as pd

#### Assignment 2.5
df=pd.read_csv('Datasets/census.data',header=None,names=['education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification'],na_values='?')
df.dtypes
df['capital-gain'].unique()

#### END Assignment 2.5

#### Assignment 2.4
'''tbl=pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2',header=1)
df=tbl[0] # only one list returned
#df.columns=['RK','Player','Team','GP','G','A','PTS','Change','PIM','Pts per G','SOG','PCT','GWG','PP-G','PP-A','SH-G','SH-A']

df = df.dropna(axis=0, thresh=4)
df['GP']=pd.to_numeric(df['GP'], errors='coerce')
df['G']=pd.to_numeric(df['G'], errors='coerce')
df['A']=pd.to_numeric(df['A'], errors='coerce')
df['PTS']=pd.to_numeric(df['PTS'], errors='coerce')
df['+/-']=pd.to_numeric(df['+/-'], errors='coerce')
df['PIM']=pd.to_numeric(df['PIM'], errors='coerce')
df['PTS/G']=pd.to_numeric(df['PTS/G'], errors='coerce')
df['SOG']=pd.to_numeric(df['SOG'], errors='coerce')
df['PCT']=pd.to_numeric(df['PCT'], errors='coerce')
df['GWG']=pd.to_numeric(df['GWG'], errors='coerce')
df['G.1']=pd.to_numeric(df['G.1'], errors='coerce')
df['A.1']=pd.to_numeric(df['A.1'], errors='coerce')
df['G.2']=pd.to_numeric(df['G.2'], errors='coerce')
df['A.2']=pd.to_numeric(df['A.2'], errors='coerce')
df = df.dropna(axis=0, thresh=4)
df = df.reset_index(drop=True)
df = df.drop(labels=['RK'], axis=1)


len(df['PCT'].unique())

df.loc[15:16,'GP'].sum()
'''
#### END Assignment 2.4

# 2.3 and other
#df=pd.read_csv('Datasets/Servo.data',header=None,names=['motor', 'screw', 'pgain', 'vgain', 'class'])
#df
'''

df=pd.read_csv('Datasets/direct_marketing.csv')
print(df.loc[:,'recency'])
df[(df['recency']<7)&(df['newbie']==0)]
'''