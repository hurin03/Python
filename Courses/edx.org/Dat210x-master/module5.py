# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 06:44:40 2017

@author: johan
"""
#Lab8
#regression
'''
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model.score(X_test, y_test)
'''
#Lab7

#Lab6

#Lab5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("ggplot")

def plotDecisionBoundary(model, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    padding = 0.6
    resolution = 0.0025
    colors = ['royalblue','forestgreen','ghostwhite']

    # Calculate the boundaris
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    y_min -= y_range * padding
    x_max += x_range * padding
    y_max += y_range * padding

    # Create a 2D Grid Matrix. The values stored in the matrix
    # are the predictions of the class at at said location
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

    # What class does the classifier say?
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour map
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)

    # Plot the test original points as well...
    for label in range(len(np.unique(y))):
        indices = np.where(y == label)
        plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], label=str(label), alpha=0.8)

    p = model.get_params()
    plt.axis('tight')
    plt.title('K = ' + str(p['n_neighbors']))
    
df=pd.read_csv('Datasets/wheat.data', index_col=0)
y=df['wheat_type']
df = df.drop(labels=['wheat_type'], axis=1)

x=df.fillna(df.mean())

from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
labels = pd.get_dummies(y) 
data_train, data_test, label_train, label_test = train_test_split(x, labels, test_size=0.33, random_state=1)


Ntrain = preprocessing.Normalizer().fit_transform(data_train)
Ntest = preprocessing.Normalizer().fit_transform(data_test)
model_train = PCA(n_components=2, svd_solver='randomized', random_state=1)
model_train.fit(data_train)
model_train.transform(data_train)
xtrain=model_train.transform(data_train)

model_test = PCA(n_components=2, svd_solver='randomized', random_state=1)
model_test.fit(data_test)
xtest=model_test.transform(data_test)

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(xtrain, label_train) 

plotDecisionBoundary(knn, data_train, label_train)

#Lab4
'''
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

matplotlib.style.use('ggplot') # Look Pretty
c = ['red', 'green', 'blue', 'orange', 'yellow', 'brown']
PLOT_TYPE_TEXT = False    # If you'd like to see indices
PLOT_VECTORS = True       # If you'd like to see your original features in P.C.-Space

def drawVectors(transformed_features, components_, columns, plt):
    num_columns = len(columns)

    # This function will project your *original* feature (columns)
    # onto your principal component feature-space, so that you can
    # visualize how "important" each one was in the
    # multi-dimensional scaling

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ## Visualize projections

    # Sort each column by its length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Projected Features by importance:\n", important_features)

    ax = plt.axes()

    for i in range(num_columns):
        # Use an arrow to project each original feature as a
        # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75, zorder=600000)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75, zorder=600000)
        
    return ax
    
def doPCA(data, dimensions=2):
    model = PCA(n_components=dimensions, svd_solver='randomized', random_state=7)
    model.fit(data)
    return model    
    
def doKMeans(data, num_clusters=0):
    # TODO: Do the KMeans clustering here, passing in the # of clusters parameter
    # and fit it against your data. Then, return a tuple containing the cluster
    # centers and the labels.
    #
    # Hint: Just like with doPCA above, you will have to create a variable called
    # `model`, which will be a SKLearn K-Means model for this to work.
    model = KMeans(n_clusters=num_clusters)
    model.fit(data)
    
    return model.cluster_centers_, model.labels_

df=pd.read_csv('Datasets/Wholesale customers data.csv')
df = df.drop(labels=['Channel','Region'], axis=1)
df.describe()

drop = {}
for col in df.columns:
    # Bottom 5
    sort = df.sort_values(by=col, ascending=True)
    if len(sort) > 5: sort=sort[:5]
    for index in sort.index: drop[index] = True # Just store the index once

    # Top 5
    sort = df.sort_values(by=col, ascending=False)
    if len(sort) > 5: sort=sort[:5]
    for index in sort.index: drop[index] = True # Just store the index once

print("Dropping {0} Outliers...".format(len(drop)))
df.drop(inplace=True, labels=drop.keys(), axis=0)
df.describe()

T = preprocessing.StandardScaler().fit_transform(df)
#T = preprocessing.MinMaxScaler().fit_transform(df)
#T = preprocessing.MaxAbsScaler().fit_transform(df)
#T = preprocessing.Normalizer().fit_transform(df)
#T = df # No Change

# Do KMeans
n_clusters = 3
centroids, labels = doKMeans(T, n_clusters)

display_pca = doPCA(T)
T = display_pca.transform(T)
CC = display_pca.transform(centroids)

fig = plt.figure()
ax = fig.add_subplot(111)
if PLOT_TYPE_TEXT:
    # Plot the index of the sample, so you can further investigate it in your dset
    for i in range(len(T)): ax.text(T[i,0], T[i,1], df.index[i], color=c[labels[i]], alpha=0.75, zorder=600000)
    ax.set_xlim(min(T[:,0])*1.2, max(T[:,0])*1.2)
    ax.set_ylim(min(T[:,1])*1.2, max(T[:,1])*1.2)
else:
    # Plot a regular scatter plot
    sample_colors = [ c[labels[i]] for i in range(len(T)) ]
    ax.scatter(T[:, 0], T[:, 1], c=sample_colors, marker='o', alpha=0.2)
    
    
ax.scatter(CC[:, 0], CC[:, 1], marker='x', s=169, linewidths=3, zorder=1000, c=c)
for i in range(len(centroids)):
    ax.text(CC[i, 0], CC[i, 1], str(i), zorder=500010, fontsize=18, color=c[i])

# Display feature vectors for investigation:
if PLOT_VECTORS:
    drawVectors(T, display_pca.components_, df.columns, plt)

# Add the cluster label back into the dataframe and display it:
df['label'] = pd.Series(labels, index=df.index)
df

plt.show()    
'''
#Lab3
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv('Datasets/CDR.csv')
df['CallDate']=pd.to_datetime(df['CallDate'],errors='coerce')
df['CallTime']=pd.to_timedelta(df['CallTime'],errors='coerce')
df['Duration']=pd.to_timedelta(df['Duration'],errors='coerce')
df = df.dropna(axis=0)
df = df.reset_index(drop=True)
#%%
def clusterInfo(model):
    print("Cluster Analysis Inertia: ", model.inertia_)
    print('------------------------------------------')
    
    for i in range(len(model.cluster_centers_)):
        print("\n  Cluster ", i)
        print("    Centroid ", model.cluster_centers_[i])
        print("    #Samples ", (model.labels_==i).sum()) # NumPy Power
# Find the cluster with the least # attached nodes
def clusterWithFewestSamples(model):
    # Ensure there's at least on cluster...
    minSamples = len(model.labels_)
    minCluster = 0
    
    for i in range(len(model.cluster_centers_)):
        if minSamples > (model.labels_==i).sum():
            minCluster = i
            minSamples = (model.labels_==i).sum()

    print("\n  Cluster With Fewest Samples: ", minCluster)
    return (model.labels_==minCluster)
    

    
def doKMeans(data, num_clusters=0):
    # TODO: Be sure to only feed in Lat and Lon coordinates to the KMeans algo, since none of the other
    # data is suitable for your purposes. Since both Lat and Lon are (approximately) on the same scale,
    # no feature scaling is required. Print out the centroid locations and add them onto your scatter
    # plot. Use a distinguishable marker and color.
    #
    # Hint: Make sure you fit ONLY the coordinates, and in the CORRECT order (lat first). This is part
    # of your domain expertise. Also, *YOU* need to create, initialize (and return) the variable named
    # `model` here, which will be a SKLearn K-Means model for this to work:
    
    model = KMeans(n_clusters=num_clusters)
    model.fit(data)
    
    return model
   
In=df.In.unique().tolist()
u=0   
for u in range(10):
    user1=df[df.In==In[u]]   
    print("Examining person: ", u)       
    
    user1slice=user1[((user1.DOW!='Sat') & (user1.DOW!='Sun')) & (user1.CallTime<'17:00:00')]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(user1slice.TowerLon,user1slice.TowerLat, c='g', marker='o', alpha=0.2)
    ax.set_title('Weekend Calls (<5 pm)')
    
    
    kmeansdf=user1slice[['TowerLon','TowerLat']]
    K=3
    result=doKMeans(kmeansdf,K)
    
    clusterInfo(result)
    clusterWithFewestSamples(result)
    for i in range(K-1):
        ax.scatter(result.cluster_centers_[i,0], result.cluster_centers_[i,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
    
    plt.show()
    
    midWayClusterIndices = clusterWithFewestSamples(result)
    midWaySamples = user1slice[midWayClusterIndices]
    print("    Its Waypoint Time: ", midWaySamples.CallTime.mean())
    
    ax.scatter(result.cluster_centers_[:,1], result.cluster_centers_[:,0], s=169, c='r', marker='x', alpha=0.8, linewidths=2)
    ax.set_title('Weekday Calls Centroids')
    plt.show()
'''
#Lab2
'''
df=pd.read_csv('Datasets/CDR.csv')
df['CallDate']=pd.to_datetime(df['CallDate'],errors='coerce')
df['CallTime']=pd.to_timedelta(df['CallTime'],errors='coerce')
df['Duration']=pd.to_timedelta(df['Duration'],errors='coerce')
df = df.dropna(axis=0)
df = df.reset_index(drop=True)
#%%
#add unique persons in a list
In=df.In.unique().tolist()
#select data for first user
allhomes=[]
for u in range(len(In)-1):
    user1=df[df.In==In[u]]
    
    user1.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title='Call Locations')
    plt.show()
    
    #inbound calls on weekend
    user1weekendinbound=user1[((user1.DOW=='Sat') | (user1.DOW=='Sun')) & (user1.Direction=='Incoming')]
    
    user1weekendinboundNight=user1weekendinbound[(user1weekendinbound.CallTime>'22:00:00') | (user1weekendinbound.CallTime<'06:00:00')]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(user1weekendinboundNight.TowerLon,user1weekendinboundNight.TowerLat, c='g', marker='o', alpha=0.2)
    ax.set_title('Weekend Calls (<6am or >10p)')
    plt.show()
    
    kmeansdf=user1weekendinboundNight[['TowerLon','TowerLat']]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(kmeansdf.TowerLon, kmeansdf.TowerLat, marker='.', alpha=0.3)
    

    model = KMeans(n_clusters=2)
    model.fit(kmeansdf)
    
        # Now we can print and plot the centroids:
    centroids = model.cluster_centers_
    allhomes.append(centroids[0])
    print(centroids)
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
    plt.show()

'''
#%%
#Lab1

'''
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Datasets/Crimes2017.csv')

df['Date']=pd.to_datetime(df['Date'],errors='coerce')
#df = df.dropna(axis=0, thresh=4)
df = df.dropna(axis=0)
df = df.reset_index(drop=True)
def doKMeans(df):
    # Let's plot your data with a '.' marker, a 0.3 alpha at the Longitude,
    # and Latitude locations in your dataset. Longitude = x, Latitude = y
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=7)
    model.fit(df)

    # Now we can print and plot the centroids:
    centroids = model.cluster_centers_
    print(centroids)
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
    plt.show()
    
    
#filter date
df=df[df['Date'] > '2011-01-01']
dfloc=df[['Longitude','Latitude']]

doKMeans(dfloc)
'''
#Excercises
'''
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
'''