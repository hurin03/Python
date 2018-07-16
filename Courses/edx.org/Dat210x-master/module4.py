# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 07:44:38 2017

@author: johan
"""
#Lab5
import pandas as pd
from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
plt.style.use('ggplot')

samples=[]
for imagepath in glob.glob("Datasets/ALOI/32/*.png"):
    samples.append(misc.imread(imagepath))
#im=np.asarray(samples)
#print(im.shape)
df=pd.DataFrame(samples)

#Lab4
'''
import math,random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io

from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')

def Plot2D(T, title, x, y, num_to_plot=40):
    # This method picks a bunch of random samples (images in your case)
    # to plot onto the chart:
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Component: {0}'.format(x))
    ax.set_ylabel('Component: {0}'.format(y))
    
    x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
    y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
    
    for i in range(num_to_plot):
        img_num = int(random.random() * num_images)
        x0, y0 = T[img_num,x]-x_size/2., T[img_num,y]-y_size/2.
        x1, y1 = T[img_num,x]+x_size/2., T[img_num,y]+y_size/2.
        img = df.iloc[img_num,:].reshape(num_pixels, num_pixels)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    # It also plots the full scatter:
    ax.scatter(T[:,x],T[:,y], marker='.',alpha=0.7)
    
mat = scipy.io.loadmat('Datasets/face_data.mat')
df = pd.DataFrame(mat['images']).T
num_images, num_pixels = df.shape
num_pixels = int(math.sqrt(num_pixels))

# Rotate the pictures, so we don't have to crane our necks:
for i in range(num_images):
    df.loc[i,:] = df.loc[i,:].reshape(num_pixels, num_pixels).T.reshape(-1)

# assignment: implement pca
from sklearn.decomposition import PCA
pca=PCA(n_components=3,svd_solver='full')
pca.fit(df)
PCA(copy=True,n_components=2,whiten=False)
T=pca.transform(df)

Plot2D(T, 'PCA', 0, 1, num_to_plot=40)

# assignment: implement isomap
from sklearn import manifold
iso = manifold.Isomap(n_neighbors=8, n_components=3)
iso.fit(df) 
#Isomap(eigen_solver='auto', max_iter=None, n_components=3, n_neighbors=4,
#    neighbors_algorithm='auto', path_method='auto', tol=0)

manifold = iso.transform(df)

Plot2D(manifold, 'Isomap', 1, 2, num_to_plot=40)
'''




#Lab3
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math

from sklearn import preprocessing
plt.style.use('ggplot')


df=pd.read_csv('Datasets/kidney_disease.csv',na_values='?',index_col=0)
dforig=df
#experimented with this one
#df = df.drop(labels=['classification', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'], axis=1)
df=df.drop(labels=['classification'], axis=1)
df['wc']=pd.to_numeric(df['wc'], errors='coerce')
df['rc']=pd.to_numeric(df['rc'], errors='coerce')
df['pcv']=pd.to_numeric(df['pcv'], errors='coerce')
df=df.dropna()
df = df.reset_index(drop=True)
df.dtypes

def scaleFeaturesDF(df):
    # Feature scaling is a type of transformation that only changes the
    # scale, but not number of features. Because of this, we can still
    # use the original dataset's column names... so long as we keep in
    # mind that the _units_ have been altered:

    scaled = preprocessing.StandardScaler().fit_transform(df)
    scaled = pd.DataFrame(scaled, columns=df.columns)
    
    print("New Variances:\n", scaled.var())
    print("New Describe:\n", scaled.describe())
    return scaled

def drawVectors(transformed_features, components_, columns, plt, scaled):
    if not scaled:
        return plt.axes() # No cheating ;-)

    num_columns = len(columns)

    # This funtion will project your *original* feature (columns)
    # onto your principal component feature-space, so that you can
    # visualize how "important" each one was in the
    # multi-dimensional scaling

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ## visualize projections

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Features by importance:\n", important_features)

    ax = plt.axes()

    for i in range(num_columns):
        # Use an arrow to project each original feature as a
        # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax

scaleFeatures = True


labels = ['red' if i=='ckd' else 'green' for i in dforig.classification]

df_subset=df
df_subset=pd.get_dummies(df_subset)
df_subset.describe()

if scaleFeatures: df_subset = scaleFeaturesDF(df_subset)


from sklearn.decomposition import PCA
pca=PCA(n_components=2,svd_solver='full')
pca.fit(df_subset)
PCA(copy=True,n_components=2,whiten=False)
T=pca.transform(df_subset)

# Since we transformed via PCA, we no longer have column names; but we know we
# are in `principal-component` space, so we'll just define the coordinates accordingly:
ax = drawVectors(T, pca.components_, df_subset.columns.values, plt, scaleFeatures)
T  = pd.DataFrame(T)

T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)

plt.show()

'''

#Lab2
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math

from sklearn import preprocessing
plt.style.use('ggplot')

def scaleFeaturesDF(df):
    # Feature scaling is a type of transformation that only changes the
    # scale, but not number of features. Because of this, we can still
    # use the original dataset's column names... so long as we keep in
    # mind that the _units_ have been altered:

    scaled = preprocessing.StandardScaler().fit_transform(df)
    scaled = pd.DataFrame(scaled, columns=df.columns)
    
    print("New Variances:\n", scaled.var())
    print("New Describe:\n", scaled.describe())
    return scaled

def drawVectors(transformed_features, components_, columns, plt, scaled):
    if not scaled:
        return plt.axes() # No cheating ;-)

    num_columns = len(columns)

    # This funtion will project your *original* feature (columns)
    # onto your principal component feature-space, so that you can
    # visualize how "important" each one was in the
    # multi-dimensional scaling

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ## visualize projections

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Features by importance:\n", important_features)

    ax = plt.axes()

    for i in range(num_columns):
        # Use an arrow to project each original feature as a
        # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax

scaleFeatures = True

df=pd.read_csv('Datasets/kidney_disease.csv')

df = df.drop(labels=['id'], axis=1)
df['wc']=pd.to_numeric(df['wc'], errors='coerce')
df['rc']=pd.to_numeric(df['rc'], errors='coerce')
df=df.dropna()
df = df.reset_index(drop=True)
labels = ['red' if i=='ckd' else 'green' for i in df.classification]

df_subset=df[['bgr','wc','rc']]

df_subset.describe()

if scaleFeatures: df_subset = scaleFeaturesDF(df_subset)


from sklearn.decomposition import PCA
pca=PCA(n_components=2,svd_solver='full')
pca.fit(df_subset)
PCA(copy=True,n_components=2,whiten=False)
T=pca.transform(df_subset)

# Since we transformed via PCA, we no longer have column names; but we know we
# are in `principal-component` space, so we'll just define the coordinates accordingly:
ax = drawVectors(T, pca.components_, df_subset.columns.values, plt, scaleFeatures)
T  = pd.DataFrame(T)

T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)

plt.show()
'''
#Lab1
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

from plyfile import PlyData, PlyElement

plt.style.use('ggplot')

reducefactor=100

plyfile=PlyData.read('Datasets/stanford_armadillo.ply')

armadillo=pd.DataFrame({
    'x':plyfile['vertex']['z'][::reducefactor],
    'y':plyfile['vertex']['x'][::reducefactor],
    'z':plyfile['vertex']['y'][::reducefactor],
})

def do_PCA(df, slvr):
    from sklearn.decomposition import PCA
    pca=PCA(n_components=2,svd_solver=slvr)
    pca.fit(df)
    PCA(copy=True,n_components=2,whiten=False)
    df_out=pca.transform(df)
    return df_out

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.set_title("original")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(armadillo.x,armadillo.y,armadillo.z,c='g',marker='.',alpha=0.75)

pca=do_PCA(armadillo, 'full')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Full PCA')
ax.scatter(pca[:,0], pca[:,1], c='blue', marker='.', alpha=0.75)
plt.show()

rpca = do_PCA(armadillo, 'randomized')    

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Randomized PCA')
ax.scatter(rpca[:,0], rpca[:,1], c='red', marker='.', alpha=0.75)
plt.show()
'''

#excercises

'''
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
pca.fit(df)
PCA(copy=True, n_components=2, whiten=False)

T = pca.transform(df)

df.shape
(430, 6) # 430 Student survey responses, 6 questions..

T.shape
(430, 2) # 430 Student survey responses, 2 principal components..
'''