######################################################################
# Python codes for Anomaly Detection Lecture 
#
# Data Mining Course, Fall 2021
# School Of Information Technology
# Halmstad University
#
# Hadi Fanaee, Ph.D., Assistant Professor
# hadi.fanaee@hh.se
# www.fanaee.com
######################################################################


#**********************************************************************
#Slide-23: Boxplot
#**********************************************************************

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("weight-height.csv")
plt.boxplot(df['Height'])


#**********************************************************************
#Slide-26: Histogram
#**********************************************************************

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("weight-height.csv")
plt.hist(df['Height'])
hist, bin_edges = np.histogram(df['Height'].to_numpy(), density=False, bins = 24)



#**********************************************************************
#Slide-29: Gaussian Model 
#**********************************************************************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.stats import norm
df=pd.read_csv("weight-height.csv")
h=df['Height'].to_numpy()
mu, std = norm.fit(h)
plt.hist(h, bins=24, density=True, color='w')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)



#**********************************************************************
#Slide-34: Gaussian Mixture Model 
#**********************************************************************

import pandas as pd
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import matplotlib.pyplot as plt
df=pd.read_csv("weight-height.csv")
h=df['Height'].to_numpy()
hp=h.reshape(-1, 1)
gmm = GaussianMixture(n_components = 2).fit(hp)
plt.figure()
plt.hist(hp, bins=24,  density=True)
plt.xlim(0, 360)
f_axis = hp.copy().ravel()
f_axis.sort()
a = []
for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_):
    a.append(weight*norm.pdf(f_axis, mean, np.sqrt(covar)).ravel())
    plt.plot(f_axis, a[-1])
plt.plot(f_axis, np.array(a).sum(axis =0), 'k-')
plt.xlabel('Height')
plt.ylabel('PDF')
plt.tight_layout()
plt.show()


#**********************************************************************
#Slide-41: Marginal Boxplot 
#**********************************************************************

from pandas import read_csv
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
df=pd.read_csv("weight-height.csv")
left = 0.1
bottom = 0.1
top = 0.8
right = 0.8
fig = plt.figure(figsize=(10, 10), dpi= 80)
main_ax = plt.axes([left,bottom,right-left,top-bottom])
top_ax = plt.axes([left,top,right - left,1-top])
plt.axis('off')
right_ax = plt.axes([right,bottom,1-right,top-bottom])
plt.axis('off')
main_ax.plot(df['Height'],  df['Weight'], 'ko', alpha=0.5)
right_ax.boxplot(df['Height'], notch=True, widths=.6)
top_ax.boxplot(df['Weight'], vert=False, notch=True, widths=.6)
plt.show()

#**********************************************************************
#Slide-48: Histogram-based Outlier Score (HBOS) 
#**********************************************************************

import pandas as pd
from pyod.models.hbos import HBOS
from pyod.utils.utility import standardizer
from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print
df=pd.read_csv("weight-height.csv")
df = df.drop('Gender',1)
labels=np.zeros(10000)
labels[2014]=1 # We already know that this sample is an anomaly
dfn = standardizer(df)
# 24 bins, 1 outlier out of 10000 examples
clf = HBOS(n_bins=24, contamination=0.0001)
clf.fit(dfn)
anomaly_labels = clf.labels_
anomaly_scores = clf.decision_scores_
visualize('HBOS', dfn, labels, dfn, labels,anomaly_labels,
          anomaly_labels, show_figure=True, save_figure=False)
evaluate_print('HBOS', labels, anomaly_scores)


#**********************************************************************
#Slide-64: K-Nearest Neighbors (kNN)  
#**********************************************************************

import pandas as pd
from pyod.models.knn import KNN
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
df=pd.read_csv("height-weight_knn_example.csv")
dfn = standardizer(df.drop('ID',1))
clf = KNN(n_neighbors=2,contamination=1/25,method='mean')
clf.fit(dfn)
df['AnomalyScore'] = clf.decision_scores_
df['AnomalyLabel'] = clf.labels_


#**********************************************************************
#Slide-80: Local Outlier Factor (LOF)  
#**********************************************************************

import pandas as pd
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
df=pd.read_csv("height-weight_lof_example.csv")
dfn = standardizer(df.drop('ID',1))
clf = LOF(n_neighbors=3,contamination=1/25)
clf.fit(dfn)
df['AnomalyScore'] = clf.decision_scores_
df['AnomalyLabel'] = clf.labels_

#**********************************************************************
#Slide-112: Connectivity Outlier Factor (COF)   
#**********************************************************************

import pandas as pd
from pyod.models.cof import COF
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
df=pd.read_csv("height-weight_cof_example.csv")
dfn = standardizer(df.drop('ID',1))
clf = COF(n_neighbors=3,contamination=2/25)
clf.fit(dfn)
anomaly_labels = clf.labels_
anomaly_scores = clf.decision_scores_
df['AnomalyScore'] = clf.decision_scores_
df['AnomalyLabel'] = clf.labels_

#**********************************************************************
#Slide-117: One-Class SVM   
#**********************************************************************

import pandas as pd
from pyod.models.ocsvm import OCSVM
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
df=pd.read_csv("height-weight_ocsvm_example.csv")
dfn = standardizer(df.drop('ID',1))
clf = OCSVM(kernel ='linear',contamination=3/25)
clf.fit(dfn)
df['AnomalyScore'] = clf.decision_scores_
df['AnomalyLabel'] = clf.labels_


#**********************************************************************
#Slide-124: DBSCAN   
#**********************************************************************

from sklearn.cluster import DBSCAN
from pyod.utils.utility import standardizer
import pandas as pd
from pyod.utils.utility import standardizer
df=pd.read_csv("height-weight_clustering_example.csv")
dfn = standardizer(df.drop('ID',1))
clustering = DBSCAN(eps=0.5, min_samples=4).fit(dfn)
anomaly_labels=clustering.labels_
anomaly_scores=clustering.labels_
df['AnomalyScore'] = anomaly_labels

#**********************************************************************
#Slide-148: PCA  
#**********************************************************************

import pandas as pd
from pyod.models.pca import PCA
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
# Generate high-dimensional artifical Data
# Normal data is generated by a multivariate Gaussian distribution and outliers are generated by a uniform distribution
X_train, y_train = generate_data(
    n_train=200, n_test=100, n_features=2000, train_only=True, contamination=0.05)
clf = PCA(contamination=0.05,n_selected_components=2)
clf.fit(X_train)
y_train_scores=clf.labels_
df = pd.DataFrame({'TrueLabel': y_train, 'AnoamlyLabel': y_train_scores})
evaluate_print('PCA', y_train, y_train_scores)

#**********************************************************************
# Slide-166: Matrix Factorization from Scratch   
# from William Falcon
#**********************************************************************

import numpy as np
import pandas as pd
df=pd.read_csv("friends.csv")
df=df.drop('Name',1)
R =np.array(df)

steps=100 # iterations
K=5 # no. of latent variables
beta=0.02 # Regularization penalty 
alpha=0.0002 # Learning rate
error_limit=0.001 #Error tolerance

N = len(R)
M = len(R[0])
P = np.random.rand(N, K)
Q = np.random.rand(M, K)
Q = Q.T

error = 0
for step in range(steps):
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:
                eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                for k in range(K):
                    P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                    Q[k][j] = Q[k][j] + alpha * ( 2 * eij * P[i][k] - beta * Q[k][j] )
                    e = 0
                    for ii in range(len(R)):
                        for jj in range(len(R[ii])):
                            if R[ii][jj] > 0:
                                e = e + pow(R[ii][jj]-np.dot(P[ii,:],Q[:,jj]), 2)
                                for kk in range(K):
                                    e = e + (beta/2) * ( pow(P[ii][kk], 2) + pow(Q[kk][jj], 2) )
                                    if e < error_limit:
                                        break
                                        Q = Q.T



#**********************************************************************
# Slide-168: Matrix Factorization + kNN  
#**********************************************************************

import numpy as np
import pandas as pd
df=pd.read_csv("friends.csv")
df=df.drop('Name',1)
R =np.array(df)

steps=100 # iterations
K=5 # no. of latent variables
beta=0.02 # Regularization penalty 
alpha=0.0002 # Learning rate
error_limit=0.001 #Error tolerance

N = len(R)
M = len(R[0])
P = np.random.rand(N, K)
Q = np.random.rand(M, K)
Q = Q.T

error = 0
for step in range(steps):
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:
                eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                for k in range(K):
                    P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                    Q[k][j] = Q[k][j] + alpha * ( 2 * eij * P[i][k] - beta * Q[k][j] )
                    e = 0
                    for ii in range(len(R)):
                        for jj in range(len(R[ii])):
                            if R[ii][jj] > 0:
                                e = e + pow(R[ii][jj]-np.dot(P[ii,:],Q[:,jj]), 2)
                                for kk in range(K):
                                    e = e + (beta/2) * ( pow(P[ii][kk], 2) + pow(Q[kk][jj], 2) )
                                    if e < error_limit:
                                        break
                                        Q = Q.T


from pyod.models.knn import KNN
clf = KNN(n_neighbors=2,contamination=1/7,method='mean')
clf.fit(P)
df['AnomalyScore'] = clf.decision_scores_
df['AnomalyLabel'] = clf.labels_

#**********************************************************************
# Slide-180: Tensor Factorization + KNN
#**********************************************************************

def decompose_three_way(tensor, rank, max_iter=501, verbose=False):
	# CP-ALS by Mohammad Bashiri
    from tensortools.operations import unfold as tt_unfold, khatri_rao
    import tensorly as tl
    b = np.random.random((rank, tensor.shape[1]))
    c = np.random.random((rank, tensor.shape[2]))
    for epoch in range(max_iter):
        input_a = khatri_rao([b.T, c.T]) # optimize A
        target_a = tl.unfold(tensor, mode=0).T
        a = np.linalg.solve(input_a.T.dot(input_a), input_a.T.dot(target_a))
        input_b = khatri_rao([a.T, c.T]) # optimize B
        target_b = tl.unfold(tensor, mode=1).T
        b = np.linalg.solve(input_b.T.dot(input_b), input_b.T.dot(target_b))
        input_c = khatri_rao([a.T, b.T]) # optimize C
        target_c = tl.unfold(tensor, mode=2).T
        c = np.linalg.solve(input_c.T.dot(input_c), input_c.T.dot(target_c))
        if verbose and epoch % int(max_iter * .2) == 0:
            res_a = np.square(input_a.dot(a) - target_a)
            res_b = np.square(input_b.dot(b) - target_b)
            res_c = np.square(input_c.dot(c) - target_c)
            print("Epoch:", epoch, "| Loss (C):", res_a.mean(), "| Loss (B):", res_b.mean(), "| Loss (C):", res_c.mean())
    return a.T, b.T, c.T


import pandas as pd
import numpy as np
from pyod.models.knn import KNN
df1=pd.read_csv("friends2001.csv",header=None)
df2=pd.read_csv("friends2021.csv",header=None)
X1=df1.to_numpy().transpose()
X2=df2.to_numpy().transpose()
X=np.concatenate((X1,X2)).transpose()
XT=X.reshape(X1.shape[1],X1.shape[0], 2)
# Dimension reduction via Tensor Factorization
A,B,C=decompose_three_way(XT, 2, max_iter=501, verbose=True)
# KNN for Anomaly Detection
clf = KNN(n_neighbors=2,contamination=1/7,method='mean')
clf.fit(A)
AnomalyScores= clf.decision_scores_
AnomalyLabels = clf.labels_

#**********************************************************************
# Slide-192: AutoEcndoer 
#**********************************************************************

import pandas as pd
from pyod.models.auto_encoder_torch import AutoEncoder
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
# Generate high-dimensional artifical Data
# Normal data is generated by a multivariate Gaussian distribution and outliers are generated by a uniform distribution
X_train, y_train = generate_data(
    n_train=200, n_test=100, n_features=2000, train_only=True, contamination=0.05)
clf = AutoEncoder(contamination=0.05)
clf.fit(X_train)
y_train_scores=clf.labels_
df = pd.DataFrame({'TrueLabel': y_train, 'AnoamlyLabel': y_train_scores})
evaluate_print('AutoEncoder', y_train, y_train_scores)


#**********************************************************************
# Slide-206: ABOD  
#**********************************************************************

import pandas as pd
from pyod.models.abod import ABOD 
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
X=pd.read_csv("arrhythmia.csv")
true_label=pd.read_csv("arrhythmia_true_labels.csv").to_numpy()
outliers_fraction = np.count_nonzero(true_label) / len(true_label)
X= standardizer(X)
clf = ABOD(contamination=outliers_fraction)
clf.fit(X)
anomaly_label=clf.labels_
evaluate_print('ABOD', true_label, anomaly_label)


#**********************************************************************
# Slide-213: Isolation Forest   
#**********************************************************************

import pandas as pd
from pyod.models.iforest import IForest
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
X=pd.read_csv("arrhythmia.csv")
true_label=pd.read_csv("arrhythmia_true_labels.csv").to_numpy()
outliers_fraction = np.count_nonzero(true_label) / len(true_label)
X= standardizer(X)
# 100 trees, 256 subsamples
clf = IForest(contamination=outliers_fraction,n_estimators=100,max_samples =256)
clf.fit(X)
anomaly_label=clf.labels_
evaluate_print('IsolationForest', true_label, anomaly_label)


#**********************************************************************
# Slide-222: Feature Bagging 
#**********************************************************************

import numpy as np
import pandas as pd
from pyod.models.feature_bagging import FeatureBagging
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
X=pd.read_csv("arrhythmia.csv")
true_label=pd.read_csv("arrhythmia_true_labels.csv").to_numpy()
outliers_fraction = np.count_nonzero(true_label) / len(true_label)
X= standardizer(X)
clf = FeatureBagging(contamination=outliers_fraction, contamination=outliers_fraction,n_estimators=100)
clf.fit(X)
anomaly_label=clf.labels_
evaluate_print(‘Feature Bagging', true_label, anomaly_label)

