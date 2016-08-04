# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:44:58 2016

@author: AbreuLastra_Work
"""

import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

#setting up SVM
from sklearn import svm
svc = svm.SVC(kernel='linear', C=2)

#Setting up the graphs Adapted from https://github.com/jakevdp/sklearn_scipy2013
from matplotlib.colors import ListedColormap
import numpy as np
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

          
def plot_estimator(estimator, X, y, k, l):
    estimator.fit(X, y)
    x_min, x_max = X[df.columns[[k]]].min()[0] - .1, X[df.columns[[k]]].max()[0] + .1
    y_min, y_max = X[df.columns[[l]]].min()[0] - .1, X[df.columns[[l]]].max()[0] + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[df.columns[[k]]], X[df.columns[[l]]], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()

for i in range(0, len(iris.target_names)):
    for j in range(0, len(iris.target_names)):
        if i != j:
            temp_df = df[(df['target'] ==i) | (df['target']==j)]
            for k in range(0,len(iris.feature_names)):
                    for l in range(0,len(iris.feature_names)):
                        if k != l:
                           plt.figure() 
                           plt.scatter(temp_df[df.columns[[k]]], temp_df[df.columns[[l]]], c = temp_df['target'])
                           plt.xlabel(iris.feature_names[k])
                           plt.ylabel(iris.feature_names[l])
                           X = temp_df[df.columns[[k,l]]]
                           y = temp_df['target']
                           svc.fit(X, y)
                           plot_estimator(svc, X, y, k, l)
                           
                           
            

