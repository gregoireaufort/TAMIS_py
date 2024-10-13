#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aufort
"""

import numpy as np
from scipy.interpolate import griddata
from .utils import compute_KL, gauss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import pandas as pd


def plot_convergence(TAMIS,title = False,save = False):
    
    KL=[]
    for i in range(TAMIS.max_iter + 1):
        bootstrapped = [resample(TAMIS.total_weight[i]) for j in range(100)]
        KL.append( [compute_KL(bootstrapped[j]) for j in range(100)])
    arr = np.array(KL).T
    df = pd.DataFrame(data = arr).melt()
    df.columns = ["Iteration","Kullback-Leibler divergence"]
    df2 = pd.DataFrame(data = np.array(TAMIS.betas)).melt()
    df2.columns = ["Iteration","Beta"]
    df2["Iteration"] = range(TAMIS.max_iter+1)
    fig,ax = plt.subplots()
    pl =sns.lineplot(x="Iteration",
                     y="Kullback-Leibler divergence", 
                     data = df,ci="sd",
                     legend = "brief",
                     label = "KLD")
    plt.legend(loc = "center left")
    #pl.set_xticks(range(self.max_iter + 1))
    ax2 = ax.twinx()
    sns.lineplot(x="Iteration",
                 y="Beta",
                 data = df2,
                 ci="sd",
                 color = 'r',
                 legend = "brief",
                 label = "Beta",
                 linestyle = "--")
    plt.legend(bbox_to_anchor=(0,.4),loc = "center left")
    if title :
        pl.set_title(title)
    if save :
        fig = pl.get_figure()
        fig.savefig(save)
          
def plot_contour(x,y,z,alpha, color):
    npts = 1000
    # define grid.
    xi = np.linspace(-4.1, 4.1, npts)
    yi = np.linspace(-4.1, 4.1, npts)
    ## grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    levels = [0.1,0.3,0.5,0.7,0.9]
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi,
                     yi,
                     zi,
                     linewidths=0.5,
                     colors=color, 
                     levels=levels, 
                     alpha = alpha)
    #CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)
    return CS
  

def plot_iteration(TAMIS,target,cluster,prev_target = None, title ="", n = 0):
        """
        plot the contour of estimated mixture at iteration max_iter - n 

        Parameters
        ----------
        n : int, optional
            Number of backward steps from last iteration. The default is 0.

        Returns
        -------
        None
        """        
        max_iter = TAMIS.max_iter
        means = TAMIS.theta_total[max_iter - n].mean
        covariances = TAMIS.theta_total[max_iter - n].variance
        n_comp = len(means)
        # make up some randomly distributed data
        npts = 5000
        x = np.random.uniform(-4, 4, npts)
        y = np.random.uniform(-4, 4, npts)
        z=[]
        z = [gauss(x, y, Sigma=covariances[i][0:2,0:2],mu=means[i][0:2]) for i in range(n_comp)]
        plt.figure()
        for i in range(n_comp):
            plot_contour(x=x,y=y,z=z[i],alpha =1, color = 'green')
        if cluster is not None:
            plt.scatter([cluster[i][0] for i in range(len(cluster))],[cluster[i][1] for i in range(len(cluster))],s=1)
        est_mean=np.average(TAMIS.total_sample, weights=TAMIS.final_weights,axis = 0)
        est_var=np.cov(TAMIS.total_sample.T, aweights=TAMIS.final_weights[:,])
        z_est = gauss(x, y, Sigma=est_var[0:2,0:2],mu=est_mean[0:2]) 
        if target is not None:
            z_true =  gauss(x, y, Sigma=target[1][0:2,0:2],mu=target[0][0:2])
            CS3 = plot_contour(x,y,z_true,0.6, color = "red")
        CS2 = plot_contour(x,y,z_est,0.6, color = "blue")
        if prev_target is not None:
            z_prev = gauss(x, y, Sigma=prev_target[1][0:2,0:2],mu=prev_target[0][0:2]) 
            CS1 = plot_contour(x,y,z_prev,0.6, color = "black")
            lines = [CS1.collections[0],CS2.collections[0],CS3.collections[0]]
            plt.legend(handles = lines,labels = ("previous","est","true"))
            
        elif target is not None:
            lines = [CS2.collections[0],CS3.collections[0]]
            plt.legend(handles = lines,labels = ("est","true"))
        else : 
            lines = [CS2.collections[0]]
            plt.legend(handles = lines,labels = ("est"))
        plt.title( title +" \n iter = "+str(max_iter-n))
        plt.show()