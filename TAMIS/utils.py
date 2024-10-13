#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aufort
"""


import numpy as np
from scipy.special import xlogy
from scipy.optimize import bisect


class theta_params(object):
    def __init__(self, params):
        self.mean = params[0]
        self.variance = params[1]
        self.proportions = params[2]
    

def remove_inf(weights,sample):
    idx_finite = np.where(np.isfinite(weights))
    weights = weights.copy()[idx_finite]
    sample = sample.copy()[idx_finite]
    return weights,sample

def mean_res(TAMIS):
    weights,sample = remove_inf(TAMIS.final_weights,TAMIS.total_sample)
    weights/=np.sum(weights)
    return np.average(sample, weights=weights,axis = 0)

def cov_res(TAMIS):
    weights,sample = remove_inf(TAMIS.final_weights,TAMIS.total_sample)
    weights/=np.sum(weights)
    return np.diag(np.cov(sample.T, aweights=weights[:,]))

        
def compute_ESS(weights):
    """
    computes the ESS estimator
    """
    with np.errstate(divide ="ignore", invalid ="ignore"):
        weights /= np.sum(weights)
        ESS = 1/np.sum(weights**2)
    if np.isnan(ESS):
        ESS =0
    return ESS

def compute_marginal_likelihood(weights):
    return np.mean(weights)

def compute_KL(weights):
    """
    computes the KL estimator
    """
    norm_weights = weights / np.sum(weights)
    KL= np.sum(xlogy(norm_weights,norm_weights))  
    return KL + np.log(len(norm_weights))

def compute_perplexity(weights):
    norm_weights = weights / np.sum(weights)
    perp = -np.sum(xlogy(norm_weights,norm_weights))  
    return np.exp(perp)/len(weights)



            
def adapt_beta(wgts, alpha):
    """
    Adapt beta by binary search to get a tempered ESS of alpha
    Parameters
    ----------
    wgts : np.array
        importance weights
    alpha : scalar
        Minimum effective sample size required

    Returns
    -------
    res : float
        value of beta such that ESS(beta) = alpha

    """
        
    def ESS_beta(beta):
        temp_wgts = np.exp(beta * wgts)
        ESS= compute_ESS(temp_wgts)
        return ESS
    
    def ESS_to_solve(beta):
        return ESS_beta(beta) - alpha
  
    res = bisect(ESS_to_solve, 1e-5,1,xtol=1e-4)
    return res
def max_scale(sample):
    return sample - np.max(sample)

def log_weights_to_norm_weights(log_weights):
    scaled = max_scale(log_weights)
    unnorm_weights = np.exp(scaled)
    norm_weights = unnorm_weights /np.sum(unnorm_weights )
    return norm_weights

def gauss(x,y,Sigma,mu):
    mu = np.array(mu)
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))
