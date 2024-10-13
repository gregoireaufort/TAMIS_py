#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aufort
"""

import numpy as np
from math import gamma
import scipy.stats as stats
import itertools

class Banana(object):
    """
    Banana likelihood problem
    (eg https://arxiv.org/pdf/0907.1254.pdf"""
    def __init__(self,
                 b=0.03,
                 var_banana = 100,
                 dim = 10):
        self.b = b
        self.var_banana = var_banana
        self.dim = dim
        
        
    def log_prior(self,sample):
        prior = np.zeros(shape = ((sample.shape[0]),))#improper 1(R)
        return prior
       
    def log_likelihood(self,sample): 
        b=self.b
        var = [self.var_banana] + [1] * (self.dim-1)
        temp = np.array([sample[:,0],sample[:,1]+b*(sample[:,0]**2-var[0])]).T
        
        res = stats.multivariate_normal.logpdf(np.hstack((temp,sample[:,2:])), mean = np.zeros((self.dim,)), cov = np.diag(var)) 

        return res 




class Eggshell(object):
    """
    https://arxiv.org/pdf/0809.3437.pdf
    """
    def __init__(self,
                 dim = 2):
        self.dim = dim
        
        
    
    def log_prior(self,sample):
        prior = np.sum(stats.uniform.logpdf(sample, loc = 0,scale = 10*np.pi),axis =1)
        return prior
       
    def log_likelihood(self,sample): 
        log_lik = (2+ np.cos(sample[:,0]/2)*np.cos(sample[:,1]/2))**5
        return log_lik

class gaussian(object):
    """
    Simple diagonal multivariate gaussian
    """
    def __init__(self,
                 dim = 2,
                 mean=40,
                 var = 1):
        self.dim = dim
        self.mean = [mean]*dim
        self.var = [var]*dim
            
    def log_prior(self,sample):
         prior =np.zeros(shape = ((sample.shape[0]),))#improper 1(R)
         return prior
       
    def log_likelihood(self,sample): 
         log_lik = stats.multivariate_normal.logpdf(sample, mean = self.mean, cov = np.diag(self.var))
         return log_lik
     
        
class gaussian_var(object):
    """
    Simple diagonal multivariate gaussian
    """
    def __init__(self,
                 dim = 2,
                 mean=40,
                 var = 1):
        self.dim = dim
        self.mean = [mean]*dim
        self.var = var
            
    def log_prior(self,sample):
         prior =np.zeros(shape = ((sample.shape[0]),))#improper 1(R)
         return prior
       
    def log_likelihood(self,sample): 
         log_lik = stats.multivariate_normal.logpdf(sample, mean = self.mean, cov = np.diag(self.var))
         return log_lik
     
        
     
class reg_log_speagle(object):    
    """
    https://arxiv.org/pdf/1904.02180.pdf
    """
    def __init__(self):
         np.random.seed(56101)
         m_true = -0.9594
         b_true = 4.294
         f_true = 0.534
         N = 50
         self.x= np.sort(10*np.random.rand(N))
         self.yerr = 0.1 + 0.5*np.random.rand(N)
         y_true = m_true *self.x + b_true
         self.y = y_true + np.abs(f_true * y_true) * np.random.randn(N)
         self.y += self.yerr * np.random.randn(N)
         self.dim = 3
         
    def log_prior(self,sample):
        logprior = stats.uniform.logpdf(sample[:,0], loc = -5,scale =5.5) + stats.uniform.logpdf(sample[:,1], loc = 0,scale =10) + stats.uniform.logpdf(sample[:,2], loc = -10,scale =11) 
        return logprior
       
    def log_likelihood(self,sample): 
         model = np.array([sample[i,0] * self.x + sample[i,1] for i in range(sample.shape[0])])
         inv_sigma2 =np.array([1./(self.yerr**2 + model[i] **2 * np.exp(2*sample[i,2])) for i in range(sample.shape[0])])
         log_lik = -0.5 *(np.sum((self.y-model)**2 * inv_sigma2 -np.log(inv_sigma2), axis = 1))
         return log_lik
     

class log_gamma(object):
    """
    https://arxiv.org/pdf/1304.7808.pdf
    """
    def __init__(self,
                 dim = 2):
        self.dim = dim
        
        
    
    def log_prior(self,sample):
        prior = np.sum(stats.uniform.logpdf(sample, loc = -30,scale =60),axis =1)
        return prior
       
    def log_likelihood(self,sample): 
         
         T1 = [np.logaddexp(0.5*stats.norm.logpdf(sample[:,0], loc = -10, scale= 1),0.5*stats.norm.logpdf(sample[:,0], loc = 10, scale= 1))]
         T2 = [np.logaddexp(0.5*stats.loggamma.logpdf(sample[:,1], loc = -10, c= 1, scale = 1),0.5*stats.loggamma.logpdf(sample[:,1], loc = 10, c= 1, scale = 1))]
         T3 =[stats.loggamma.logpdf(sample[:,i], loc = 10, c= 1, scale = 1) for i in range(2,int((self.dim+2)/2))]
         T4 =[stats.norm.logpdf(sample[:,i], loc = 10, scale= 1)  for i in range(int((self.dim+2)/2),self.dim)]
         temp = [T1,T2,list(itertools.chain(*[T3,T4]))]
         log_lik = np.sum(list(itertools.chain(*temp)), axis = 0)
         return log_lik
     
class student(object):
    def __init__(self,
               dim = 2,
               mean = 10,
               var = 1):
        self.dim = dim
        self.cov =  np.diag([var]*dim)
        self.mean =  [mean]*dim
        
    def log_prior(self,sample):
        prior = np.sum(stats.uniform.logpdf(sample, loc = 0,scale = 200),axis =1) #improper 1(R)
        return prior
    
    def log_likelihood(self,sample,df=1):
        '''Computes logpdf of multivariate t distribution
        Parameters
        ----------
        x : array_like
            points where the logpdf must be computed
        m : array_like
            mean of random variable, length determines dimension of random variable
        S : array_like
            square array of covariance  matrix
        df : int or float
            degrees of freedom
        Returns
        -------
        rvs : array, (x.shape[0], )
            each value of the logpdf
        '''
        [n,d] = sample.shape
        x_m = sample-self.mean
        cov = self.cov #the Sigma parameter of a student is not the covariance matrix
        inv_cov = np.linalg.pinv(cov)
        det_cov = np.linalg.det(cov)
        log_num = np.log(gamma((df+d)/2))
        L = np.asarray([np.dot(np.dot(x_m[i,:].T,inv_cov),x_m[i,:]) for i in range(n)])
        log_denom = np.log(gamma(df/2.)) + (d/2.)*(np.log(df) +np.log(np.pi)) + .5*np.log(det_cov) +((df+d)/2)*np.log(1+ (1/df)* L)
        #np.diagonal( np.dot( np.dot(x_m, inv_cov), x_m.T)))
        res = log_num-log_denom
        res[np.sum(sample<0, axis =1)!=0] =  -np.inf
        #res = np.sum([stats.chi2.logpdf(sample[:,i], df = 15) for i in range(sample.shape[1])],axis = 0)
        return res