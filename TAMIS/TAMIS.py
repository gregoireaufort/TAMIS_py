#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aufort
"""

import numpy as np
import time
from .utils import compute_ESS,compute_KL, compute_perplexity, adapt_beta,log_weights_to_norm_weights
import scipy.stats as stats
from .GMM import GMM_fit
import copy
from scipy.special import logsumexp


class TAMIS(object):
    def __init__(self,
                 target,
                 proposal=None,
                 n_comp = 3,
                 init_theta =[[0,0,0,0],200*np.eye(4)],
                 n_sample=10000,
                 ESS_tol = 1000,
                 alpha = 500,
                 tau = 0.6,
                 EM_solver = "sklearn",
                 integer_weights = False,
                 adapt = True,
                 betas = None,
                 recycle= True,
                 recycling_iters = None,
                 verbose= 0):
        """
       Initializes the TAMIS algorithm with necessary parameters.

       :param target: The target distribution for importance sampling.
       :param proposal: The proposal distribution, defaults to a multivariate normal if None.
       :param n_comp: Number of components in the mixture proposal distribution.
       :param init_theta: Initial parameters for the proposal distribution.
       :param n_sample: The number of samples to draw at each iteration.
       :param ESS_tol: Effective Sample Size threshold for stopping criterion.
       :param alpha: Minimal effective sample size required by the tempering scheme.
       :param tau: Truncation threshold used in weight modification.
       :param EM_solver: Solver to use for Expectation-Maximization.
       :param integer_weights: Flag to determine how weights are interpreted if sklearn is the solver.
       :param adapt: Flag to enable automatic adjustment of Beta.
       :param betas: List of inverse temperatures, used if adapt is False.
       :param recycle: Flag to enable recycling of previous iterations' samples.
       :param recycling_iters: Specifies how many iterations to recycle.
       :param verbose: Verbosity level for logging.
       """

        self.iteration = 0
        self.target = target
        self.theta =init_theta
        self.dim = target.dim
        self.total_sample = []
        self.total_target = []
        self.n_sample = n_sample
        self.theta_total = [init_theta]
        self.total_proposal= []
        self.ESS = []
        self.ESS_tol = ESS_tol
        self.tmprd_ESS = []
        self.betas = []
        self.previous_temp = 0
        self.max_iter = 0
        self.n_comp = n_comp
        self.alpha = alpha
        self.verbose = verbose
        self.adapt = adapt
        self.total_weight = []
        self.KL = []
        self.EM_solver = EM_solver
        self.tau = tau
        self.integer_weights = integer_weights
        self.recycle = recycle
        self.recycling_iters = recycling_iters
        self.path_betas = betas
        if proposal is None:
            self.proposal = stats.multivariate_normal
            self.p=0
        else:
            self.proposal = proposal
            self.p = 1
        
        
    def proposal_rvs(self):
        """
        Samples from the proposal distribution, a multivariate gaussian if
        None is selected
        """
        n= self.n_sample[self.iteration]
        theta = self.theta
        if self.p == 0:
            self.sample = self.proposal.rvs(size = n, 
                                            mean =theta.mean,
                                            cov = theta.variance)
        else :
            self.sample = self.proposal.rvs(size = n, 
                                            means =theta.mean,
                                            covs = theta.variance,
                                            weights =theta.proportions)

    def proposal_logpdf(self, x):
        """
        Computes the logpdf of x from the proposal
        """
        theta = self.theta
        if self.p == 0:
            pdf = self.proposal.logpdf(x,
                                       mean = theta.mean, 
                                       cov = theta.variance)
        else :
            pdf = self.proposal.logpdf(x, 
                                       means = theta.mean, 
                                       covs = theta.variance,
                                       weights = theta.proportions)
        return pdf 
    
    
    
    
    def compute_weights(self, targ_sample):
        """
        Computes the importance weights for the given target samples.
        """
        sample = self.sample
        proposal_sample = self.proposal_logpdf(sample)
        log_wgts = targ_sample - proposal_sample
        return log_wgts
    
    def compute_targ_sample(self):
        """
       evaluate target on sample based on the log likelihood and log prior of the target distribution.
       """
        sample = self.sample
        log_lkhd_sample = np.array(self.target.log_likelihood(sample = sample))
        log_prior_sample = np.array(self.target.log_prior(sample))

        targ_sample = log_prior_sample + log_lkhd_sample
        return targ_sample
    
   
    
    def set_beta(self,log_wgts):
        """
        Determines the tempering parameter (beta) based on the effective sample size.
        """
        ESS = compute_ESS(self.weights)
        self.ESS.append(ESS)
        if ESS <self.alpha and self.adapt and (self.path_betas is None):
            beta = adapt_beta(log_wgts,
                                alpha = self.alpha)
        elif (self.path_betas is not None):
            beta = self.path_betas[self.iteration]
        else :
            beta = 1     
        return beta
    
    def weight(self):
        """
        Computes the importance sampling weights and updates the proposal parameters.
        """
        targ_sample =self.compute_targ_sample()
        log_wgts = self.compute_weights(targ_sample)
        self.weights = log_weights_to_norm_weights(log_wgts)
        
        self.total_weight.append(self.weights)

        beta = self.set_beta(log_wgts)
        
        log_tmprd_wgts = beta * log_wgts
        self.tmprd_wgts = log_weights_to_norm_weights(log_tmprd_wgts)
        
        self.betas.append(beta)
        self.total_target.append(np.vstack(targ_sample))
        self.previous_temp = beta
        
    def update_metrics(self):
        """
        Tracks effective sample size and Kullback-Leibler divergence.
        """
        tmprd_ESS = compute_ESS(self.tmprd_wgts)
        self.tmprd_ESS.append(tmprd_ESS)
        
        KL= compute_KL(self.weights)
        self.KL.append(KL)
        
        
        
    def test_stop(self):
        """
       Tests whether the stopping criterion based on the effective sample size is met.
       """
    
        if self.verbose == 1:
            print("Iteration" , self.iteration)
            print("ESS = ",self.ESS[self.iteration])
            print("Kullback-Leibler divergence = ", self.KL[self.iteration] )
            print("perplexity = ",compute_perplexity(self.weights))
            
        return np.sum(self.ESS)>self.ESS_tol
    
    def store_sample(self):
        """
        stores the current sample with the ones from previous iterations
        """
        self.total_sample.append(self.sample)
        
    
    def update_theta(self):
        """
        Updates proposal parameters using sample and tempered weights
        """
        theta = copy.copy(self.theta)
        weights = self.tmprd_wgts 
        if self.p == 0:
            theta.mean = np.average(self.sample, 
                                    weights = weights,
                                    axis = 0)
            theta.variance = np.cov(self.sample, 
                                    aweights = weights,
                                    rowvar = False ) 
        else :
            temp = GMM_fit(sample = self.sample,
                           weights= weights,
                           n_comp = self.n_comp,
                           tau =self.tau,
                           init_parameters = theta,
                           EM_solver = self.EM_solver,
                           integer_weights=self.integer_weights)
            theta.mean=temp.mean
            theta.variance=temp.variance
            theta.proportions= temp.proportions
        self.theta = theta
        self.theta_total.append(theta)
        
    def iterate(self):
        """
        Makes one iteration of the AMIS scheme without recycling
        """
        self.proposal_rvs()
        self.weight()
        self.update_metrics()
        self.update_theta()
        self.store_sample()

    
    def _recycling_step(self, iters_to_keep):
        """
        Performs the recycling step in the TAMIS algorithm. This step reuses samples from specified iterations 
        to enhance the estimation process.
    
        :param iters_to_keep: Indices of the iterations whose samples and targets are to be recycled.
        """
    
        # Combining samples and targets from the specified iterations.
        self.total_sample = np.row_stack([self.total_sample[i] 
                                          for i in iters_to_keep])
        self.total_target = np.row_stack([self.total_target[i] 
                                          for i in iters_to_keep])
    
        # Summing up the number of samples across the specified iterations.
        N_total = np.sum([self.n_sample[i] for i in iters_to_keep])
    
        # Timing the execution of the recycling process for performance analysis.
        t1 = time.time()
    
        # Loop through each iteration to be recycled.
        for i in iters_to_keep:
            # Set the proposal distribution parameters for the current iteration.
            self.theta = self.theta_total[i]
    
            # Compute and store the logpdf of the total sample under the current proposal distribution.
            prop = self.proposal_logpdf(self.total_sample)
            self.total_proposal.append(prop)
    
        # Mark the end time of the iterations processing.
        t2 = time.time()
        print(t2 - t1, "for the iterations")
    
        # Reshape the target log probabilities for further processing.
        log_target = self.total_target.reshape((len(self.total_target),))
    
        # Rescale the target log probabilities to prevent numerical underflow.
        num = log_target - np.max(log_target)
    
        # Gather the log probabilities under the proposal distribution.
        prop = self.total_proposal
    
        # Prepare the number of samples in each iteration for weighting.
        smpl = np.array(self.n_sample)[iters_to_keep, None]
    
        # Compute the log denominator for normalization, using log-sum-exp for numerical stability.
        denom = -np.log(N_total) + logsumexp(a=prop, b=smpl, axis=0)
    
        # Calculate the unnormalized log weights.
        unnom_weights_log = num - denom
    
        # Compute the log marginal likelihood 
        self.log_marg_lkhd = -np.log(N_total) + logsumexp(unnom_weights_log)
    
        # Normalize the final weights to prevent numerical overflow.
        max_norm = np.max(unnom_weights_log)
        self.final_weights = np.exp(unnom_weights_log - max_norm)

    def final_step(self):
        
        T = len(self.total_sample)
        t1 = time.time()

        if not self.recycling_iters:
            self._recycling_step(range(T))
        else:
            if self.recycling_iters == "auto":
                midway = (np.max(self.betas) - np.min(self.betas)) / 2
                iters_to_keep = list(np.where(self.betas > midway)[0])
            elif isinstance(self.recycling_iters, int):
                n = self.recycling_iters
                iters_to_keep = np.argpartition(self.betas, -n)[-n:]
            else:
                raise ValueError("recycling_iters must be 'auto' or int")
            self._recycling_step(iters_to_keep)
    
        t2 = time.time()
        print(t2 - t1, "for the recycling")
        

    def result(self, T = 10):
        """
        complete TAMIS scheme
        """
        t1 = time.time()
        for i in range(T):
            self.iteration = i
            self.iterate()
            a= self.test_stop()
            if a:
                break
        self.max_iter = i
        print("Stopped after step ",i+1)
        print("final_step")
        t2=time.time()
        print(t2-t1, "for the  ", i+1,"iterations")
        if self.recycle == True:
            self.final_step()
            print(time.time()-t2, "for recycling")
            self.ESS_final = compute_ESS(self.final_weights)
            self.KL_final = compute_KL(self.final_weights)
            print(time.time()-t1, "seconds total")
        return self
    
   
    
        
    
 
       