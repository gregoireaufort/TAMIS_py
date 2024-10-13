# -*- coding: utf-8 -*-
"""
@author: aufort
"""

import itertools
from collections import Counter
import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning



def E_step(sample,n_comp, means,variances, weights_comp):
    """
    Performs the Expectation step of the EM algorithm.

    :param sample: Array of observed data.
    :param n_comp: Number of components (clusters) in the mixture model.
    :param means: Array of mean vectors for each component.
    :param variances: Array of variance matrices for each component.
    :param weights_comp: Array of weights for each component in the mixture.
    :return: Log probabilities of each observation belonging to each component.
    """

    # Calculate the log probability for each component and each sample.
   
    num =  [weights_comp[i]+ 
             stats.multivariate_normal.logpdf(sample,
                                              means[i],
                                              variances[i],
                                              allow_singular = True) 
             for i in range(n_comp)]
    # Compute the log-sum-exp for numerical stability.

    denom = np.logaddexp.reduce(num,axis = 0)
    
    # Subtract the denominator from the numerator to normalize.

    return num-denom

def M_step(sample, probs,weights, n_comp):
    """
    Performs the Maximization step of the EM algorithm.

    :param sample: Array of observed data.
    :param probs: Probabilities from the E step, indicating the likelihood of each sample belonging to each component.
    :param weights: Sample weights.
    :param n_comp: Number of components (clusters) in the mixture model.
    :return: Updated means, covariance matrices, and weights of each component.
    """

    # Small constant to prevent division by zero.
    eps = 1e-6
    
    # Initialize arrays for means, covariance matrices, and coefficients.
    means = np.zeros((sample.shape[1],n_comp))
    covar = []
    coeffs = np.zeros((sample.shape[0],n_comp))
    new_weights_comp = np.zeros(n_comp)
    
    # Resample data according to provided weights.
    sample = resampling_data(sample, weights, sampling_type="random")
    for i in range(n_comp):
        # Compute coefficients for the current component.
        coeffs[:,i] = np.exp(probs[i])
        
        # Update the mean for the current component.
        means[:,i] = np.average(sample,
                                weights = coeffs[:,i],
                                axis =0)
        
        # Compute and update the covariance matrix for the current component.
        variances = [np.cov(sample[:,j],
                             aweights = coeffs[:,i]) + eps 
                     for j in range(sample.shape[1])]
        covar.append(np.diag(variances))
        
        # Update the component weight.
        new_weights_comp[i] = np.mean(np.exp(probs[i]))
    return means.T,covar, new_weights_comp

    
def diagonal_EM_homemade(sample,weights, n_comp, init_parameters):
    """
    Performs the Expectation-Maximization algorithm with diagonal covariance matrices.

    :param sample: Array of observed data.
    :param weights: Sample weights.
    :param n_comp: Number of components (clusters) in the mixture model.
    :param init_parameters: Object containing initial parameters for means, variances, and component weights.
    :return: Final parameters after running the EM algorithm.
    """

    # Initialize parameters.
    means = init_parameters.mean
    variances = init_parameters.variance
    weights_comp = init_parameters.proportions
    
    # Perform EM algorithm for a fixed number of steps (3).
    for i in range(3):
        probs = E_step(sample,n_comp, means,variances, weights_comp)
        means,variances,weights_comp = M_step(sample, probs,weights,n_comp) 
    
    # Wrap the final parameters in a 'parameters' object.
    params = parameters(means,variances,weights_comp)
    return params




class parameters(object):
    """
    class to have identical attributes between pomegranate and Mixmod outputs
    """
    def __init__(self,means,variances, proportions):
        self.mean  = means
        self.variance = variances
        self.proportions = proportions
            
def resampling_data(data,weights,sampling_type ="random"):
    """
    Resamples the given data based on provided weights.

    :param data: Array of data points to be resampled.
    :param weights: Array of weights corresponding to the data points.
    :param sampling_type: The type of resampling to be done. Default is "random".
                          If "random", performs weighted random sampling.
                          Otherwise, it performs deterministic resampling based on weights.
    :return: Resampled data array.
    """

    # Get the number of data points.
    N = len(data)
    if sampling_type == "random":
        
        # Normalize the weights to sum to 1.
        norm_weights =  weights/np.sum(weights)
        
        # Perform weighted random sampling.
        sample_index = np.random.choice(range(N), 
                                        size=N,
                                        replace=True,
                                        p = norm_weights)
        
        # Gather the sampled data based on the selected indices.
        sample = data[sample_index]
    else : 
        # Perform deterministic resampling.
        sample =np.repeat(data,weights, axis = 0)
    return sample

@ignore_warnings(category = ConvergenceWarning)
def sklearn_diagonal_mixture_with_init(sample,
                                        weights_int,
                                        n_comp, 
                                        init_parameters,
                                        sampling_type = "random"):
    data = resampling_data(sample, weights_int, sampling_type=sampling_type)
    init_means =  [init_parameters.mean[i] for i in range(n_comp)]

    clf = GaussianMixture(n_components = n_comp, 
                          covariance_type ="diag", 
                          means_init = init_means, 
                          max_iter = 10)
    clf.fit(data)
    params = parameters(means = clf.means_,
                        variances = clf.covariances_,
                        proportions = clf.weights_)
    
    return params

def truncate_weights(weights,isint,trunc):
    """
   Modifies the weights array based on a truncation threshold and an option to convert weights to integers.

   :param weights: Array of weights to be modified.
   :param isint: Boolean flag indicating whether to convert the modified weights to integers.
   :param trunc: Truncation threshold as a quantile value (between 0 and 1).
                 Weights below this quantile value will be increased to the value at this quantile.
   :return: The modified weights array. It's either a float array or an integer array based on the 'isint' flag.
   """

    # Calculate the truncation value as the quantile of the weights.
    trunc =  np.quantile(weights,trunc)
    
    # Copy the original weights array to avoid modifying it in place.
    weights2  = weights.copy()
    
    # Increase the weights that are below the truncation threshold to the truncation value.
    weights2[weights<trunc] = trunc
    if isint == True:
        # If 'isint' is True, convert weights to integers.
        # This scales the weights and then adds 1 to ensure a minimum weight of 1.
        weights_int = np.int64((1/trunc)*weights2) + 1
        return weights_int
    return weights2





def GMM_fit(sample, 
            weights,
            tau = 0.6,
            n_comp = 3, 
            init_parameters = None,
            EM_solver = "homemade",
            integer_weights = False):
    """
   Fits a Gaussian Mixture Model (GMM) to the given sample data using specified parameters and solver.

   :param sample: Array of sample data to fit the GMM to.
   :param weights: Array of weights for the sample data.
   :param tau: Truncation threshold for truncateing weights (used in truncate_weights function).
   :param n_comp: Number of components (clusters) in the GMM.
   :param init_parameters: Initial parameters for the GMM, if provided.
   :param EM_solver: Specifies which solver to use for the EM algorithm. Can be 'sklearn' or 'homemade'.
   :param integer_weights: Boolean flag indicating whether to convert weights to integers.
   :return: Parameters of the fitted GMM (mean, covariance, and weights of components).
   """

   # truncate the weights based on the truncation threshold and integer_weights flag.
   
    weights = truncate_weights(weights,integer_weights,tau)
    
    if EM_solver == "sklearn":
        if integer_weights == True : 
            sampling_type = "deterministic"
        else : 
            sampling_type = "random"
        params = sklearn_diagonal_mixture_with_init(sample,
                                        weights,
                                        n_comp, 
                                        init_parameters,
                                        sampling_type)
    elif EM_solver =="homemade":
        params = diagonal_EM_homemade(sample,weights,n_comp,init_parameters)
    else :
        assert ValueError("unknown solver")

    return params
    
          

class Mixture_gaussian(object):
    
   
    def rvs(size, means, covs, weights):
        """
       Simulates random variates from the Gaussian Mixture Model.

       :param size: Number of random variates to generate.
       :param means: List of mean vectors for each Gaussian component.
       :param covs: List of covariance matrices for each Gaussian component.
       :param weights: List of mixture weights for each Gaussian component.
       :return: A numpy array of random variates sampled from the mixture model.
       """
        n_comp = len(means)
        S=[]
        index= np.random.choice(range(n_comp), p = weights, size = size)
        
        # Count the number of draws from each component
        n_draw = Counter(index)
        
        # For each component, draw the specified number of samples.
        for i in  n_draw.keys():
            temp = stats.multivariate_normal.rvs(mean= means[i],
                                      cov=covs[i],
                                      size=n_draw[i])
            S.append(temp.reshape(n_draw[i],means[i].shape[0]))
        
        # Flatten the list of samples and reshape to the required size.
        res = list(itertools.chain(*S))
        return np.array(res).reshape((len(index),means[0].shape[0]))
    
    def logpdf(x,means, covs,weights):
         n_comp = len(means)
         
         # Compute the logpdf for each component and each data point.
         a=[np.log(weights[i]) + 
            stats.multivariate_normal.logpdf(x,
                                             means[i],
                                             covs[i], 
                                             allow_singular = True) 
            for i in range(n_comp)]
         
         
         # Combine the component logpdfs using log-sum-exp for numerical stability.
         lpdf =  np.logaddexp.reduce(a,axis = 0)
         return lpdf

