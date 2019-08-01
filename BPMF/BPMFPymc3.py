# coding:utf-8  
'''
@author: Jason.F
@data: 2019.07.31
@function: baseline BPMF(Bayesian Probabilistic Matrix Factorization)
           Datatset: MovieLens-1m:https://grouplens.org/datasets/movielens/  
           Evaluation: RMSE
'''
import sys
import time
import logging
import random
import heapq
import math
import copy
from collections import defaultdict
import pymc3 as pm
import numpy as np
from numpy import linalg as LA
from numpy.random import RandomState
import pandas as pd
import theano
import theano.tensor as tt
import tensorflow as tf

class DataSet:
    def __init__(self):
        self.trainset, self.testset, self.maxu, self.maxi, self.maxr = self._getDataset_as_list()
        
    def _getDataset_as_list(self):
        #trainset
        filePath = "/data/fjsdata/BMF/ml-1m.train.rating" 
        data = pd.read_csv(filePath, sep='\t', header=None, names=['user', 'item', 'rating'], \
                                 usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.float})
        maxu, maxi, maxr = data['user'].max()+1, data['item'].max()+1, data['rating'].max()
        print('Dataset Statistics: Interaction = %d, User = %d, Item = %d, Sparsity = %.4f' % \
                  (data.shape[0], maxu, maxi, data.shape[0]/(maxu*maxi)))
        trainset = data.values.tolist()
        #testset
        filePath = "/data/fjsdata/BMF/ml-1m.test.rating" 
        data = pd.read_csv(filePath, sep='\t', header=None, names=['user', 'item', 'rating'], \
                                 usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.float})
        testset = data.values.tolist()
        return trainset, testset, maxu, maxi, maxr 
    
    def list_to_matrix(self, dataset, maxu, maxi):              
        dataMat = np.zeros([maxu, maxi], dtype=np.float32)
        for u,i,r in dataset:
            dataMat[int(u)][int(i)] = float(r)
        return np.array(dataMat)
'''  
def build_bpmf_model(train, dim, alpha=2, std=0.01):
    """Build the modified BPMF model using pymc3. The original model uses
    Wishart priors on the covariance matrices. Unfortunately, the Wishart
    distribution in pymc3 is currently not suitable for sampling. This
    version decomposes the covariance matrix into:
        diag(sigma) \dot corr_matrix \dot diag(std).
    We use uniform priors on the standard deviations (sigma) and LKJCorr
    priors on the correlation matrices (corr_matrix):
        sigma ~ Uniform
        corr_matrix ~ LKJCorr(n=1, p=dim)
    """
    n, m = train.shape
    beta_0 = 1  # scaling factor for lambdas; unclear on its use
 
    # Mean value imputation on training data.
    train = train.copy()
    nan_mask = np.isnan(train)
    train[nan_mask] = train[~nan_mask].mean()
 
    # We will use separate priors for sigma and correlation matrix.
    # In order to convert the upper triangular correlation values to a
    # complete correlation matrix, we need to construct an index matrix:
    n_elem = int(dim * (dim - 1) / 2)
    tri_index = np.zeros([dim, dim], dtype=int)
    tri_index[np.triu_indices(dim, k=1)] = np.arange(n_elem)
    tri_index[np.triu_indices(dim, k=1)[::-1]] = np.arange(n_elem)
 
    logging.info('building the BPMF model')
    with pm.Model() as bpmf:
        # Specify user feature matrix
        sigma_u = pm.Uniform('sigma_u', shape=dim)
        corr_triangle_u = pm.LKJCorr('corr_u', n=1, p=dim, testval=np.random.randn(n_elem) * std)
 
        corr_matrix_u = corr_triangle_u[tri_index]
        corr_matrix_u = tt.fill_diagonal(corr_matrix_u, 1)
        cov_matrix_u = tt.diag(sigma_u).dot(corr_matrix_u.dot(tt.diag(sigma_u)))
        lambda_u = tt.nlinalg.matrix_inverse(cov_matrix_u)
 
        mu_u = pm.Normal('mu_u', mu=0, tau=beta_0 * tt.diag(lambda_u), shape=dim,testval=np.random.randn(dim) * std)
        U = pm.MvNormal('U', mu=mu_u, tau=lambda_u, shape=(n, dim),testval=np.random.randn(n, dim) * std)
 
        # Specify item feature matrix
        sigma_v = pm.Uniform('sigma_v', shape=dim)
        corr_triangle_v = pm.LKJCorr('corr_v', n=1, p=dim,testval=np.random.randn(n_elem) * std)
 
        corr_matrix_v = corr_triangle_v[tri_index]
        corr_matrix_v = tt.fill_diagonal(corr_matrix_v, 1)
        cov_matrix_v = tt.diag(sigma_v).dot(corr_matrix_v.dot(tt.diag(sigma_v)))
        lambda_v = tt.nlinalg.matrix_inverse(cov_matrix_v)
 
        mu_v = pm.Normal('mu_v', mu=0, tau=beta_0 * tt.diag(lambda_v), shape=dim,testval=np.random.randn(dim) * std)
        V = pm.MvNormal( 'V', mu=mu_v, tau=lambda_v, shape=(m, dim),testval=np.random.randn(m, dim) * std)
 
        # Specify rating likelihood function
        R = pm.Normal('R', mu=tt.dot(U, V.T), tau=alpha * np.ones((n, m)),observed=train)
 
    logging.info('done building the BPMF model')
    return bpmf
'''
def build_bpmf_model(train, dim, alpha=2, std=0.01):
    # Mean value imputation on training data.
    train = train.copy()
    nan_mask = np.isnan(train)
    train[nan_mask] = train[~nan_mask].mean()
 
    # Low precision reflects uncertainty; prevents overfitting.
    # We use point estimates from the data to intialize.
    # Set to mean variance across users and items.
    alpha_u = 1 / train.var(axis=1).mean()
    alpha_v = 1 / train.var(axis=0).mean()
 
    logging.info('building the BPMF model')
    n, m = train.shape
    with pm.Model() as bpmf:
        U = pm.MvNormal('U', mu=0, tau=alpha_u * np.eye(dim),shape=(n, dim), testval=np.random.randn(n, dim) * std)
        V = pm.MvNormal('V', mu=0, tau=alpha_v * np.eye(dim),shape=(m, dim), testval=np.random.randn(m, dim) * std)
        R = pm.Normal('R', mu=tt.dot(U, V.T), tau=alpha * np.ones(train.shape),observed=train)
    logging.info('done building BPMF model')
    return bpmf
   
if __name__ == "__main__":
    ds = DataSet()#loading dataset\
    R = ds.list_to_matrix(ds.trainset, ds.maxu, ds.maxi)#get matrix
    for K in [8, 16, 32, 64]:
        bpmf = build_bpmf_model(train=R, dim=K)#dim is the number of latent factors
        with bpmf:# sample with BPMF
            tstart = time.time()
            logging.info('Starting BPMF training')
            approx = pm.fit(n=1000, method=pm.ADVI())
            trace = approx.sample(draws=500)
            #start = pm.find_MAP()    
            #step = pm.NUTS()
            #trace = pm.sample(1000, step=step, start=start)
            elapsed = time.time() - tstart    
            logging.info('Completed BPMF in %d seconds' % int(elapsed))
        
        with bpmf:#evaluation
            ppc = pm.sample_posterior_predictive(trace, progressbar=True)
            nR = np.mean(ppc['R'],0)#three dims, calcuate the mean with the first dim for posterior

        squaredError = []
        for u,i,r in ds.testset:
            error=r - nR[int(u)][int(i)]
            squaredError.append(error * error)
        rmse =math.sqrt(sum(squaredError) / len(squaredError))
        print("RMSE@{}:{}".format(K, rmse))