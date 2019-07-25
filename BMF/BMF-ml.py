# -*- Encoding:UTF-8 -*-
'''
@author: Jason.F
@data: 2019.07.22
@function: Implementing BMF(Bayesian Matrix Factorization) By VI
           Dataset: Movielen Dataset(ml-1m) 
           Evaluating: hitradio,ndcg
'''
import sys
import time
import logging

import pymc3 as pm
import numpy as np
import pandas as pd
import theano
import theano.tensor as t
import heapq
import math

def getTraindata():
    data = []
    filePath = '/data/fjsdata/ctKngBase/ml/ml-1m.train.rating'
    u = 0
    i = 0
    maxr = 0.0
    with open(filePath, 'r') as f:
        for line in f:
            if line:
                lines = line[:-1].split("\t")
                user = int(lines[0])
                item = int(lines[1])
                score = float(lines[2])
                data.append((user, item, score))
                if user > u: u = user
                if item > i: i = item
                if score > maxr: maxr = score
    print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}\n"
                  "\tItem Num: {}\n"
                  "\tData Size: {}".format(u, i, len(data)))
    R = np.zeros([u+1, i+1], dtype=np.float32)
    for i in data:
        user = i[0]
        item = i[1]
        rating = i[2]
        R[user][item] = rating
    return R

def getTestdata():
    testset = []
    filePath = '/data/fjsdata/ctKngBase/ml/ml-1m.test.negative'
    with open(filePath, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            testset.append([u, eval(arr[0])[1]])#one postive item
            for i in arr[1:]:
                testset.append([u, int(i)]) #99 negative items
            line = fd.readline()
    return testset

def getHitRatio(ranklist, targetItem):
    for item in ranklist:
        if item == targetItem:
            return 1
    return 0
def getNDCG(ranklist, targetItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == targetItem:
            return math.log(2) / math.log(i+2)
    return 0

def build_BMF(R, K=8, alpha=2, std=0.01):
    
    alpha_u = 1 / R.var(axis=1).mean()
    alpha_v = 1 / R.var(axis=0).mean()

    logging.info('building the BMF model')
    n, m = R.shape
    with pm.Model() as bmf:
        U = pm.MvNormal('U', mu=0, tau=alpha_u * np.eye(K), shape=(n, K), testval=np.random.randn(n, K) * std)
        V = pm.MvNormal('V', mu=0, tau=alpha_v * np.eye(K), shape=(m, K), testval=np.random.randn(m, K) * std)
        nR = pm.Normal('nR', mu=t.dot(U, V.T), tau=alpha * np.ones(R.shape),observed=R)
    logging.info('done building BMF model')
    return bmf

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s')

    # Read data and build BMF model.
    R = getTraindata()
    bmf = build_BMF(R, K=8)#dim is the number of latent factors

    with bmf:# sample with BMF
        tstart = time.time()
        logging.info('Start BMF sampling')
        inference = pm.ADVI()
        approx = pm.fit(n=1000, method=inference)
        trace = approx.sample(draws=500)
        '''
        start = pm.find_MAP()
        step = pm.NUTS()
        trace = pm.sample(1000, step, start=start, njobs=8)
        '''
        elapsed = time.time() - tstart    
        logging.info('Complete BMF sampling in %d seconds' % int(elapsed))
        
    with bmf:#evaluation
        testset = getTestdata()
        ppc = pm.sample_posterior_predictive(trace, progressbar=True)
        nR = np.mean(ppc['nR'],0)#three dims, calcuate the mean with the first dim for posterior
        hits = []
        ndcgs = []
        prev_u = testset[0][0]
        pos_i = testset[0][1]
        scorelist = []
        for u, i in testset:
            if prev_u == u:
                scorelist.append([i,nR[u,i]])
            else:
                map_item_score = {}
                for item, rate in scorelist: #turn dict
                    map_item_score[item] = rate
                ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)#default Topn=10
                hr = getHitRatio(ranklist, pos_i)
                hits.append(hr)
                ndcg = getNDCG(ranklist, pos_i)
                ndcgs.append(ndcg)
                #next user
                scorelist = []
                prev_u = u
                pos_i = i
                scorelist.append([i,nR[u,i]])
        hitratio,ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print("hr: {}, NDCG: {}, At K {}".format(hitratio, ndcg, 8))
'''
HR@10: 0.10018214936247723, NDCG@10: 0.04508370537288355, At K 8
HR@10: 0.0991886073853287, NDCG@10: 0.04537995020828834, At K 16
HR@10: 0.09604239112435833, NDCG@10: 0.04273292643985209, At K 32
HR@10: 0.10200364298724955, NDCG@10: 0.04524941138321496, At K 64
'''
        