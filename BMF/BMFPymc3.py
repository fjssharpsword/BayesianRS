# -*- Encoding:UTF-8 -*-
'''
@author: Jason.F
@data: 2019.07.28
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
import theano.tensor as tt
import heapq
import math

class DataSet:
    def __init__(self, fileName, negNum):
        self.negNum = negNum #negative sample ratio
        self.trainList, self.maxu, self.maxi = self.getTrainset_as_list(fileName)
        self.testList = self.getTestset_as_list(fileName)
        
    def getTrainset_as_list(self, fileName):
        if (fileName == 'ml-1m') or (fileName == 'pinterest-20'):
            filePath = "/data/fjsdata/ctKngBase/ml/"+fileName+".train.rating" 
            data = pd.read_csv(filePath, sep='\t', header=None, names=['user', 'item', 'rating'], \
                                 usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.float})
            data['rating']=data['rating'].apply(lambda x: 1.0 if float(x)>0.0 else 0.0)
            maxu, maxi = data['user'].max()+1, data['item'].max()+1
            print('Dataset Statistics: Interaction = %d, User = %d, Item = %d, Sparsity = %.4f' % \
                  (data.shape[0], maxu, maxi, data.shape[0]/(maxu*maxi)))
            dataList = data.values.tolist()
            return dataList, maxu, maxi
        if (fileName == 'kb-cc'):
            filePath = "/data/fjsdata/ctKngBase/kbcc_trainset.csv"
            data = pd.read_csv(filePath, sep='|', low_memory=False, dtype={'csr':int, 'ke':int, 'num':float})
            data['num']=data['num'].apply(lambda x: 1.0 if float(x)>0.0 else 0.0)
            maxu, maxi = data['user'].max()+1, data['item'].max()+1
            print('Dataset Statistics: Interaction = %d, User = %d, Item = %d, Sparsity = %.4f' % \
                  (data.shape[0], maxu, maxi, data.shape[0]/(maxu*maxi)))
            dataList = data.values.tolist()
            return dataList, maxu, maxi
    
    def getTestset_as_list(self, fileName):
        if (fileName == 'ml-1m') or (fileName == 'pinterest-20'):
            filePath = "/data/fjsdata/ctKngBase/ml/"+fileName+".test.negative" 
            dataList = []
            with open(filePath, 'r') as fd:
                line = fd.readline()
                while line != None and line != '':
                    arr = line.split('\t')
                    u = eval(arr[0])[0]
                    dataList.append([u, eval(arr[0])[1], 1.0])#first is one postive item
                    for i in arr[1:]:
                        dataList.append([u, int(i), 0.0]) #99 negative items
                    line = fd.readline()
            return dataList
        if (fileName == 'kb-cc'):
            filePath = "/data/fjsdata/ctKngBase/kbcc_testset.csv"
            data = pd.read_csv(filePath, sep='|', low_memory=False, dtype={'csr':int, 'ke':int, 'num':float})
            data['num']=data['num'].apply(lambda x: 1.0 if float(x)>0.0 else 0.0)
            dataList = data.values.tolist()
            return dataList
        
    def list_to_matrix(self):              
        dataMat = np.zeros([self.maxu, self.maxi], dtype=np.float32)
        for u,i,r in self.trainList:
            dataMat[int(u)][int(i)] = float(r)
        return np.array(dataMat)
    
    def list_to_dict(self):
        dataDict = {}
        for u,i,r in self.trainList:
            dataDict[int(u), int(i)] = float(r)
        return dataDict
    
    def getInstances(self, isTest=False):
        user = []
        item = []
        rate = []
        if isTest==True: #test
            for u, i, r in self.testList:
                user.append(int(u))
                item.append(int(i))
                rate.append(float(r))
        else:#train
            for u, i, r in self.trainList:
                user.append(int(u))
                item.append(int(i))
                rate.append(float(r))
            #negative samples
            dataDict = self.list_to_dict()
            for j in range(len(self.trainList)*self.negNum):
                u = np.random.randint(self.maxu)
                i = np.random.randint(self.maxi)
                while (u, i) in dataDict:
                    u = np.random.randint(self.maxu)
                    i = np.random.randint(self.maxi)
                user.append(int(u))
                item.append(int(i))
                rate.append(float(0.0)) 
        return np.array(user), np.array(item), np.array(rate)
    
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s')
    for fileName in ['ml-1m', 'pinterest-20', 'kb-cc']:
        dataset = DataSet(fileName=fileName, negNum=4)#loading dataset
        #get the trainset and testset
        train_u, train_i, train_r = dataset.getInstances(isTest=False)
        assert(len(train_u) == len(train_i) and len(train_i) == len(train_r)) 
        shuffled_idx = np.random.permutation(np.arange(len(train_u)))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]
        test_u, test_i, test_r = dataset.getInstances(isTest=True)
        assert(len(test_u) == len(test_i) and len(test_i) == len(test_r))
        #R = dataset.list_to_matrix()
        for K in [8, 16, 32, 64]:
            x_u = theano.shared(train_u)
            x_i = theano.shared(train_i)
            y_r = theano.shared(train_r)
            with pm.Model() as bmf:#bulid probabilistic model
                # Creating the model
                P = pm.Normal('P', mu=0, sd=1, shape=(dataset.maxu,K))
                Q = pm.Normal('Q', mu=0, sd=1, shape=(dataset.maxi,K))
                #R = pm.Deterministic('R', tt.dot(P,Q))#pm.math.dot
                #tY = pm.Deterministic('tY ',[R[x_u[j]][x_i[j]] for j in range(y_r.eval().shape[0])])
                #tY =  pm.Deterministic('tY', pm.math.sum(tt.mul(P[x_u,:].T,Q[x_i,:].T), axis=1, keepdims=True))
                tY = pm.Deterministic('tY', pm.math.sum(P[x_u,:]*Q[x_i,:], axis=1))
                nY = pm.Deterministic('nY', pm.math.sigmoid(tY))
                # likelihood of observed data
                Y = pm.Bernoulli('Y', nY, observed=y_r)#total_size=y_r.eval().shape[0]
                
            with bmf: #train the probabilistic model by Bayesian inference
                tstart = time.time()
                logging.info('Start BMF sampling')
                approx = pm.fit(n=1000, method=pm.ADVI())
                trace = approx.sample(draws=500)
                elapsed = time.time() - tstart 
                logging.info('Complete BMF sampling in %d seconds' % int(elapsed))
                
            x_u.set_value(test_u)
            x_i.set_value(test_i)
            y_r.set_value(test_r)
            with bmf:
                ppc = pm.sample_posterior_predictive(trace, progressbar=True)
                pre_r = ppc['Y'].mean(axis=0)
            assert(pre_r.shape[0]==test_i.shape[0])
            #every user have one positive item and 99 negative items
            num_batches = len(test_r) // 100
            hits = []
            ndcgs = []
            for i in range(num_batches):
                test_i_batch = test_i[i*100: (i+1)*100]
                pre_r_batch = pre_r[i*100: (i+1)*100]
                map_item_score = {}
                for j in range(100):
                    map_item_score[test_i_batch[j]] = pre_r_batch[j]
                ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)#default Topn=10
                hits.append(getHitRatio(ranklist, test_i_batch[0]))
                ndcgs.append(getNDCG(ranklist, test_i_batch[0]))
            hit, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print("HR@10: {}, NDCG@10: {}, At K {}".format(hit, ndcg, K))
            
'''
HR@10: 0.10860927152317881, NDCG@10: 0.048635873105124044, At K 8
HR@10: 0.10894039735099338, NDCG@10: 0.050145891288496384, At K 16
'''