# -*- Encoding:UTF-8 -*-
'''
@author: Jason.F
@data: 2019.07.18
@function: Implementing PMF
           Dataset: Pinterest-20
           Evaluating: hitradio,ndcg
           https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf
           Matlab: http://www.utstat.toronto.edu/~rsalakhu/BPMF.html 
@reference: https://github.com/adamzjw/Probabilistic-matrix-factorization-in-Python
'''
import numpy as np
import pandas as pd
from numpy.random import RandomState
import copy
import heapq
import math
from numpy import linalg as LA
import random
#define class PMF
class PMF:
    def __init__(self, num_feat=8, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        self.num_feat = num_feat
        self.epsilon = epsilon
        self._lambda = _lambda
        self.momentum = momentum
        self.maxepoch = maxepoch
        self.num_batches = num_batches
        self.batch_size = batch_size
        
        self.w_C = None
        self.w_I = None

        self.err_train = []
        self.err_val = []
        
    def fit(self, train_vec, val_vec):   
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:,2])
        
        pairs_tr = train_vec.shape[0]
        pairs_va = val_vec.shape[0]
        
        # 1-p-i, 2-m-c
        num_inv = int(max(np.amax(train_vec[:,0]), np.amax(val_vec[:,0]))) + 1
        num_com = int(max(np.amax(train_vec[:,1]), np.amax(val_vec[:,1]))) + 1

        incremental = False
        if ((not incremental) or (self.w_C is None)):
            # initialize
            self.epoch = 0
            self.w_C = 0.1 * np.random.randn(num_com, self.num_feat)
            self.w_I = 0.1 * np.random.randn(num_inv, self.num_feat)
            
            self.w_C_inc = np.zeros((num_com, self.num_feat))
            self.w_I_inc = np.zeros((num_inv, self.num_feat))
        
        
        while self.epoch < self.maxepoch:
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])
            np.random.shuffle(shuffled_order)

            # Batch update
            for batch in range(self.num_batches):
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                batch_idx = np.mod(np.arange(self.batch_size * batch,
                                             self.batch_size * (batch+1)),
                                   shuffled_order.shape[0])

                batch_invID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_comID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_I[batch_invID,:], 
                                                self.w_C[batch_comID,:]),
                                axis=1) # mean_inv subtracted

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_C = 2 * np.multiply(rawErr[:, np.newaxis], self.w_I[batch_invID,:]) \
                        + self._lambda * self.w_C[batch_comID,:]
                Ix_I = 2 * np.multiply(rawErr[:, np.newaxis], self.w_C[batch_comID,:]) \
                        + self._lambda * self.w_I[batch_invID,:]
            
                dw_C = np.zeros((num_com, self.num_feat))
                dw_I = np.zeros((num_inv, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_C[batch_comID[i],:] += Ix_C[i,:]
                    dw_I[batch_invID[i],:] += Ix_I[i,:]


                # Update with momentum
                self.w_C_inc = self.momentum * self.w_C_inc + self.epsilon * dw_C / self.batch_size
                self.w_I_inc = self.momentum * self.w_I_inc + self.epsilon * dw_I / self.batch_size


                self.w_C = self.w_C - self.w_C_inc
                self.w_I = self.w_I - self.w_I_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(train_vec[:,0], dtype='int32'),:],
                                                    self.w_C[np.array(train_vec[:,1], dtype='int32'),:]),
                                        axis=1) # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = LA.norm(rawErr) ** 2 \
                            + 0.5*self._lambda*(LA.norm(self.w_I) ** 2 + LA.norm(self.w_C) ** 2)

                    self.err_train.append(np.sqrt(obj/pairs_tr))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(val_vec[:,0], dtype='int32'),:],
                                                    self.w_C[np.array(val_vec[:,1], dtype='int32'),:]),
                                        axis=1) # mean_inv subtracted
                    rawErr = pred_out - val_vec[:, 2] + self.mean_inv
                    self.err_val.append(LA.norm(rawErr)/np.sqrt(pairs_va))

                # Print info
                if batch == self.num_batches - 1:
                    print ('Training RMSE: %f, Val RMSE %f' % (self.err_train[-1], self.err_val[-1]))
    
    def predict(self, invID, comID): 
        return np.dot(self.w_C[comID,:], self.w_I[invID,:]) + self.mean_inv
               
    def evaluate(self, test_vec, k=10):
        def getHitRatio(ranklist, gtItem):
            for item in ranklist:
                if item == gtItem:
                    return 1
            return 0

        def getNDCG(ranklist, gtItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == gtItem:
                    return math.log(2) / math.log(i+2)
            return 0
        
        testset = pd.DataFrame(test_vec, columns=['u','i'])
        hits = []
        ndcgs = []
        list_csr = list(set(np.array(testset['u']).tolist()))
        for u in list_csr:
            csrset = np.array(testset[testset['u']==u]).tolist()
            scorelist = []
            positem = csrset[0][1]#first item is positive
            for _, i in csrset:
                scorelist.append([i, self.predict(u,i)])
            #get topk 
            map_item_score = {}
            for item, rate in scorelist: #turn dict
                map_item_score[item] = rate
            ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)#default Topn=10
            hr = getHitRatio(ranklist, positem)
            hits.append(hr)
            ndcg = getNDCG(ranklist, positem)
            ndcgs.append(ndcg)
        hitratio,ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hitratio,ndcg
        
#loading dataset
def getTrainset(filePath):
    trainset = []
    maxu = 0 
    maxi = 0 
    maxr = 0.0
    with open(filePath, 'r') as fd:
        line = fd.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i, rating = int(arr[0]), int(arr[1]), float(arr[2])
            trainset.append([int(arr[0]), int(arr[1]), float(arr[2])])
            if rating > maxr: maxr = rating
            if u > maxu: maxu = u
            if i > maxi: maxi = i
            line = fd.readline()
        return trainset, maxr, maxu, maxi

def getTestset(filePath):
    testset = []
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

def getTrainDict(data):
        dataDict = {}
        for i in data:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict
    
def getNegTrain(data, maxi, negNum=4):
        datadict = getTrainDict(data)
        trainneg = []
        for i in data:
            trainneg.append([i[0],i[1],i[2]])
            for t in range(negNum):
                j = np.random.randint(maxi)
                while (i[0], j) in datadict:
                    j = np.random.randint(maxi)
                trainneg.append([i[0], j, 0.0])
        return trainneg
    
if __name__ == '__main__':
    trainset, maxr, maxu, maxi = getTrainset("/data/fjsdata/ctKngBase/ml/pinterest-20.train.rating")
    trainneg = getNegTrain(trainset, maxi)
    testset = getTestset("/data/fjsdata/ctKngBase/ml/pinterest-20.test.negative")
    print('Dataset Statistics: Interaction = %d, User = %d, Item = %d, Sparsity = %.4f' % (len(trainset), maxu+1, maxi+1, len(trainset)/(maxu*maxr)))

    print ("%3s%20s%20s" % ('K','HR@10', 'NDCG@10'))
    for K in [8, 16, 32, 64]:
        pmf = PMF(num_feat=K)
        valtest = random.sample(trainset,int(0.2*len(trainset)))
        pmf.fit(np.array(trainneg), np.array(valtest))
        hit, ndcg = pmf.evaluate(testset)
        print ("%3d%20.6f%20.6f" % (K, hit, ndcg))
'''
nohup python -u PMF-pe.py > pmf-pe.log  &
 K               HR@10             NDCG@10
8            0.100404            0.045550
16            0.100060            0.045315
32            0.099154            0.044934
64            0.101238            0.045676
'''