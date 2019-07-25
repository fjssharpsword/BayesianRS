# -*- Encoding:UTF-8 -*-
'''
@author: Jason.F
@data: 2019.07.23
@function: Implementing BMF(Bayesian Neural Collaborative Filtering) which is designed by Jason.F
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
import theano.tensor as T
import tensorflow as tf
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
            maxu, maxi = data['user'].max()+1, data['item'].max()+1
            print('Dataset Statistics: Interaction = %d, User = %d, Item = %d, Sparsity = %.4f' % \
                  (data.shape[0], maxu, maxi, data.shape[0]/(maxu*maxi)))
            dataList = data.values.tolist()
            return dataList, maxu, maxi
        if (fileName == 'kb-cc'):
            filePath = "/data/fjsdata/ctKngBase/kbcc_trainset.csv"
            data = pd.read_csv(filePath, sep='|', low_memory=False, dtype={'csr':int, 'ke':int, 'num':float})
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
            dataList = data.values.tolist()
            return dataList
        
    def list_to_matrix(self):              
        dataMat = np.zeros([self.maxu, self.maxi], dtype=np.float32)
        for u,i,r in self.trainList:
            dataMat[int(u)][int(i)] = float(1.0)#float(r)
        return np.array(dataMat)
    
    def list_to_dict(self):
        dataDict = {}
        for u,i,r in self.trainList:
            dataDict[int(u), int(i)] = float(1.0)#float(r)
        return dataDict
    
    def getInstances(self, isTest=False):
        #dataMat = self.list_to_matrix(self.trainList, self.maxu, self.maxi)
        user = []
        item = []
        rate = []
        if isTest==True: #test
            for u, i, r in self.testList:
                user.append(int(u))#user.append(dataMat[int(u),:].tolist())
                item.append(int(i))#item.append(dataMat[:,int(i)].tolist())
                rate.append(1.0)#rate.append(float(r))
        else:#train
            for u, i, r in self.trainList:
                user.append(int(u))#user.append(dataMat[int(u),:].tolist())
                item.append(int(i))#item.append(dataMat[:,int(i)].tolist())
                rate.append(1.0)#rate.append(float(r))
            #negative samples
            dataDict = self.list_to_dict()
            for j in range(len(self.trainList)*self.negNum):
                u = np.random.randint(self.maxu)
                i = np.random.randint(self.maxi)
                while (u, i) in dataDict:
                    u = np.random.randint(self.maxu)
                    i = np.random.randint(self.maxi)
                user.append(int(u))#user.append(dataMat[u,:].tolist())
                item.append(int(i))#item.append(dataMat[:,i].tolist())
                rate.append(0.0) 
        return np.array(user), np.array(item), np.array(rate)
    
    def getHitRatio(self, ranklist, targetItem):
        for item in ranklist:
            if item == targetItem:
                return 1
        return 0
    
    def getNDCG(self, ranklist, targetItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == targetItem:
                return math.log(2) / math.log(i+2)
        return 0
    
class BNCF:
    def __init__(self, dataset):
        self.dataset = dataset
        #self.trainList = self.dataset.trainList
        #self.testList = self.dataset.testList
        self.maxu = self.dataset.maxu
        self.maxi = self.dataset.maxi
        
        #get the test data
        self.test_u, self.test_i, self.test_r = self.dataset.getInstances(isTest=True)
        #get the training data and setting the input 
        self.train_u, self.train_i, self.train_r = self.dataset.getInstances(isTest=False)
        assert(self.train_u.shape == self.train_i.shape and self.train_i.shape == self.train_r.shape)
        #initiate the seesion
        self.init_sess()
        #train data by mini-batch
        self.arrUser, self.arrItem = self.batchTrainset()
       
    def init_sess(self):
        tf.reset_default_graph()
        
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)
        self.R = tf.placeholder(tf.float32, shape=(self.maxu, self.maxi))
        #embedding layer
        self.user_item_embedding = tf.convert_to_tensor(self.R)
        self.item_user_embedding = tf.transpose(self.user_item_embedding)
        self.user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
        self.item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)
        #define seesion
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())
    
    def batchTrainset(self):
        #get input in the way of mini-batch
        batchSize=8192
        num_batches = len(self.train_u) // batchSize + 1
        train_u_batch = self.train_u[0: batchSize]
        train_i_batch = self.train_i[0: batchSize]
        arrUser, arrItem = self.sess.run([self.user_input, self.item_input], \
                                             feed_dict={self.user: train_u_batch, \
                                                        self.item: train_i_batch, \
                                                        self.R: self.dataset.list_to_matrix()})
        '''
        for i in range(1, num_batches):
            min_idx = i * batchSize
            max_idx = np.min([len(self.train_u), (i+1)*batchSize])
            train_u_batch = self.train_u[min_idx: max_idx]
            train_i_batch = self.train_i[min_idx: max_idx]
            train_r_batch = self.train_r[min_idx: max_idx]
            batchUser, batchItem = self.sess.run([self.user_input, self.item_input], \
                                             feed_dict={self.user: train_u_batch, \
                                                        self.item: train_i_batch, \
                                                        self.R: self.dataset.list_to_matrix()})
            arrUser = np.concatenate((arrUser, batchUser),axis=0)
            arrItem = np.concatenate((arrItem, batchItem),axis=0)
        '''
        return arrUser, arrItem
    
    
    def build_BNCF(self, K = 8):
        layers = [1024, K] #number of latent factors
        logging.info('start building the BNCF model')
        
        self.x_u = theano.shared(self.arrUser)
        self.x_i = theano.shared(self.arrItem)
        self.y_r = theano.shared(self.train_r)       
        with pm.Model() as self.bncf:
            #user layer
            user_W1 = pm.Normal('user_W1', 0, sd=1, shape=[self.maxi, layers[0]] )
            user_O1 = pm.math.tanh(pm.math.dot(self.x_u, user_W1))
            user_W2 = pm.Normal('user_W2', 0, sd=1, shape=[layers[0],layers[1]] )
            user_O2 = pm.math.tanh(pm.math.dot(user_O1, user_W2))
            #item layer
            item_W1 = pm.Normal('item_W1', 0, sd=1, shape=[self.maxu, layers[0]] )
            item_O1 = pm.math.tanh(pm.math.dot(self.x_i, item_W1))
            item_W2 = pm.Normal('item_W2', 0, sd=1, shape=[layers[0],layers[1]] )
            item_O2 = pm.math.tanh(pm.math.dot(item_O1, item_W2))
            #output layer
            #act_out = pm.math.sigmoid(pm.math.dot(user_O2, item_O2.T))
            #act_out = pm.math.sigmoid(np.sum(np.multiply(user_O2, item_O2),axis=1, keepdims=True))
            act_out = pm.math.sigmoid(np.multiply(user_O2, item_O2).sum(axis=1, keepdims=True))
            #act_out = pm.math.sigmoid(tf.reduce_sum(tf.multiply(user_O2, item_O2),axis=1, keep_dims=True))
            # Binary classification -> Bernoulli likelihood
            r = pm.Bernoulli('r', act_out, observed=self.y_r, total_size=self.y_r.shape[0]) # IMPORTANT for minibatches                         
        logging.info('done building BNCF model')
                
    def train_BNCF(self):
        logging.info('start training the BNCF model')
        tstart = time.time()
        with self.bncf:
            inference = pm.ADVI()
            approx = pm.fit(n=1000, method=inference)
            self.trace = approx.sample(draws=500)       
        elapsed = time.time() - tstart    
        logging.info('Completed training the BNCF model in %d seconds' % int(elapsed))
           
    def evaluate_BNCF(self):
        arrUser, arrItem = self.sess.run([self.user_input, self.item_input], \
                               feed_dict={self.user: self.test_u, \
                                          self.item: self.test_i, \
                                          self.R: self.dataset.list_to_matrix()})
        self.x_u.set_value(arrUser)
        self.x_i.set_value(arrItem)
        self.y_r.set_value(self.test_r)
        with self.bncf:#evaluation
            ppc = pm.sample_posterior_predictive(self.trace, progressbar=True)
            pre_r = ppc['r'].mean(axis=0)

            hits = []
            ndcgs = []
            prev_u = self.dataset.testList[0][0]
            pos_i = self.dataset.testList[0][1]
            scorelist = []
            iLen = 0
            for u, i in self.dataset.testList:
                if prev_u == u:
                    scorelist.append([i,pre_r[iLen]])
                else:
                    map_item_score = {}
                    for item, rate in scorelist: #turn dict
                        map_item_score[item] = rate
                    ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)#default Topn=10
                    hits.append(self.dataset.getHitRatio(ranklist, pos_i))
                    ndcgs.append(self.dataset.getNDCG(ranklist, pos_i))
                    #next user
                    scorelist = []
                    prev_u = u
                    pos_i = i
                    scorelist.append([i,pre_r[iLen]])
                iLen = iLen + 1
            hit, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            return hit, ndcg
        
if __name__ == "__main__":
    for fileName in ['ml-1m', 'pinterest-20', 'kb-cc']:
        dataset = DataSet(fileName=fileName, negNum=4)#loading dataset
        model = BNCF(dataset)
        for K in [8, 16, 32, 64]:
            model.build_BNCF(K)
            model.train_BNCF()
            hit, ndcg = model.evaluate_BNCF()
            print("HR@10: {}, NDCG@10: {}, At K {} and Dataset {}".format(hit, ndcg, K, fileName ))