# -*- Encoding:UTF-8 -*-
'''
@author: Jason.F
@data: 2019.07.29
@function: BNMF(Bayesian Neural Matrix Factorization) 
           Dataset: Movielen Dataset(ml-1m) 
           Evaluating: hitradio,ndcg
'''
import sys
import time
import heapq
import math
import gc
import numpy as np
import pandas as pd

import pymc3 as pm
import theano
import tensorflow as tf
import theano.tensor as tt

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
    
class BNMF:
    def __init__(self, dataset, K=8):
        self.shape = [dataset.maxu, dataset.maxi]
        #get the trainset and testset
        self.train_u, self.train_i, self.train_r = dataset.getInstances(isTest=False)
        assert(len(self.train_u) == len(self.train_i) and len(self.train_i) == len(self.train_r))
        shuffled_idx = np.random.permutation(np.arange(len(self.train_u)))
        self.train_u = self.train_u[shuffled_idx]
        self.train_i = self.train_i[shuffled_idx]
        self.train_r = self.train_r[shuffled_idx]
        self.test_u, self.test_i, self.test_r = dataset.getInstances(isTest=True)
        assert(len(self.test_u) == len(self.test_i) and len(self.test_i) == len(self.test_r))
        
        #initialize
        #K is number of latent factors
        self.userLayer = [512, K]
        self.itemLayer = [512, K]
        self.batchSize = 1024
        self.lr = 0.001
        tf.reset_default_graph()
        self._build_MLP()
        self._init_sess()
        
    def _init_sess(self):
        #define seesion
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer(), feed_dict={self.R: dataset.list_to_matrix()})
        
    def _build_MLP(self):
        print('start building the Multi-layer non-linear projection')
        # add placeholder
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)
        self.R = tf.placeholder(tf.float32, shape=(self.shape[0], self.shape[1]))
        user_item_embedding = tf.convert_to_tensor(tf.Variable(self.R))
        item_user_embedding = tf.transpose(user_item_embedding)
        user_input = tf.nn.embedding_lookup(user_item_embedding, self.user)
        item_input = tf.nn.embedding_lookup(item_user_embedding, self.item)
        
        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer)-1):
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.itemLayer)-1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))
           
        self.r_ui = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keepdims=False)
        self.loss = tf.reduce_sum(tf.losses.mean_squared_error(labels = self.rate, predictions=self.r_ui))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)
        print('done building the the Multi-layer non-linear projection')
        
    def _build_BPF(self):
        print('start building the Bayesian probabilistic model')  
        self.x_u = theano.shared(self.train_u)
        self.x_i = theano.shared(self.train_i)
        self.y_r = theano.shared(self.train_r)
        self.y_r_ui = theano.shared(np.array(self.nn_r_ui))
        assert(len(self.y_r.get_value())==len(self.y_r_ui.get_value()))
        with pm.Model() as self.bncf:#define the prior and likelihood
            b_u = pm.Normal('b_u', 0, sd=1, shape=self.shape[0])
            b_i = pm.Normal('b_i', 0, sd=1, shape=self.shape[1])
            u = pm.Normal('u', 0, sd=1)
            tY = pm.Deterministic('tY', tt.add(tt.add(tt.add(b_u[self.x_u],b_i[self.x_i]),self.y_r_ui),u))
            #tY = pm.Deterministic('tY', ((b_u[self.x_u]+b_i[self.x_i])+self.y_r_ui)+u)#b_u+b_i+u+nn_r_ui
            nY = pm.Deterministic('nY', pm.math.sigmoid(tY))
            # likelihood of observed data
            Y = pm.Bernoulli('Y', nY, observed=self.y_r)#total_size=self.y_r.get_value().shape[0]
        with self.bncf:#inference
            approx = pm.fit(n=1000, method=pm.ADVI())
            self.trace = approx.sample(draws=500)
        with self.bncf: #posterior prediction
            ppc = pm.sample_posterior_predictive(self.trace, progressbar=True)
            self.by_r_ui = ppc['Y'].mean(axis=0)
        print('done building the Bayesian probabilistic model')
        
    def train_BNMF(self, verbose=10):       
        print('start training the BNCF model')
        tstart = time.time()
        
        num_batches = len(self.train_u) // self.batchSize + 1
        #1.traing r_ui in neural network 
        self.nn_r_ui=[]
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([len(self.train_u), (i+1)*self.batchSize])
            train_u_batch = self.train_u[min_idx: max_idx]
            train_i_batch = self.train_i[min_idx: max_idx]
            #train_r_batch = self.train_r[min_idx: max_idx]
            pre_r_ui_batch = self.sess.run(self.r_ui, feed_dict={self.user: train_u_batch, \
                                                                 self.item: train_i_batch})
            self.nn_r_ui.extend(pre_r_ui_batch)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : shape = {} '.format(i, num_batches, len(self.nn_r_ui)))
                sys.stdout.flush()
        #2.training bias in Bayesian inference
        self._build_BPF()
        #3.training self.loss in neural network
        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([len(self.train_u), (i+1)*self.batchSize])
            train_u_batch = self.train_u[min_idx: max_idx]
            train_i_batch = self.train_i[min_idx: max_idx]
            train_r_batch = self.by_r_ui[min_idx: max_idx]
            _, tmp_loss = self.sess.run([self.train_step, self.loss], feed_dict={self.user: train_u_batch, \
                                                                                 self.item: train_i_batch, \
                                                                                 self.rate: train_r_batch})
            losses.append(tmp_loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(i, num_batches, np.mean(losses[-verbose:])))
                sys.stdout.flush()
        loss = np.mean(losses)  
        elapsed = time.time() - tstart    
        print('Completed training the BNCF model in %d seconds' % int(elapsed))
        return loss
           
    def eval_BNMF(self, verbose=10):
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
    
        #1.get r_ui in neural network
        num_batches = len(self.test_u) // self.batchSize + 1
        #1.traing r_ui in neural network 
        self.nn_r_ui=[]
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([len(self.test_u), (i+1)*self.batchSize])
            test_u_batch = self.test_u[min_idx: max_idx]
            test_i_batch = self.test_i[min_idx: max_idx]
            pre_r_ui_batch = self.sess.run(self.r_ui, feed_dict={self.user: test_u_batch, \
                                                                 self.item: test_i_batch})
            self.nn_r_ui.extend(pre_r_ui_batch)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : shape = {} '.format(i, num_batches, len(self.nn_r_ui)))
                sys.stdout.flush()
                
        #2. get biais in Bayesian inference
        self.x_u.set_value(self.test_u)
        self.x_i.set_value(self.test_i)
        self.y_r.set_value(self.test_r)
        self.y_r_ui.set_value(self.nn_r_ui)
        with self.bncf:#evaluation
            ppc = pm.sample_posterior_predictive(self.trace, progressbar=True) 
            pre_r = ppc['Y'].mean(axis=0)
        assert(pre_r.shape[0]==self.test_i.shape[0])
        #every user have one positive item and 99 negative items
        num_batches = len(self.test_r) // 100
        hits = []
        ndcgs = []
        for i in range(num_batches):
            test_i_batch = self.test_i[i*100: (i+1)*100]
            pre_r_batch = pre_r[i*100: (i+1)*100]
            map_item_score = {}
            for j in range(100):
                map_item_score[test_i_batch[j]] = pre_r_batch[j]
            ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)#default Topn=10
            hits.append(getHitRatio(ranklist, test_i_batch[0]))
            ndcgs.append(getNDCG(ranklist, test_i_batch[0]))
        hit, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hit, ndcg
    
if __name__ == "__main__":
    for fileName in ['pinterest-20', 'kb-cc']:#['ml-1m', 'pinterest-20', 'kb-cc']:
        dataset = DataSet(fileName=fileName, negNum=4)#loading dataset
        for K in [8, 16, 32, 64]:
            model = BNMF(dataset, K)
            best_hr = 0.0
            best_ndcg = 0.0
            for epoch in range(2):
                loss = model.train_BNMF()
                print("\nMean loss in this epoch is: {}".format(loss))
                hit, ndcg = model.eval_BNMF()
                print("HR@10: {}, NDCG@10: {}, At K {} and Dataset {}".format(hit, ndcg, K, fileName ))
                if hit>best_hr: best_hr=hit
                if ndcg>best_ndcg: best_ndcg=ndcg
            print("Best HR@10: {}, Best NDCG@10: {}, At K {} and Dataset {}".format(best_hr, best_ndcg, K, fileName ))
            
'''
Best HR@10: 0.4197019867549669, Best NDCG@10: 0.22775834012977136, At K 8 and Dataset ml-1m
Best HR@10: 0.41721854304635764, Best NDCG@10: 0.2251975547330144, At K 16 and Dataset ml-1m
Best HR@10: 0.4197019867549669, Best NDCG@10: 0.22811742858950668, At K 32 and Dataset ml-1m
Best HR@10: 0.42566225165562915, Best NDCG@10: 0.2340599456026715, At K 64 and Dataset ml-1m
HR@10: 0.2364143729501513, NDCG@10: 0.12314803893484755, At K 8 and Dataset pinterest-20
'''