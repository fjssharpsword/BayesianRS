# coding:utf-8  
'''
@author: Jason.F
@data: 2019.07.15
@function: Implementation: SVDBias 
           Datatset: KnowledgeBase-CC  
           Evaluation: hitradio,ndcg
           Squared loss function with implicit feedback.
'''
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import heapq
import random

#1. loading the KnowledgeBase dataset.
trainset = pd.read_csv("./data/kbcc_trainset.csv", sep='|', low_memory=False, dtype={'csr':int, 'ke':int, 'num':float})
trainset['num']=trainset['num'].apply(lambda x: 1 if float(x)>0.0 else 0)
print ('Trainset shape is:%d rows and %d columns'%(trainset.shape[0],trainset.shape[1]))
#testset includes 100 items for every user, one item is positive and other 99 is negtive items.
testset = pd.read_csv("./data/kbcc_testset.csv", sep='|', low_memory=False, dtype={'csr':int, 'ke':int, 'num':float})
testset['num']=testset['num'].apply(lambda x: 1 if float(x)>0.0 else 0)
print ('Testset shape is:%d rows and %d columns'%(testset.shape[0],testset.shape[1]))
csrNum = trainset['csr'].max()+1
keNum = trainset['ke'].max()+1
print('Dataset Statistics: Interaction = %d, User = %d, Item = %d, Sparsity = %.4f' % 
      (trainset.shape[0], csrNum, keNum, trainset.shape[0]/(csrNum*keNum)) )

#2. SVDBias class
class SVDBias():
    
    def __init__(self, R, num_ng=4):
        """
        Perform matrix factorization to predict empty entries in a matrix.     
        Arguments
        - R (ndarray)   : user-item rating matrix
        - num_ng (int)  : number of negative items
        """
        self.R = R
        self.num_users, self.num_items = R.shape
        self.num_ng = num_ng
        
        # Create a list of training samples
        pos_samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        #smapling the negative items
        neg_samples = random.sample([
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] == 0
        ], len(pos_samples)*num_ng)
        
        self.samples = pos_samples + neg_samples

    def train(self, K, alpha=0.001, beta=0.01, epochs=20):
        '''
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        - K (int)       : number of latent dimensions
        -epochs(int)    : number of iterations
        '''
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
               
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.epochs):
            np.random.shuffle(self.samples)
            self.sgd()
            #if (i+1) % 10 == 0:
            #    mse = self.mse()
            #    print("Iteration: %d ; error = %.4f" % (i+1, mse))
        
        return self.full_matrix()

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
    
#3. Training and Evaluating
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

R = np.zeros((csrNum, keNum))
for _, row in trainset.iterrows(): 
    R[int(row['csr'])][int(row['ke'])] = float(row['num'])
print ("%3s%20s%20s" % ('K','HR@10', 'NDCG@10'))
mdl = SVDBias(R=R, num_ng=4)# K is latent factors
for K in [8,16,32,64]:#iterations epoches
    nR = mdl.train(K=K, alpha=0.001, beta=0.01, epochs=20)
    hits = []
    ndcgs = []
    list_csr = list(set(np.array(testset['csr']).tolist()))
    for csr in list_csr:
        csrset = testset[testset['csr']==csr]
        scorelist = []
        positem = 0
        for u, i, r in np.array(csrset).tolist():   
            if float(r)>0.0:#one positive item
                scorelist.append([int(i),nR[int(u),int(i)]])
                positem = int(i) 
            else:# 99 negative items
                scorelist.append([int(i),nR[int(u),int(i)]])
        map_item_score = {}
        for item, rate in scorelist: #turn dict
            map_item_score[item] = rate
        ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)#default Topn=10
        hr = getHitRatio(ranklist, positem)
        hits.append(hr)
        ndcg = getNDCG(ranklist, positem)
        ndcgs.append(ndcg)
    hitratio,ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print ("%3d%20.6f%20.6f" % (K, hitratio, ndcg))
    
'''
nohup python -u SVDBias-kb.py > svdbias-kb.log  &
Trainset shape is:2547452 rows and 3 columns
Testset shape is:1021600 rows and 3 columns
Dataset Statistics: Interaction = 2547452, User = 10216, Item = 96324, Sparsity = 0.0026
  K               HR@10             NDCG@10
  8            0.788469            0.547458
 16            0.787980            0.546995
 32            0.788078            0.547242
 64            0.787392            0.547680
    
'''