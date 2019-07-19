# coding:utf-8  
'''
@author: Jason.F
@data: 2019.07.15
@function: Implementation: IRT-MF 
           Datatset: Pinterest-20 
           Evaluation: hitradio,ndcg
'''
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import heapq
import random

#1.Loading the  MovienLen dataset, ml-1m
def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList
def load_negative_file_as_list(filename):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1: ]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList
def load_rating_file_as_matrix(filename):
    #Read .rating file and Return dok matrix.
    #The first line of .rating file is: num_users\t num_items
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    # Construct matrix
    #mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
    mat = np.zeros((num_users+1, num_items+1))
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            mat[user, item] = rating
            line = f.readline()    
    return mat
trainMatrix = load_rating_file_as_matrix("./data/pinterest-20.train.rating")
testRatings = load_rating_file_as_list("./data/pinterest-20.test.rating")
testNegatives = load_negative_file_as_list("/data/fjsdata/ctKngBase/ml/pinterest-20.test.negative")
print('Dataset Statistics: Interaction = %d, User = %d, Item = %d, Sparsity = %.4f' % \
      (len(trainMatrix[np.where(trainMatrix!= 0)]),trainMatrix.shape[0],trainMatrix.shape[1],\
       len(trainMatrix[np.where(trainMatrix!= 0)])/(trainMatrix.shape[0]*trainMatrix.shape[1]) ))

#2. IRT-MF class
class IRTMF():
    
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
        self.a_u = np.zeros(self.num_users)
        self.d_i = np.zeros(self.num_items)
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
            prob = self.IRT_Rasch(self.a_u[i], self.d_i[j])
            e = (r - prediction-prob)
            
            # Update biases
            
            self.a_u[i] += self.alpha * (e * prob*(prob-1) - self.beta * self.a_u[i])
            self.d_i[j] += self.alpha * (e * prob*(1-prob) - self.beta * self.d_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.IRT_Rasch(self.a_u[i], self.d_i[j]) + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        nR = np.zeros((self.num_users, self.num_items))
        for i in range(self.num_users):
            for j in range(self.num_items):
                nR[i,j] = self.IRT_Rasch(self.a_u[i], self.d_i[j]) + self.P[i, :].dot(self.Q[j, :].T)
        return nR
    
    def IRT_Rasch(self, a, d):
        """
        Compute the probability of respons
        """
        x = a-d
        if x < 0:
            return 1 - 1/(1 + math.exp(x))
        else:
            return 1/(1 + math.exp(-x))
        
#3. training and evaluating 
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

print ("%3s%20s%20s" % ('K','HR@10', 'NDCG@10'))
mdl = IRTMF(R=trainMatrix, num_ng=4)# K is latent factors
for K in [8,16,32,64]:#latent factors
    nR = mdl.train(K=K, alpha=0.001, beta=0.01, epochs=20)
    hits = []
    ndcgs = []
    for u, i in testRatings:
        scorelist= [ [ni,nR[u,ni]] for ni in testNegatives[u]]
        scorelist.append([i,nR[u,i]])
        map_item_score = {}
        for item, rate in scorelist: #turn dict
            map_item_score[item] = rate
        ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)#default Topn=10
        hr = getHitRatio(ranklist, i)
        hits.append(hr)
        ndcg = getNDCG(ranklist, i)
        ndcgs.append(ndcg)
    hitratio,ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print ("%3d%20.6f%20.6f" % (K, hitratio, ndcg))
    
'''
nohup python -u IRT-MF-pe.py > irtmf-pe.log  &
Dataset Statistics: Interaction = 1408394, User = 55187, Item = 9916, Sparsity = 0.0026
  K               HR@10             NDCG@10
  8            0.251019            0.129946
 16            0.250856            0.130591
 32            0.236070            0.125136
 64            0.136953            0.068325
    
'''